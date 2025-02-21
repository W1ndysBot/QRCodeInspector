# script/QRCodeInspector/main.py

import logging
import os
import sys
import re
import json
import os
import cv2
import numpy as np
import threading
import _thread
import requests
import aiohttp
import asyncio
from functools import partial

# 添加项目根目录到sys.path
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from app.config import *
from app.api import *
from app.switch import load_switch, save_switch


# 数据存储路径，实际开发时，请将QRCodeInspector替换为具体的数据存放路径
DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data",
    "QRCodeInspector",
)


# 查看功能开关状态
def load_function_status(group_id):
    return load_switch(group_id, "QRCodeInspector")


# 保存功能开关状态
def save_function_status(group_id, status):
    save_switch(group_id, "QRCodeInspector", status)


# 处理元事件，用于启动时确保数据目录存在
async def handle_QRCodeInspector_meta_event(websocket):
    os.makedirs(DATA_DIR, exist_ok=True)


# 处理开关状态
async def toggle_function_status(websocket, group_id, message_id, authorized):
    if not authorized:
        await send_group_msg(
            websocket,
            group_id,
            f"[CQ:reply,id={message_id}]❌❌❌你没有权限对QRCodeInspector功能进行操作,请联系管理员。",
        )
        return

    if load_function_status(group_id):
        save_function_status(group_id, False)
        await send_group_msg(
            websocket,
            group_id,
            f"[CQ:reply,id={message_id}]🚫🚫🚫QRCodeInspector功能已关闭",
        )
    else:
        save_function_status(group_id, True)
        await send_group_msg(
            websocket,
            group_id,
            f"[CQ:reply,id={message_id}]✅✅✅QRCodeInspector功能已开启",
        )


def check_image_quality(image):
    """检查图像质量"""
    try:
        # 检查图像是否为空
        if image is None or image.size == 0:
            return False

        # 检查亮度
        brightness = np.mean(image)
        if brightness < 30 or brightness > 225:
            logging.info(f"图像亮度异常: {brightness}")
            return False

        # 检查对比度
        contrast = image.std()
        if contrast < 20:
            logging.info(f"图像对比度过低: {contrast}")
            return False

        # 检查模糊度
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if laplacian_var < 100:
            logging.info(f"图像可能模糊: {laplacian_var}")
            return False

        return True
    except Exception as e:
        logging.error(f"检查图像质量时出错: {str(e)}")
        return False


def detect_qr_code(image_content):
    """
    检测图片中是否包含二维码
    :param image_content: 图片内容
    :return: 布尔值（是否包含二维码）
    """
    try:
        # 直接从内容读取图片
        image_data = np.frombuffer(image_content, np.uint8)
        image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)

        if image is None:
            logging.error("无法解码图片内容")
            return False

        # 检查图像尺寸
        if image.shape[0] * image.shape[1] > 4000 * 4000:  # 限制最大分辨率
            logging.info("图像分辨率过大，进行压缩")
            scale = min(4000 / image.shape[0], 4000 / image.shape[1])
            new_size = (int(image.shape[1] * scale), int(image.shape[0] * scale))
            image = cv2.resize(image, new_size)

        # 检查图像质量
        if not check_image_quality(image):
            logging.info("图像质量不足，尝试进行图像增强")
            # 进行基础图像增强
            image = cv2.convertScaleAbs(image, alpha=1.2, beta=10)

        try:
            # 创建微信二维码检测器
            model_base_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "models"
            )  # 确保这个目录存在并包含所需模型文件
            detector = cv2.wechat_qrcode.WeChatQRCode(
                os.path.join(model_base_path, "detect.prototxt"),
                os.path.join(model_base_path, "detect.caffemodel"),
                os.path.join(model_base_path, "sr.prototxt"),
                os.path.join(model_base_path, "sr.caffemodel"),
            )

            # 图像预处理
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

            # 创建多个图像处理版本
            processed_images = []

            # 1. 原图（保持原图优先级最高）
            processed_images.append(image)

            # 2. 基础预处理（轻微增强）
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            denoised = cv2.GaussianBlur(gray, (3, 3), 0)
            _, binary = cv2.threshold(
                denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            binary_rgb = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
            processed_images.append(binary_rgb)

            # 3. 自适应二值化（对复杂背景更好）
            adaptive_binary = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
            adaptive_rgb = cv2.cvtColor(adaptive_binary, cv2.COLOR_GRAY2RGB)
            processed_images.append(adaptive_rgb)

            # 4. 轻微对比度增强
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))  # 降低clipLimit
            l = clahe.apply(l)
            enhanced_lab = cv2.merge([l, a, b])
            processed_images.append(cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB))

            # 5. 轻微锐化
            kernel = np.array(
                [[-0.5, -0.5, -0.5], [-0.5, 5, -0.5], [-0.5, -0.5, -0.5]]
            )  # 降低锐化强度
            sharpened = cv2.filter2D(image, -1, kernel)
            processed_images.append(sharpened)

            # 使用threading.Timer替代signal实现超时
            def timeout_handler():
                _thread.interrupt_main()

            timer = threading.Timer(30.0, timeout_handler)  # 30秒超时
            timer.start()

            try:
                # 在所有处理后的图像上尝试检测
                decoded_text = []
                for test_image in processed_images:
                    current_decoded_text, _ = detector.detectAndDecode(test_image)
                    if len(current_decoded_text) > 0:
                        decoded_text.extend(current_decoded_text)

                # 去重结果
                decoded_text = list(set(decoded_text))

                # 记录解码内容
                if len(decoded_text) > 0:
                    logging.info(f"解码内容: {decoded_text}")
                    return True

            except KeyboardInterrupt:
                logging.error("图像处理超时")
                return False
            except Exception as e:
                logging.error(f"处理图像时出错: {str(e)}")
                return False
            finally:
                timer.cancel()

        except Exception as e:
            logging.error(f"二维码检测过程中出错: {str(e)}")
            return False

        return False
    except Exception as e:
        logging.error(f"检测二维码时出错: {str(e)}")
        return False


async def process_image_async(websocket, group_id, user_id, message_id, image_url):
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(image_url) as response:
                image_content = await response.read()

        # 将同步的 detect_qr_code 函数放在线程池中执行
        loop = asyncio.get_event_loop()
        has_qr = await loop.run_in_executor(None, detect_qr_code, image_content)

        if has_qr:
            await send_group_msg(
                websocket,
                group_id,
                "[CQ:at,qq=" + user_id + "]本群禁止发送二维码，请遵守群规。",
            )
            await delete_msg(websocket, message_id)
    except Exception as e:
        logging.error(f"处理图片时发生错误: {e}")


# 群消息处理函数
async def handle_QRCodeInspector_group_message(websocket, msg):
    # 确保数据目录存在
    os.makedirs(DATA_DIR, exist_ok=True)
    try:
        user_id = str(msg.get("user_id"))
        group_id = str(msg.get("group_id"))
        raw_message = str(msg.get("raw_message"))
        role = str(msg.get("sender", {}).get("role"))
        message_id = str(msg.get("message_id"))
        authorized = user_id in owner_id

        # 开关
        if raw_message == "qri":
            await toggle_function_status(websocket, group_id, message_id, authorized)
            return
        # 检查是否开启
        if load_function_status(group_id):
            # 检测二维码
            if "CQ:image" in raw_message:
                # 获取图片url
                image_url = re.search(r"url=([^,]+)", raw_message)
                if image_url:
                    image_url = image_url.group(1)
                    # 把HTML的转义字符转换为标准的URL
                    image_url = image_url.replace("&amp;", "&").replace("https", "http")
                    # 异步处理图片
                    asyncio.create_task(
                        process_image_async(
                            websocket, group_id, user_id, message_id, image_url
                        )
                    )

    except Exception as e:
        logging.error(f"处理QRCodeInspector群消息失败: {e}")
        await send_group_msg(
            websocket,
            group_id,
            "处理QRCodeInspector群消息失败，错误信息：" + str(e),
        )
        return


# 回应事件处理函数
async def handle_QRCodeInspector_response_message(websocket, msg):
    try:

        if msg.get("status") == "ok":
            echo = msg.get("echo")

            if echo and echo.startswith("xxx"):
                pass
    except Exception as e:
        logging.error(f"处理QRCodeInspector回应事件时发生错误: {e}")


# 统一事件处理入口
async def handle_events(websocket, msg):
    """统一事件处理入口"""
    post_type = msg.get("post_type", "response")  # 添加默认值
    try:
        # 处理回调事件
        if msg.get("status") == "ok":
            await handle_QRCodeInspector_response_message(websocket, msg)
            return

        post_type = msg.get("post_type")

        # 处理元事件
        if post_type == "meta_event":
            await handle_QRCodeInspector_meta_event(websocket)

        # 处理消息事件
        elif post_type == "message":
            message_type = msg.get("message_type")
            if message_type == "group":
                await handle_QRCodeInspector_group_message(websocket, msg)
            elif message_type == "private":
                return

        # 处理通知事件
        elif post_type == "notice":
            if msg.get("notice_type") == "group":
                return

    except Exception as e:
        error_type = {
            "message": "消息",
            "notice": "通知",
            "request": "请求",
            "meta_event": "元事件",
        }.get(post_type, "未知")

        logging.error(f"处理QRCodeInspector{error_type}事件失败: {e}")

        # 发送错误提示
        if post_type == "message":
            message_type = msg.get("message_type")
            if message_type == "group":
                await send_group_msg(
                    websocket,
                    msg.get("group_id"),
                    f"处理QRCodeInspector{error_type}事件失败，错误信息：{str(e)}",
                )
            elif message_type == "private":
                await send_private_msg(
                    websocket,
                    msg.get("user_id"),
                    f"处理QRCodeInspector{error_type}事件失败，错误信息：{str(e)}",
                )
