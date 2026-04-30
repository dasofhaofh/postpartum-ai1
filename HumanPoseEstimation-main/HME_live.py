import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
import os

# ---------- 设置页面标题 ----------
st.set_page_config(page_title="产后恢复 AI 姿态评估", layout="wide")
st.title("产后恢复 AI 姿态评估助手")
st.markdown("请面对摄像头，保持全身在画面内，系统会自动检测你的身体关键点。")

# ---------- 获取当前脚本所在目录（用于定位模型文件）----------
base_dir = os.path.dirname(os.path.abspath(__file__))

# 方式一：使用本地模型文件（如果你已经上传了）
model_filename = "pose_landmark_lite.tflite"
model_path = os.path.join(base_dir, model_filename)

# 如果模型文件不存在，则让 MediaPipe 使用默认下载（但如果云端无写权限仍会失败）
# 我们优先尝试使用本地模型
use_local_model = os.path.exists(model_path)

# ---------- 初始化 MediaPipe Pose ----------
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# 根据是否有本地模型文件选择参数
if use_local_model:
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=0,               # 0 = 轻量级模型（对应 lite）
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_asset_path=model_path      # 直接使用本地文件
    )
    st.success(f"使用本地模型文件: {model_filename}")
else:
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=0,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    st.warning("未找到本地模型文件，尝试使用默认下载（云端可能失败）。建议上传姿态模型文件。")

# ---------- 侧边栏：使用说明 ----------
with st.sidebar:
    st.header("使用说明")
    st.markdown("""
    1. 点击「开始摄像头」按钮。
    2. 允许浏览器访问你的摄像头。
    3. 保持距离，使全身出现在画面中。
    4. 点击「捕捉并分析」按钮，系统将检测你的身体姿态。
    5. 检测结果会在右侧显示。
    """)
    st.info("提示：建议穿紧身或浅色衣物，背景简洁，光线充足。")

# ---------- 使用 st.camera_input 获取图像 ----------
camera_image = st.camera_input("拍照分析", key="pose_camera")

if camera_image is not None:
    # 将 PIL Image 转换为 numpy array (RGB)
    image = Image.open(camera_image)
    image_np = np.array(image)

    # 转换为 BGR 格式（MediaPipe 处理 RGB，但 OpenCV 画图习惯用 BGR，这里直接使用 RGB）
    frame_rgb = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)   # 兼容绘图习惯
    results = pose.process(cv2.cvtColor(image_np, cv2.COLOR_RGB2RGB))  # MediaPipe 需要 RGB

    # 绘制关键点
    annotated_image = image_np.copy()
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            annotated_image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
        )
        st.success("检测到人体姿态！")
    else:
        st.warning("未检测到人体姿态，请调整位置或光线。")

    # 显示结果
    st.image(annotated_image, caption="姿态分析结果", use_container_width=True)

    # 附加说明：可以在此添加针对产后恢复的动作评估逻辑
    if results.pose_landmarks:
        # 示例：获取关键点坐标 (髋、膝、肩等)
        landmarks = results.pose_landmarks.landmark
        # 你可以在这里加入角度计算等逻辑
        st.info("姿态数据已获取，后续可扩展动作标准度评分。")

else:
    st.info("请点击左侧「开始摄像头」按钮并拍照分析。")

# 释放资源（Streamlit 不需要手动释放）
