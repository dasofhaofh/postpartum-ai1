import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
import os

# ---------- 关键：设置模型文件目录，避免云端下载 ----------
base_dir = os.path.dirname(os.path.abspath(__file__))
os.environ["MEDIAPIPE_DOWNLOAD_OSS_MODEL_DIR"] = base_dir

st.set_page_config(page_title="产后恢复 AI 姿态评估", layout="wide")
st.title("产后恢复 AI 姿态评估助手")
st.markdown("请面对摄像头，保持全身在画面内，系统会自动检测你的身体关键点。")

# ---------- 初始化 MediaPipe Pose ----------
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,      # 处理图片流时设为False
    model_complexity=0,           # 0 = 轻量级模型（对应 pose_landmark_lite.tflite）
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# 侧边栏使用说明
with st.sidebar:
    st.header("使用说明")
    st.markdown("""
    1. 点击「拍照分析」按钮。
    2. 允许浏览器访问你的摄像头。
    3. 保持距离，使全身出现在画面中。
    4. 点击「捕捉并分析」按钮，系统将检测你的身体姿态。
    5. 检测结果会在右侧显示。
    """)
    st.info("提示：建议穿紧身或浅色衣物，背景简洁，光线充足。")

# ---------- 摄像头输入 ----------
camera_image = st.camera_input("拍照分析", key="pose_camera")

if camera_image is not None:
    # 读取图片并转为 OpenCV 格式
    image = Image.open(camera_image)
    image_np = np.array(image)
    
    # MediaPipe 需要 RGB 图像
    results = pose.process(cv2.cvtColor(image_np, cv2.COLOR_RGB2RGB))

    # 绘制骨骼点
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

        # 可选：提取关键点坐标（示例）
        landmarks = results.pose_landmarks.landmark
        st.info(f"共检测到 {len(landmarks)} 个关键点，可用于后续动作标准度分析。")
    else:
        st.warning("未检测到人体姿态，请调整位置或光线。")

    # 显示结果
    st.image(annotated_image, caption="姿态分析结果", use_container_width=True)
else:
    st.info("请点击「拍照分析」按钮，允许摄像头后拍照。")

# 尾部说明
st.markdown("---")
st.caption("本应用使用 MediaPipe Pose 进行实时人体姿态估计，适用于产后恢复动作辅助评估。")
