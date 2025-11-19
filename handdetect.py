import cv2
import mediapipe as mp
import numpy as np


class HandDetector:
    """
    手部检测器，基于 MediaPipe Hands 实现手部关键点的检测与绘制
    """

    def __init__(self, min_detection_confidence=0.7, min_tracking_confidence=0.6, flip=True):
        # 初始化 MediaPipe Hands 解决方案
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        # 使用更稳定的设置
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,  # 视频流模式
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            max_num_hands=2,  # 最大检测手数
            model_complexity=1  # 平衡准确性和性能的复杂度
        )

        # 打开摄像头
        self.cap = cv2.VideoCapture(0)
        # 设置摄像头参数以获得更好的性能
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        # 是否水平翻转图像（常用于镜像效果）
        self.flip = flip

        # 手部跟踪稳定性
        self.previous_hands = []
        self.smoothing_factor = 0.7  # 平滑因子

    def get_hand_data(self):
        """
        获取摄像头图像，并检测手部关键点
        Returns:
            frame: 当前处理后的图像帧
            hand_info_list: 检测到的手部信息列表，每个元素包含手部边界框、中心点、关键点列表、左右手标签等
        """
        hand_info_list = []
        ret, frame = self.cap.read()
        if not ret:
            print("无法读取摄像头数据")
            return frame, hand_info_list

        # 如果需要翻转，则进行水平翻转
        if self.flip:
            frame = cv2.flip(frame, 1)
        height, width, _ = frame.shape

        # 在左上角绘制提示文字
        cv2.putText(frame, "virtual mouse", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 255), 2, cv2.LINE_AA)

        # 对图像进行优化处理
        frame_processed = self.preprocess_image(frame)

        # 处理图像，检测手部关键点
        results = self.hands.process(frame_processed)

        # 如果检测到手部
        new_hand_info_list = []
        if results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # 将关键点坐标从归一化坐标转换为像素坐标
                x_coords = [int(lm.x * width) for lm in hand_landmarks.landmark]
                y_coords = [int(lm.y * height) for lm in hand_landmarks.landmark]
                min_x, max_x = min(x_coords), max(x_coords)
                min_y, max_y = min(y_coords), max(y_coords)
                center_x = (min_x + max_x) // 2
                center_y = (min_y + max_y) // 2

                # 计算手掌面积，排除面积过小或过大的检测结果
                palm_area = (max_x - min_x) * (max_y - min_y)
                if palm_area < (width * height * 0.005) or palm_area > (width * height * 0.3):
                    continue

                # 获取左右手信息及检测置信度
                detected_type = results.multi_handedness[idx].classification[0].label
                confidence = results.multi_handedness[idx].classification[0].score

                # 如果置信度过低，跳过
                if confidence < 0.8:
                    continue

                # 绘制手部骨架及关键点
                self.mp_drawing.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2),
                    self.mp_drawing.DrawingSpec(color=(0, 125, 255), thickness=2)
                )

                # 显示左/右手和置信度
                hand_text = f"{detected_type} ({confidence:.2f})"
                cv2.putText(frame, hand_text, (min_x, min_y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                # 将检测到的手部信息加入列表
                new_hand_info_list.append({
                    'bbox': (min_x, min_y, max_x, max_y),
                    'center': (center_x, center_y),
                    'landmarks': hand_landmarks.landmark,
                    'label': detected_type,
                    'confidence': confidence
                })

        # 应用平滑处理以提高稳定性
        hand_info_list = self.smooth_hand_tracking(new_hand_info_list)

        return frame, hand_info_list

    def preprocess_image(self, frame):
        """
        预处理图像以提高手部检测的稳定性
        """
        # 对图像进行高斯模糊，降低噪声
        frame_blur = cv2.GaussianBlur(frame, (5, 5), 0)

        # 应用颜色校正和增强对比度
        frame_lab = cv2.cvtColor(frame_blur, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(frame_lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        frame_lab = cv2.merge((l, a, b))
        frame_enhanced = cv2.cvtColor(frame_lab, cv2.COLOR_LAB2BGR)

        # 将 BGR 图像转换为 RGB，因为 MediaPipe 需要 RGB 格式
        frame_rgb = cv2.cvtColor(frame_enhanced, cv2.COLOR_BGR2RGB)

        return frame_rgb

    def smooth_hand_tracking(self, new_hand_info_list):
        """
        平滑手部跟踪，减少抖动
        """
        if not self.previous_hands:
            self.previous_hands = new_hand_info_list
            return new_hand_info_list

        smoothed_hands = []

        # 匹配新旧手部，进行平滑处理
        for new_hand in new_hand_info_list:
            matched = False
            for i, prev_hand in enumerate(self.previous_hands):
                # 检查是否是同一只手（通过标签和位置匹配）
                if prev_hand['label'] == new_hand['label']:
                    prev_center = prev_hand['center']
                    new_center = new_hand['center']

                    # 计算欧几里得距离
                    distance = ((prev_center[0] - new_center[0]) ** 2 +
                                (prev_center[1] - new_center[1]) ** 2) ** 0.5

                    # 如果距离足够近，认为是同一只手
                    if distance < 100:  # 可调整的阈值
                        # 对关键点位置应用平滑滤波
                        smoothed_landmarks = []
                        for j, landmark in enumerate(new_hand['landmarks']):
                            prev_lm = prev_hand['landmarks'][j]
                            smoothed_x = self.smoothing_factor * prev_lm.x + (1 - self.smoothing_factor) * landmark.x
                            smoothed_y = self.smoothing_factor * prev_lm.y + (1 - self.smoothing_factor) * landmark.y
                            smoothed_z = self.smoothing_factor * prev_lm.z + (1 - self.smoothing_factor) * landmark.z

                            # 创建新的landmark对象
                            smoothed_lm = type('obj', (object,), {
                                'x': smoothed_x,
                                'y': smoothed_y,
                                'z': smoothed_z
                            })
                            smoothed_landmarks.append(smoothed_lm)

                        # 更新平滑后的手信息
                        smoothed_hand = new_hand.copy()
                        smoothed_hand['landmarks'] = smoothed_landmarks

                        # 重新计算边界框和中心点
                        x_coords = [int(lm.x * self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) for lm in smoothed_landmarks]
                        y_coords = [int(lm.y * self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) for lm in smoothed_landmarks]
                        min_x, max_x = min(x_coords), max(x_coords)
                        min_y, max_y = min(y_coords), max(y_coords)
                        center_x = (min_x + max_x) // 2
                        center_y = (min_y + max_y) // 2

                        smoothed_hand['bbox'] = (min_x, min_y, max_x, max_y)
                        smoothed_hand['center'] = (center_x, center_y)

                        smoothed_hands.append(smoothed_hand)
                        matched = True
                        break

            # 如果没有匹配到之前的手，直接添加
            if not matched:
                smoothed_hands.append(new_hand)

        # 更新之前的手信息
        self.previous_hands = smoothed_hands

        return smoothed_hands

    def release(self):
        """
        释放摄像头及窗口资源
        """
        self.cap.release()
        cv2.destroyAllWindows()
