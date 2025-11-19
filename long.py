import cv2
import mediapipe as mp
import time
import numpy as np
from collections import Counter


class PinchGestureDetector:
    def __init__(self, hands_model):
        self.hands_model = hands_model

        # 状态跟踪
        self.is_pinching = False
        self.last_process_time = 0
        self.results_cache = None
        self.processing_interval = 0.01
        self.confidence_threshold = 0.7  # 手部检测置信度阈值

        # 手势历史记录 - 使用较短的历史以获得更快的响应
        self.pinch_gesture_history = []  # 记录捏合手势历史
        self.open_hand_history = []  # 记录松开手势历史
        self.history_length = 3  # 每种手势只记录3帧历史

        # MediaPipe 绘图工具
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_hands = mp.solutions.hands

        # 捏合距离阈值 (可根据实际情况调整)
        self.pinch_distance_threshold = 0.05

    def process_hand(self, img):
        """处理图像并获取手部关键点数据"""
        current_time = time.time()

        # 使用缓存的结果，如果时间间隔很短
        if self.results_cache is not None and current_time - self.last_process_time < self.processing_interval:
            return self.results_cache

        # 处理新帧
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results_cache = self.hands_model.process(rgb_img)
        self.last_process_time = current_time

        return self.results_cache

    def calculate_distance(self, landmark1, landmark2):
        """计算两个关键点之间的3D距离"""
        return np.sqrt((landmark1.x - landmark2.x) ** 2 +
                       (landmark1.y - landmark2.y) ** 2 +
                       (landmark1.z - landmark2.z) ** 2)

    def detect_pinch_gesture(self, hand_landmarks):
        """检测拇指、食指和中指是否捏合"""
        # 检查是否为右手
        is_right_hand = True
        if hasattr(self.results_cache, 'multi_handedness') and self.results_cache.multi_handedness:
            # MediaPipe的handedness检测 (score > 0.5为右手，否则为左手)
            handedness = self.results_cache.multi_handedness[0].classification[0]
            is_right_hand = handedness.label == "Right" or handedness.score > 0.5

            # 如果置信度太低则跳过
            if handedness.score < self.confidence_threshold:
                return False

        if not is_right_hand:
            return False

        # 获取手指尖的关键点
        thumb_tip = hand_landmarks.landmark[4]  # 拇指尖
        index_tip = hand_landmarks.landmark[8]  # 食指尖
        middle_tip = hand_landmarks.landmark[12]  # 中指尖

        # 计算三指之间的距离
        thumb_to_index_distance = self.calculate_distance(thumb_tip, index_tip)
        thumb_to_middle_distance = self.calculate_distance(thumb_tip, middle_tip)
        index_to_middle_distance = self.calculate_distance(index_tip, middle_tip)

        # 判断三指是否都很近
        all_fingers_close = (thumb_to_index_distance < self.pinch_distance_threshold and
                             thumb_to_middle_distance < self.pinch_distance_threshold and
                             index_to_middle_distance < self.pinch_distance_threshold)

        return all_fingers_close

    def detect_open_hand(self, hand_landmarks):
        """检测手是否打开（三指分开）"""
        # 检查是否为右手
        is_right_hand = True
        if hasattr(self.results_cache, 'multi_handedness') and self.results_cache.multi_handedness:
            # MediaPipe的handedness检测 (score > 0.5为右手，否则为左手)
            handedness = self.results_cache.multi_handedness[0].classification[0]
            is_right_hand = handedness.label == "Right" or handedness.score > 0.5

            # 如果置信度太低则跳过
            if handedness.score < self.confidence_threshold:
                return False

        if not is_right_hand:
            return False

        # 获取手指尖的关键点
        thumb_tip = hand_landmarks.landmark[4]  # 拇指尖
        index_tip = hand_landmarks.landmark[8]  # 食指尖
        middle_tip = hand_landmarks.landmark[12]  # 中指尖

        # 计算三指之间的距离
        thumb_to_index_distance = self.calculate_distance(thumb_tip, index_tip)
        thumb_to_middle_distance = self.calculate_distance(thumb_tip, middle_tip)
        index_to_middle_distance = self.calculate_distance(index_tip, middle_tip)

        # 判断三指是否都分开
        all_fingers_apart = (thumb_to_index_distance > self.pinch_distance_threshold * 2 and
                             thumb_to_middle_distance > self.pinch_distance_threshold * 2 and
                             index_to_middle_distance > self.pinch_distance_threshold * 2)

        return all_fingers_apart

    def update_gesture_history(self, is_pinching, is_open_hand):
        """更新手势历史记录"""
        # 更新捏合手势历史
        self.pinch_gesture_history.append(1 if is_pinching else 0)
        if len(self.pinch_gesture_history) > self.history_length:
            self.pinch_gesture_history.pop(0)

        # 更新松开手势历史
        self.open_hand_history.append(1 if is_open_hand else 0)
        if len(self.open_hand_history) > self.history_length:
            self.open_hand_history.pop(0)

    def get_pinch_action(self, img):
        """
        检测捏合相关动作
        返回:
        - "pinch_start": 开始捏合
        - "pinch_end": 结束捏合
        - None: 无变化
        """
        results = self.process_hand(img)

        # 如果没有检测到手，且当前正在捏合，则结束捏合
        if not results.multi_hand_landmarks:
            if self.is_pinching:
                self.is_pinching = False
                # 清空手势历史
                self.pinch_gesture_history = []
                self.open_hand_history = []
                return "pinch_end"
            else:
                # 清空手势历史
                self.pinch_gesture_history = []
                self.open_hand_history = []
                return None

        # 获取当前手势
        hand_landmarks = results.multi_hand_landmarks[0]
        is_pinching = self.detect_pinch_gesture(hand_landmarks)
        is_open_hand = self.detect_open_hand(hand_landmarks)

        # 更新手势历史
        self.update_gesture_history(is_pinching, is_open_hand)

        # 检测捏合开始 - 需要在最近3帧中有至少2帧检测到捏合手势
        if not self.is_pinching and sum(self.pinch_gesture_history) >= 2:
            self.is_pinching = True
            return "pinch_start"

        # 检测捏合结束 - 如果在捏合状态下，检测到松开手势
        if self.is_pinching and sum(self.open_hand_history) >= 2:
            self.is_pinching = False
            return "pinch_end"

        # 无变化
        return None

    def draw_status(self, img):
        """在图像上显示捏合状态"""
        result_img = img.copy()

        if self.is_pinching:
            cv2.putText(result_img, "长按状态: 开启", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0), 2)

        return result_img


# 如果直接运行此文件，则执行以下测试代码
if __name__ == "__main__":
    # 初始化摄像头
    cap = cv2.VideoCapture(0)

    # 设置摄像头分辨率
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # 初始化MediaPipe手势模型
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # 初始化捏合手势检测器
    pinch_detector = PinchGestureDetector(hands)

    # 创建窗口
    window_name = "long_test"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 640, 480)

    # 绘画状态
    is_pinching = False
    drawing_points = []  # 存储绘画点

    try:
        while True:
            # 读取摄像头帧
            success, frame = cap.read()
            if not success:
                print("无法读取摄像头")
                break

            # 水平翻转图像，使其更直观
            frame = cv2.flip(frame, 1)

            # 检测捏合相关动作
            action = pinch_detector.get_pinch_action(frame)

            # 处理动作
            if action == "pinch_start":
                is_pinching = True
                print("开始长按")  # 只在控制台打印，不在屏幕显示
            elif action == "pinch_end":
                is_pinching = False
                print("结束长按")  # 只在控制台打印，不在屏幕显示

            # 获取当前鼠标位置 (使用食指尖作为鼠标位置)
            if pinch_detector.results_cache and pinch_detector.results_cache.multi_hand_landmarks:
                hand = pinch_detector.results_cache.multi_hand_landmarks[0]
                # 获取食指尖的位置 (8号关键点)
                h, w, c = frame.shape
                cx = int(hand.landmark[8].x * w)
                cy = int(hand.landmark[8].y * h)

                # 在帧上显示鼠标光标
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

                # 如果在捏合模式，添加当前点到绘画点列表
                if is_pinching:
                    drawing_points.append((cx, cy))

            # 绘制所有点
            if len(drawing_points) > 1:
                for i in range(1, len(drawing_points)):
                    if i == 0:
                        continue
                    # 连接点以形成线条
                    cv2.line(frame, drawing_points[i - 1], drawing_points[i], (0, 255, 0), 2)

            # 绘制手部关键点
            if pinch_detector.results_cache and pinch_detector.results_cache.multi_hand_landmarks:
                pinch_detector.mp_drawing.draw_landmarks(
                    frame,
                    pinch_detector.results_cache.multi_hand_landmarks[0],
                    pinch_detector.mp_hands.HAND_CONNECTIONS,
                    pinch_detector.mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2),
                    pinch_detector.mp_drawing.DrawingSpec(color=(0, 125, 255), thickness=2)
                )

            # 显示帧
            cv2.imshow(window_name, frame)

            # 键盘输入检测
            key = cv2.waitKey(1)
            if key == 27:  # ESC键退出
                break
            elif key == ord('c'):  # 'c'键清除绘画
                drawing_points = []
                print("清除绘画")  # 只在控制台打印，不在屏幕显示

    except Exception as e:
        print(f"发生错误: {e}")
    finally:
        # 释放资源
        cap.release()
        cv2.destroyAllWindows()
