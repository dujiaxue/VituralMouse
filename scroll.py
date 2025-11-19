import cv2
import pyautogui
import time
import numpy as np


class ScrollHandler:
    def __init__(self):
        self.last_scroll_time = 0
        self.last_speed = 0
        self.speed_history = []
        self.history_size = 5
        self.min_scroll_interval = 0.04  # 缩短间隔提高响应速度
        self.smoothing_factor = 0.3  # 平滑因子

    def smooth_speed(self, new_speed):
        """使用滑动平均平滑滚动速度"""
        self.speed_history.append(new_speed)
        if len(self.speed_history) > self.history_size:
            self.speed_history.pop(0)

        # 应用平滑因子
        smoothed_speed = self.last_speed * (1 - self.smoothing_factor) + new_speed * self.smoothing_factor
        self.last_speed = smoothed_speed
        return int(smoothed_speed)


def is_finger_extended(landmarks, tip_idx, pip_idx):
    """
    判断手指是否伸出：
    如果指尖 (tip) 的 y 坐标小于 PIP 节点的 y 坐标，则认为该手指伸出

    Args:
        landmarks: 手部关键点列表
        tip_idx: 指尖关键点索引
        pip_idx: PIP 节点关键点索引

    Returns:
        bool: True 表示手指伸出，否则 False
    """
    tip = landmarks[tip_idx]
    pip = landmarks[pip_idx]
    return tip.y < pip.y


def is_thumb_extended(landmarks):
    """
    判断拇指是否伸出：
    对于拇指，由于其运动方向主要是水平的，因此我们比较拇指尖和拇指掌指关节的x坐标

    Args:
        landmarks: 手部关键点列表

    Returns:
        bool: True 表示拇指伸出，否则 False
    """
    thumb_tip = landmarks[4]  # 拇指尖
    thumb_mcp = landmarks[2]  # 拇指掌指关节
    # 对于右手，如果拇指尖的x坐标小于拇指掌指关节的x坐标，表示拇指伸出
    # 注：此判断方法的前提是按习惯来讲人手心面对摄像头
    return thumb_tip.x < thumb_mcp.x


def compute_scroll_speed(landmarks, indices, min_speed=5, max_speed=60):
    """
    根据传入的手指索引对（tip_idx, pip_idx）列表计算滚动速度，
    增加速度范围和非线性映射提高精细控制

    Args:
        landmarks: 手部关键点列表
        indices: 每个元素为 (tip_idx, pip_idx) 的元组列表
        min_speed: 最小滚动速度
        max_speed: 最大滚动速度 (已增加到60)

    Returns:
        int: 映射到[min_speed, max_speed]区间的滚动速度
    """
    # 计算各手指 tip 与 pip 之间的距离（使用 y 坐标差）
    diffs = [abs(landmarks[tip].y - landmarks[pip].y) for tip, pip in indices]
    avg_diff = sum(diffs) / len(diffs)

    # 假设手指伸展时的距离范围在 0.02 到 0.15 之间
    normalized = (avg_diff - 0.02) / (0.15 - 0.02)
    normalized = max(0, min(normalized, 1))

    # 使用平方函数进行非线性映射，提高低速区间的精细控制
    normalized = normalized ** 1.5

    # 线性映射到指定的滚动速度区间
    speed = int(min_speed + normalized * (max_speed - min_speed))
    return speed


# 创建全局滚动处理器
scroll_handler = ScrollHandler()


def process_scroll_control(frame, hand_info_list):
    """
    处理滚动控制逻辑，根据右手手势实现向上或向下滚动
    新增要求：，两种手势都要求拇指不伸出

    Args:
        frame: 当前视频帧
        hand_info_list: 检测到的手部信息列表

    Returns:
        frame: 处理后的图像帧
    """
    global scroll_handler
    current_time = time.time()

    # 查找标记为 "Right Hand" 的手，并获取其 landmarks
    right_hand_landmarks = None
    for hand_info in hand_info_list:
        if hand_info.get('label', '') == "Right Hand" and 'landmarks' in hand_info:
            right_hand_landmarks = hand_info['landmarks']
            break

    # 如果没有找到"Right Hand"，尝试找到任何一只手
    if right_hand_landmarks is None and hand_info_list:
        for hand_info in hand_info_list:
            if 'landmarks' in hand_info:
                right_hand_landmarks = hand_info['landmarks']
                break

    if right_hand_landmarks is not None:
        # 判断右手各个手指是否伸出
        thumb_extended = is_thumb_extended(right_hand_landmarks)  # 检查拇指是否伸出
        index_extended = is_finger_extended(right_hand_landmarks, 8, 6)  # 检查食指
        middle_extended = is_finger_extended(right_hand_landmarks, 12, 10)  # 检查中指
        ring_extended = is_finger_extended(right_hand_landmarks, 16, 14)  # 检查无名指
        pinky_extended = is_finger_extended(right_hand_landmarks, 20, 18)  # 检查小指

        # 定义用于计算滚动速度的手指索引对
        upward_indices = [(8, 6), (12, 10), (16, 14)]  # 用于向上滚动的手指索引对
        downward_indices = [(8, 6), (12, 10), (16, 14), (20, 18)]  # 用于向下滚动的手指索引对

        # 只有当时间间隔大于最小滚动间隔时才执行滚动
        if current_time - scroll_handler.last_scroll_time >= scroll_handler.min_scroll_interval:
            scroll_handler.last_scroll_time = current_time

            # 条件1：拇指不伸出 + 右手中间三根手指伸出 + 小指未伸出，向上滚动
            if not thumb_extended and index_extended and middle_extended and ring_extended and not pinky_extended:
                raw_speed = compute_scroll_speed(right_hand_landmarks, upward_indices, 5, 60)
                smooth_speed = scroll_handler.smooth_speed(raw_speed)

                # 使用非线性映射提供更精细的控制
                if smooth_speed < 10:
                    actual_speed = smooth_speed // 2  # 低速区更精细
                else:
                    actual_speed = smooth_speed

                pyautogui.scroll(actual_speed)  # 向上滚动

            # 条件2：拇指不伸出 + 右手除拇指之外所有手指伸出，向下滚动
            elif not thumb_extended and index_extended and middle_extended and ring_extended and pinky_extended:
                raw_speed = compute_scroll_speed(right_hand_landmarks, downward_indices, 5, 60)
                smooth_speed = scroll_handler.smooth_speed(raw_speed)

                # 使用非线性映射提供更精细的控制
                if smooth_speed < 10:
                    actual_speed = smooth_speed // 2  # 低速区更精细
                else:
                    actual_speed = smooth_speed

                pyautogui.scroll(-actual_speed)  # 向下滚动

    return frame


# 添加本地测试代码
if __name__ == "__main__":
    import mediapipe as mp
    from handdetect import HandDetector

    # 初始化手部检测器
    detector = HandDetector()

    while True:
        # 获取手部数据
        frame, hand_info_list = detector.get_hand_data()
        if frame is None:
            continue

        # 处理滚动控制
        frame = process_scroll_control(frame, hand_info_list)

        # 显示测试窗口
        cv2.imshow("Scroll Control Test", frame)

        # 按 Esc 键退出
        if cv2.waitKey(1) & 0xFF == 27:
            break

    # 释放资源
    detector.release()
