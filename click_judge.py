import cv2
import time
import pyautogui
import numpy as np
from handdetect import HandDetector


class ClickController:
    def __init__(self):
        self.last_click_time = 0
        self.click_cooldown = 0.5  # 点击冷却时间（秒）

    def process_clicks(self, frame, hand_info_list):
        """
        处理点击控制逻辑，根据手势实现左右键点击

        Args:
            frame: 当前视频帧
            hand_info_list: 检测到的手部信息列表

        Returns:
            frame: 处理后的图像帧
        """
        # 查找右手
        right_hand = None
        for hand in hand_info_list:
            if hand.get('label', '') == "Right Hand" and 'landmarks' in hand:
                right_hand = hand
                break

        # 如果没找到右手，尝试找到任何一只手
        if right_hand is None and hand_info_list:
            for hand in hand_info_list:
                if 'landmarks' in hand:
                    right_hand = hand
                    break

        if right_hand is None:
            return frame

        # 获取手部关键点
        landmarks = right_hand['landmarks']

        # 判断各手指是否伸出
        thumb_extended = self.is_thumb_extended(landmarks)
        index_extended = self.is_finger_extended(landmarks, 8, 6)  # 食指
        middle_extended = self.is_finger_extended(landmarks, 12, 10)  # 中指
        ring_extended = self.is_finger_extended(landmarks, 16, 14)  # 无名指
        pinky_extended = self.is_finger_extended(landmarks, 20, 18)  # 小指

        # 获取图像尺寸
        h, w, _ = frame.shape

        # 当前时间
        current_time = time.time()

        # 获取关键点坐标
        thumb_tip_x = int(landmarks[4].x * w)
        thumb_tip_y = int(landmarks[4].y * h)
        index_tip_x = int(landmarks[8].x * w)
        index_tip_y = int(landmarks[8].y * h)
        middle_tip_x = int(landmarks[12].x * w)
        middle_tip_y = int(landmarks[12].y * h)

        # 计算距离
        thumb_index_distance = ((index_tip_x - thumb_tip_x) ** 2 + (index_tip_y - thumb_tip_y) ** 2) ** 0.5
        index_middle_distance = ((index_tip_x - middle_tip_x) ** 2 + (index_tip_y - middle_tip_y) ** 2) ** 0.5

        # 统计伸出的手指数量（用于避免滚动冲突）
        extended_fingers_count = sum([thumb_extended, index_extended, middle_extended, ring_extended, pinky_extended])

        # 左键点击：拇指食指捏合，且不是在滚动状态（伸出手指数≤2）
        if (thumb_index_distance < 40 and extended_fingers_count <= 2 and 
            current_time - self.last_click_time > self.click_cooldown):
            
            # 显示点击位置
            center_x = (index_tip_x + thumb_tip_x) // 2
            center_y = (index_tip_y + thumb_tip_y) // 2
            cv2.circle(frame, (center_x, center_y), 10, (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, "LEFT", (center_x - 25, center_y - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # 执行左键点击
            pyautogui.click()
            self.last_click_time = current_time

        # 右键点击：食指中指捏合，但要求拇指收起且总手指数不超过2个
        elif (index_middle_distance < 30 and not thumb_extended and extended_fingers_count <= 2 and
              current_time - self.last_click_time > self.click_cooldown):
            
            # 显示点击位置
            center_x = (index_tip_x + middle_tip_x) // 2
            center_y = (index_tip_y + middle_tip_y) // 2
            cv2.circle(frame, (center_x, center_y), 10, (0, 0, 255), cv2.FILLED)
            cv2.putText(frame, "RIGHT", (center_x - 30, center_y - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # 执行右键点击
            pyautogui.rightClick()
            self.last_click_time = current_time

        return frame

    def is_finger_extended(self, landmarks, tip_idx, pip_idx):
        """
        判断手指是否伸出：
        如果指尖 (tip) 的 y 坐标小于 PIP 节点的 y 坐标，则认为该手指伸出
        """
        tip = landmarks[tip_idx]
        pip = landmarks[pip_idx]
        return tip.y < pip.y

    def is_thumb_extended(self, landmarks):
        """
        判断拇指是否伸出：
        对于拇指，由于其运动方向主要是水平的，因此我们比较拇指尖和拇指掌指关节的x坐标
        """
        thumb_tip = landmarks[4]  # 拇指尖
        thumb_mcp = landmarks[2]  # 拇指掌指关节

        # 对于右手，如果拇指尖的x坐标小于拇指掌指关节的x坐标，表示拇指伸出
        # 注：此判断方法的前提是按习惯来讲人手心面对摄像头
        return thumb_tip.x < thumb_mcp.x


# 添加本地测试代码
if __name__ == "__main__":
    # 初始化手部检测器
    detector = HandDetector()

    # 初始化点击控制器
    click_controller = ClickController()

    # 禁用pyautogui安全功能
    pyautogui.FAILSAFE = False

    print("======== 实用的手势点击测试 ========")
    print("手势说明:")
    print("1. 拇指和食指捏合（伸出手指总数≤2） - 左键点击")
    print("2. 食指和中指捏合（拇指收起，总数≤2） - 右键点击")
    print("3. 伸出3+个手指时不会误触发点击，避免滚动冲突")
    print("按ESC退出")

    while True:
        # 获取手部数据
        frame, hand_info_list = detector.get_hand_data()
        if frame is None:
            continue

        # 处理点击控制
        frame = click_controller.process_clicks(frame, hand_info_list)

        # 显示测试窗口
        cv2.imshow("Practical Click Gesture Test", frame)

        # 按 Esc 键退出
        if cv2.waitKey(1) & 0xFF == 27:
            break

    # 释放资源
    detector.release()
    cv2.destroyAllWindows()