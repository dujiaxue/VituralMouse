import cv2
import win32gui
import win32con
import pyautogui
import mediapipe as mp
import time

from handdetect import HandDetector
from mouse import process_mouse_control
from scroll import process_scroll_control
from click_judge import ClickController


def make_window_always_on_top(window_name):
    """将指定窗口置顶"""
    try:
        hwnd = win32gui.FindWindow(None, window_name)
        if hwnd:
            win32gui.SetWindowPos(hwnd, win32con.HWND_TOPMOST, 0, 0, 0, 0,
                                  win32con.SWP_NOMOVE | win32con.SWP_NOSIZE)
    except Exception:
        pass


def adjust_hand_info(hand_info_list):
    """调整手部信息格式，确保与mouse.py期望的格式一致"""
    adjusted_list = []

    for hand in hand_info_list:
        # 复制原始手部信息
        adjusted_hand = hand.copy()

        # 修改标签格式，从"Right"变为"Right Hand"，从"Left"变为"Left Hand"
        original_label = hand.get('label', '')
        if original_label == "Right":
            adjusted_hand['label'] = "Right Hand"
        elif original_label == "Left":
            adjusted_hand['label'] = "Left Hand"

        adjusted_list.append(adjusted_hand)

    return adjusted_list


def main():
    """主函数"""
    # 初始化手部检测器
    detector = HandDetector(min_detection_confidence=0.65, min_tracking_confidence=0.5)
    window_name = "Virtual Mouse"

    # 初始化MediaPipe手势模型
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # 初始化点击控制器
    click_controller = ClickController()

    # 创建窗口
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 640, 480)

    # 初始化鼠标控制数据
    last_mouse_data = {
        'last_x': None,
        'last_y': None,
        'smooth_mouse_x': None,
        'smooth_mouse_y': None,
        'smoothing_factor': 0.15,
        'frame_counter': 0
    }

    # 设置pyautogui不延迟并关闭安全检查
    pyautogui.PAUSE = 0
    pyautogui.FAILSAFE = False

    try:
        while True:
            # 获取当前帧和手部信息
            frame, hand_info_list = detector.get_hand_data()
            if frame is None:
                continue

            # 调整手部信息格式以匹配mouse.py的期望
            adjusted_hand_info = adjust_hand_info(hand_info_list)

            # 尝试将窗口置顶
            make_window_always_on_top(window_name)

            # 1. 处理点击操作
            frame = click_controller.process_clicks(frame, adjusted_hand_info)

            # 2. 处理鼠标移动
            if adjusted_hand_info:
                frame, last_mouse_data = process_mouse_control(frame, adjusted_hand_info, last_mouse_data)
            else:
                # 重置鼠标数据
                last_mouse_data['last_x'] = None
                last_mouse_data['last_y'] = None
                last_mouse_data['smooth_mouse_x'] = None
                last_mouse_data['smooth_mouse_y'] = None
                last_mouse_data['frame_counter'] = 0

            # 3. 处理滚动功能
            if adjusted_hand_info:
                frame = process_scroll_control(frame, adjusted_hand_info)

            # 检查窗口是否被关闭
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break

            # 显示当前帧
            cv2.imshow(window_name, frame)

            # 键盘检测
            key = cv2.waitKey(1)
            if key == 27:  # ESC键退出
                break

    except Exception as e:
        print(f"发生错误: {e}")
    finally:
        # 释放资源
        detector.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
