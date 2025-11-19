import cv2
import pyautogui

# 获取当前屏幕分辨率
screen_width, screen_height = pyautogui.size()


def process_mouse_control(frame, hand_info_list, last_data=None):
    """
    处理鼠标控制逻辑，根据右手食指位置平滑移动鼠标

    Args:
        frame: 当前视频帧
        hand_info_list: 检测到的手部信息列表
        last_data: 上一帧保存的鼠标平滑数据（包含上一次坐标、平滑坐标、平滑因子等）

    Returns:
        frame: 处理后的图像帧（绘制了食指位置的圆点）
        last_data: 更新后的平滑数据
    """
    # 初始化上一次数据字典
    if last_data is None:
        last_data = {
            'last_x': None,
            'last_y': None,
            'smooth_mouse_x': None,
            'smooth_mouse_y': None,
            'smoothing_factor': 0.3,
            'frame_counter': 0
        }

    last_x = last_data['last_x']
    last_y = last_data['last_y']
    smooth_mouse_x = last_data['smooth_mouse_x']
    smooth_mouse_y = last_data['smooth_mouse_y']
    smoothing_factor = last_data['smoothing_factor']
    frame_counter = last_data['frame_counter']

    # 定义控制区域（归一化坐标）
    control_x_min, control_y_min = 0.1, 0.1
    control_x_max, control_y_max = 0.9, 0.9
    min_stable_frames = 3  # 稳定帧数阈值
    h, w, _ = frame.shape
    right_hand_found = False

    # 遍历检测到的手部信息，寻找右手
    for hand in hand_info_list:
        if hand.get('label', "") == "Right Hand":
            # 确保关键点数量足够
            if len(hand['landmarks']) > 8:
                index_finger = hand['landmarks'][8]
                # 将归一化坐标转换为像素坐标
                finger_x = int(index_finger.x * w)
                finger_y = int(index_finger.y * h)

                # 根据上一次位置调整平滑因子
                if last_x is not None and last_y is not None:
                    move_x = finger_x - last_x
                    move_y = finger_y - last_y
                    movement_speed = (move_x ** 2 + move_y ** 2) ** 0.5

                    if movement_speed > 50:
                        smoothing_factor = 0.6
                    elif movement_speed < 10:
                        smoothing_factor = 0.2
                    else:
                        smoothing_factor = 0.3
                else:
                    move_x, move_y = 0, 0

                # 更新上一次位置
                last_x, last_y = finger_x, finger_y
                relative_finger_x = index_finger.x
                relative_finger_y = index_finger.y

                # 将归一化坐标映射到控制区域，并限制在 [0, 1] 范围内
                normalized_x = (relative_finger_x - control_x_min) / (control_x_max - control_x_min)
                normalized_y = (relative_finger_y - control_y_min) / (control_y_max - control_y_min)
                normalized_x = max(0, min(1, normalized_x))
                normalized_y = max(0, min(1, normalized_y))

                # 如果手指位于屏幕上边缘一定范围内，则置为0（不移动鼠标）
                OFFSET_TOP = 0.05
                if normalized_y < OFFSET_TOP:
                    normalized_y = 0

                # 计算目标鼠标坐标
                target_mouse_x = int(normalized_x * screen_width)
                target_mouse_y = int(normalized_y * screen_height)

                # 平滑移动：第一次直接赋值，否则根据平滑因子逐步调整
                if smooth_mouse_x is None or smooth_mouse_y is None:
                    smooth_mouse_x, smooth_mouse_y = target_mouse_x, target_mouse_y
                else:
                    smooth_mouse_x = int(smooth_mouse_x + smoothing_factor * (target_mouse_x - smooth_mouse_x))
                    smooth_mouse_y = int(smooth_mouse_y + smoothing_factor * (target_mouse_y - smooth_mouse_y))

                right_hand_found = True
                frame_counter += 1

                # 当检测到的连续帧数超过阈值后，执行鼠标移动操作
                if frame_counter >= min_stable_frames:
                    pyautogui.moveTo(smooth_mouse_x, smooth_mouse_y)

                # 在当前帧绘制一个圆圈标记食指位置
                cv2.circle(frame, (finger_x, finger_y), 10, (255, 0, 255), -1)
            break

    # 如果未检测到右手，则重置稳定帧数和平滑数据
    if not right_hand_found:
        frame_counter = 0
        smooth_mouse_x, smooth_mouse_y = None, None

    # 更新 last_data 字典
    last_data.update({
        'last_x': last_x,
        'last_y': last_y,
        'smooth_mouse_x': smooth_mouse_x,
        'smooth_mouse_y': smooth_mouse_y,
        'smoothing_factor': smoothing_factor,
        'frame_counter': frame_counter
    })

    return frame, last_data


# 添加本地测试代码
if __name__ == "__main__":
    import mediapipe as mp
    from handdetect import HandDetector

    # 初始化手部检测器
    detector = HandDetector()
    last_mouse_data = None

    while True:
        # 获取手部数据
        frame, hand_info_list = detector.get_hand_data()
        if frame is None:
            continue

        # 处理鼠标控制
        frame, last_mouse_data = process_mouse_control(frame, hand_info_list, last_mouse_data)

        # 显示测试窗口
        cv2.imshow("Mouse Control Test", frame)

        # 退出
        if cv2.waitKey(1) & 0xFF == 27:
            break

    # 释放资源
    detector.release()
