from flask import Flask, render_template, Response, jsonify, request
import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import time

app = Flask(__name__)

# 全局变量
detector = None
mouse_controller = None
click_controller = None
scroll_controller = None
is_running = False
system_status = {
    'camera_connected': False,
    'hand_detection_active': False,
    'mouse_control_active': False,
    'click_control_active': False,
    'scroll_control_active': False,
    'fps': 0,
    'hands_detected': 0
}

class WebHandDetector:
    """Web版手部检测器"""
    def __init__(self):
        self.settings = {
            'mouse_control': True,
            'click_control': True,
            'scroll_control': True,
            'hand_detection': True,
            'mouse_smooth_factor': 0.3,
            'click_cooldown': 0.5,
            'min_detection_confidence': 0.7,
            'min_tracking_confidence': 0.6
        }
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=self.settings['min_detection_confidence'],
            min_tracking_confidence=self.settings['min_tracking_confidence']
        )
        self.cap = None
        self.hands_info = []
        
    def initialize_camera(self):
        """初始化摄像头"""
        try:
            self.cap = cv2.VideoCapture(0)
            if self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.cap.set(cv2.CAP_PROP_FPS, 30)
                print("摄像头初始化成功")
                return True
            print("摄像头打开失败")
            return False
        except Exception as e:
            print(f"摄像头初始化失败: {e}")
            return False
    
    def get_frame(self):
        """获取处理后的视频帧"""
        if not self.cap or not self.cap.isOpened():
            return None
            
        ret, frame = self.cap.read()
        if not ret:
            return None
            
        # 水平翻转（镜像效果）
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 手部检测
        hands_info = []
        if self.settings.get('hand_detection', True):
            results = self.hands.process(frame_rgb)
            if results.multi_hand_landmarks:
                for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    # 绘制手部关键点
                    self.mp_drawing.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2),
                        self.mp_drawing.DrawingSpec(color=(0, 125, 255), thickness=2)
                    )
                    
                    # 获取手部信息
                    height, width, _ = frame.shape
                    x_coords = [int(lm.x * width) for lm in hand_landmarks.landmark]
                    y_coords = [int(lm.y * height) for lm in hand_landmarks.landmark]
                    
                    detected_type = results.multi_handedness[idx].classification[0].label
                    confidence = results.multi_handedness[idx].classification[0].score
                    
                    hands_info.append({
                        'landmarks': hand_landmarks.landmark,
                        'label': detected_type,
                        'confidence': confidence,
                        'bbox': (min(x_coords), min(y_coords), max(x_coords), max(y_coords)),
                        'center': ((min(x_coords) + max(x_coords)) // 2, (min(y_coords) + max(y_coords)) // 2)
                    })
                    
                    # 显示手部信息
                    hand_text = f"{detected_type} ({confidence:.2f})"
                    cv2.putText(frame, hand_text, (min(x_coords), min(y_coords) - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # 去掉左上角的文字显示
        # 原来的代码：
        # cv2.putText(frame, f"Virtual Mouse - Web", (10, 30), 
        #            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        # cv2.putText(frame, f"Hands: {len(hands_info)}", (10, 60), 
        #            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        
        self.hands_info = hands_info
        return frame, hands_info
    
    def release(self):
        """释放资源"""
        if self.cap:
            self.cap.release()

class MouseController:
    """鼠标控制器"""
    def __init__(self):
        pyautogui.PAUSE = 0
        pyautogui.FAILSAFE = False
        self.screen_width, self.screen_height = pyautogui.size()
        self.last_mouse_data = {
            'last_x': None, 'last_y': None,
            'smooth_mouse_x': None, 'smooth_mouse_y': None,
            'smoothing_factor': 0.15, 'frame_counter': 0
        }
    
    def process_mouse_control(self, frame, hand_info):
        """处理鼠标控制"""
        if not hand_info or not detector.settings.get('mouse_control', True):
            return frame
            
        for hand in hand_info:
            if hand.get('label') == "Right":
                landmarks = hand.get('landmarks', [])
                if len(landmarks) > 8:
                    index_finger = landmarks[8]
                    
                    h, w, _ = frame.shape
                    finger_x = int(index_finger.x * w)
                    finger_y = int(index_finger.y * h)
                    
                    screen_x = int(index_finger.x * self.screen_width)
                    screen_y = int(index_finger.y * self.screen_height)
                    
                    smooth_factor = detector.settings.get('mouse_smooth_factor', 0.3)
                    
                    if self.last_mouse_data['smooth_mouse_x'] is None:
                        self.last_mouse_data['smooth_mouse_x'] = screen_x
                        self.last_mouse_data['smooth_mouse_y'] = screen_y
                    else:
                        smooth_x = int(self.last_mouse_data['smooth_mouse_x'] * (1 - smooth_factor) + screen_x * smooth_factor)
                        smooth_y = int(self.last_mouse_data['smooth_mouse_y'] * (1 - smooth_factor) + screen_y * smooth_factor)
                        
                        self.last_mouse_data['smooth_mouse_x'] = smooth_x
                        self.last_mouse_data['smooth_mouse_y'] = smooth_y
                        
                        pyautogui.moveTo(smooth_x, smooth_y)
                    
                    cv2.circle(frame, (finger_x, finger_y), 10, (255, 0, 255), -1)
                    break
        
        return frame

class ClickController:
    """点击控制器 - 方案一实现"""
    def __init__(self):
        self.last_click_time = 0
        self.click_cooldown = 0.5
    
    def is_finger_extended(self, landmarks, tip_idx, pip_idx):
        """判断手指是否伸出"""
        tip = landmarks[tip_idx]
        pip = landmarks[pip_idx]
        return tip.y < pip.y
    
    def is_thumb_extended(self, landmarks):
        """判断拇指是否伸出"""
        thumb_tip = landmarks[4]
        thumb_mcp = landmarks[2]
        return thumb_tip.x < thumb_mcp.x
    
    def process_clicks(self, frame, hand_info):
        """处理点击事件"""
        if not hand_info or not detector.settings.get('click_control', True):
            return frame
            
        current_time = time.time()
        h, w, _ = frame.shape
        
        for hand in hand_info:
            if hand.get('label') == "Right":
                landmarks = hand.get('landmarks', [])
                if len(landmarks) > 20:
                    # 判断各手指是否伸出
                    thumb_extended = self.is_thumb_extended(landmarks)
                    index_extended = self.is_finger_extended(landmarks, 8, 6)
                    middle_extended = self.is_finger_extended(landmarks, 12, 10)
                    ring_extended = self.is_finger_extended(landmarks, 16, 14)
                    pinky_extended = self.is_finger_extended(landmarks, 20, 18)
                    
                    # 获取关键点坐标
                    thumb_x = int(landmarks[4].x * w)
                    thumb_y = int(landmarks[4].y * h)
                    index_x = int(landmarks[8].x * w)
                    index_y = int(landmarks[8].y * h)
                    middle_x = int(landmarks[12].x * w)
                    middle_y = int(landmarks[12].y * h)
                    
                    # 计算距离
                    thumb_index_distance = ((thumb_x - index_x) ** 2 + (thumb_y - index_y) ** 2) ** 0.5
                    index_middle_distance = ((index_x - middle_x) ** 2 + (index_y - middle_y) ** 2) ** 0.5
                    
                    # 统计伸出的手指数量
                    extended_fingers_count = sum([thumb_extended, index_extended, middle_extended, ring_extended, pinky_extended])
                    
                    click_cooldown = detector.settings.get('click_cooldown', 0.5)
                    
                    # 左键：拇指食指捏合，且不是在滚动状态（伸出手指数≤2）
                    if (thumb_index_distance < 40 and extended_fingers_count <= 2 and 
                        current_time - self.last_click_time > click_cooldown):
                        
                        center_x = (thumb_x + index_x) // 2
                        center_y = (thumb_y + index_y) // 2
                        cv2.circle(frame, (center_x, center_y), 15, (0, 255, 0), cv2.FILLED)
                        cv2.putText(frame, "LEFT", (center_x - 25, center_y - 20),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        
                        pyautogui.click()
                        self.last_click_time = current_time
                        print("左键点击")
                    
                    # 右键：食指中指捏合，但要求拇指收起且总手指数不超过2个
                    elif (index_middle_distance < 30 and not thumb_extended and extended_fingers_count <= 2 and
                          current_time - self.last_click_time > click_cooldown):
                        
                        center_x = (index_x + middle_x) // 2
                        center_y = (index_y + middle_y) // 2
                        cv2.circle(frame, (center_x, center_y), 15, (0, 0, 255), cv2.FILLED)
                        cv2.putText(frame, "RIGHT", (center_x - 30, center_y - 20),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                        
                        pyautogui.rightClick()
                        self.last_click_time = current_time
                        print("右键点击")
                
                break
        
        return frame

class ScrollController:
    """滚动控制器 - 恢复原始的动态速度逻辑"""
    def __init__(self):
        self.last_scroll_time = 0
        self.last_speed = 0
        self.speed_history = []
        self.history_size = 5
        self.min_scroll_interval = 0.04
        self.smoothing_factor = 0.3
    
    def is_finger_extended(self, landmarks, tip_idx, pip_idx):
        """判断手指是否伸出"""
        tip = landmarks[tip_idx]
        pip = landmarks[pip_idx]
        return tip.y < pip.y
    
    def is_thumb_extended(self, landmarks):
        """判断拇指是否伸出"""
        thumb_tip = landmarks[4]
        thumb_mcp = landmarks[2]
        return thumb_tip.x < thumb_mcp.x
    
    def compute_scroll_speed(self, landmarks, indices, min_speed=5, max_speed=60):
        """根据手指伸展程度计算滚动速度"""
        diffs = [abs(landmarks[tip].y - landmarks[pip].y) for tip, pip in indices]
        avg_diff = sum(diffs) / len(diffs)
        
        normalized = (avg_diff - 0.02) / (0.15 - 0.02)
        normalized = max(0, min(normalized, 1))
        normalized = normalized ** 1.5
        
        speed = int(min_speed + normalized * (max_speed - min_speed))
        return speed
    
    def smooth_speed(self, new_speed):
        """平滑滚动速度"""
        self.speed_history.append(new_speed)
        if len(self.speed_history) > self.history_size:
            self.speed_history.pop(0)
        
        smoothed_speed = self.last_speed * (1 - self.smoothing_factor) + new_speed * self.smoothing_factor
        self.last_speed = smoothed_speed
        return int(smoothed_speed)
    
    def process_scroll(self, frame, hand_info):
        """处理滚动事件"""
        if not hand_info or not detector.settings.get('scroll_control', True):
            return frame
            
        current_time = time.time()
        
        for hand in hand_info:
            if hand.get('label') == "Right":
                landmarks = hand.get('landmarks', [])
                if len(landmarks) > 20:
                    thumb_extended = self.is_thumb_extended(landmarks)
                    index_extended = self.is_finger_extended(landmarks, 8, 6)
                    middle_extended = self.is_finger_extended(landmarks, 12, 10)
                    ring_extended = self.is_finger_extended(landmarks, 16, 14)
                    pinky_extended = self.is_finger_extended(landmarks, 20, 18)
                    
                    upward_indices = [(8, 6), (12, 10), (16, 14)]
                    downward_indices = [(8, 6), (12, 10), (16, 14), (20, 18)]
                    
                    if current_time - self.last_scroll_time >= self.min_scroll_interval:
                        self.last_scroll_time = current_time
                        
                        # 向上滚动：拇指不伸出 + 中间三指伸出 + 小指不伸出
                        if not thumb_extended and index_extended and middle_extended and ring_extended and not pinky_extended:
                            raw_speed = self.compute_scroll_speed(landmarks, upward_indices, 5, 60)
                            smooth_speed = self.smooth_speed(raw_speed)
                            actual_speed = smooth_speed // 2 if smooth_speed < 10 else smooth_speed
                            pyautogui.scroll(actual_speed)
                            print(f"向上滚动: {actual_speed}")
                        
                        # 向下滚动：拇指不伸出 + 除拇指外所有手指伸出
                        elif not thumb_extended and index_extended and middle_extended and ring_extended and pinky_extended:
                            raw_speed = self.compute_scroll_speed(landmarks, downward_indices, 5, 60)
                            smooth_speed = self.smooth_speed(raw_speed)
                            actual_speed = smooth_speed // 2 if smooth_speed < 10 else smooth_speed
                            pyautogui.scroll(-actual_speed)
                            print(f"向下滚动: {actual_speed}")
                
                break
        
        return frame

def generate_frames():
    """生成视频流帧"""
    global detector, mouse_controller, click_controller, scroll_controller, system_status
    
    while is_running:
        try:
            result = detector.get_frame()
            if result is not None:
                frame, hands_info = result
                
                system_status['hands_detected'] = len(hands_info)
                system_status['hand_detection_active'] = bool(len(hands_info) > 0)
                system_status['mouse_control_active'] = detector.settings.get('mouse_control', True) and system_status['hand_detection_active']
                system_status['click_control_active'] = detector.settings.get('click_control', True) and system_status['hand_detection_active']
                system_status['scroll_control_active'] = detector.settings.get('scroll_control', True) and system_status['hand_detection_active']
                
                if hands_info:
                    frame = click_controller.process_clicks(frame, hands_info)
                    frame = mouse_controller.process_mouse_control(frame, hands_info)
                    frame = scroll_controller.process_scroll(frame, hands_info)
                
                ret, buffer = cv2.imencode('.jpg', frame)
                frame_bytes = buffer.tobytes()
                
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            else:
                time.sleep(0.1)
        except Exception as e:
            print(f"视频流错误: {e}")
            time.sleep(0.1)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def get_status():
    return jsonify(system_status)

@app.route('/control', methods=['POST'])
def control():
    global is_running, detector, mouse_controller, click_controller, scroll_controller
    
    try:
        data = request.get_json(force=True, silent=True)
        if not data:
            data = request.json
        
        if not data:
            return jsonify({'success': False, 'message': '无法解析请求数据'})
        
        action = data.get('action')
        
        if action == 'start':
            if not is_running:
                detector = WebHandDetector()
                if detector.initialize_camera():
                    mouse_controller = MouseController()
                    click_controller = ClickController()
                    scroll_controller = ScrollController()
                    is_running = True
                    
                    system_status.update({
                        'camera_connected': True,
                        'mouse_control_active': True,
                        'click_control_active': True,
                        'scroll_control_active': True
                    })
                    
                    return jsonify({'success': True, 'message': '系统已启动'})
                else:
                    return jsonify({'success': False, 'message': '摄像头连接失败'})
        
        elif action == 'stop':
            is_running = False
            if detector:
                detector.release()
            
            system_status.update({
                'camera_connected': False,
                'hand_detection_active': False,
                'mouse_control_active': False,
                'click_control_active': False,
                'scroll_control_active': False
            })
            
            return jsonify({'success': True, 'message': '系统已停止'})
        
        return jsonify({'success': False, 'message': f'无效的操作: {action}'})
            
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': f'服务器错误: {str(e)}'})

@app.route('/settings', methods=['POST'])
def settings():
    try:
        data = request.get_json(force=True, silent=True)
        if not data:
            data = request.json
            
        if not data:
            return jsonify({'success': False, 'message': '无效的请求数据'})
            
        setting_name = data.get('setting')
        value = data.get('value')
        
        if detector and setting_name in detector.settings:
            detector.settings[setting_name] = value
            
            if setting_name in ['mouse_control', 'click_control', 'scroll_control', 'hand_detection']:
                has_hands = system_status.get('hands_detected', 0) > 0
                hand_detection_enabled = detector.settings.get('hand_detection', True)
                
                system_status['hand_detection_active'] = has_hands and hand_detection_enabled
                system_status['mouse_control_active'] = detector.settings.get('mouse_control', True) and system_status['hand_detection_active']
                system_status['click_control_active'] = detector.settings.get('click_control', True) and system_status['hand_detection_active']
                system_status['scroll_control_active'] = detector.settings.get('scroll_control', True) and system_status['hand_detection_active']
            
            return jsonify({'success': True, 'message': f'设置 {setting_name} 已更新'})
        else:
            return jsonify({'success': False, 'message': '无效的设置项'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

if __name__ == '__main__':
    app.run(debug=False, host='127.0.0.1', port=5000)