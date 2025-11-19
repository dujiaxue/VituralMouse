import tkinter as tk
from tkinter import ttk
import cv2
from PIL import Image, ImageTk
import threading
import time
import mediapipe as mp
import numpy as np

class FloatingHandWindow:
    """悬浮窗手部显示窗口"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("手部监控")
        self.root.geometry("320x240+100+100")
        
        # 设置窗口属性 - 悬浮窗关键设置
        self.root.attributes('-topmost', True)  # 窗口置顶
        self.root.overrideredirect(False)  # 显示窗口边框，方便移动
        self.root.attributes('-alpha', 0.95)  # 轻微透明度
        
        # 摄像头相关
        self.cap = None
        self.is_running = False
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.6
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # 创建UI
        self.create_ui()
        self.bind_events()
        
    def create_ui(self):
        """创建用户界面"""
        # 主框架
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 标题栏
        title_frame = ttk.Frame(main_frame)
        title_frame.pack(fill=tk.X, pady=(0, 5))
        
        title_label = ttk.Label(title_frame, text="手部监控", font=('Arial', 10, 'bold'))
        title_label.pack(side=tk.LEFT, padx=5)
        
        # 控制按钮
        close_btn = ttk.Button(title_frame, text="×", command=self.on_closing, width=2)
        close_btn.pack(side=tk.RIGHT, padx=2)
        
        minimize_btn = ttk.Button(title_frame, text="_", command=self.minimize_window, width=2)
        minimize_btn.pack(side=tk.RIGHT, padx=2)
        
        # 视频显示
        self.video_label = ttk.Label(main_frame, background='black', relief=tk.SUNKEN)
        self.video_label.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # 状态栏
        status_frame = ttk.Frame(main_frame)
        status_frame.pack(fill=tk.X, pady=(5, 0))
        
        self.status_label = ttk.Label(status_frame, text="状态: 未连接", font=('Arial', 8))
        self.status_label.pack(side=tk.LEFT, padx=5)
        
        self.hands_label = ttk.Label(status_frame, text="手部: 0", font=('Arial', 8))
        self.hands_label.pack(side=tk.RIGHT, padx=5)
        
        # 控制按钮
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(5, 0))
        
        self.connect_btn = ttk.Button(button_frame, text="连接摄像头", command=self.toggle_camera)
        self.connect_btn.pack(fill=tk.X, padx=5)
        
        self.always_on_top_btn = ttk.Button(button_frame, text="取消置顶", command=self.toggle_always_on_top)
        self.always_on_top_btn.pack(fill=tk.X, padx=5, pady=(5, 0))
        
    def bind_events(self):
        """绑定窗口事件"""
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # 使窗口可拖动
        self.root.bind('<Button-1>', self.start_move)
        self.root.bind('<B1-Motion>', self.on_move)
        
    def start_move(self, event):
        """开始拖动窗口"""
        self.x = event.x
        self.y = event.y
        
    def on_move(self, event):
        """拖动窗口"""
        deltax = event.x - self.x
        deltay = event.y - self.y
        x = self.root.winfo_x() + deltax
        y = self.root.winfo_y() + deltay
        self.root.geometry(f"+{x}+{y}")
        
    def minimize_window(self):
        """最小化窗口"""
        self.root.iconify()
        
    def toggle_always_on_top(self):
        """切换置顶状态"""
        current_state = self.root.attributes('-topmost')
        new_state = not current_state
        self.root.attributes('-topmost', new_state)
        
        if new_state:
            self.always_on_top_btn.config(text="取消置顶")
        else:
            self.always_on_top_btn.config(text="置顶窗口")
            
    def toggle_camera(self):
        """切换摄像头连接状态"""
        if not self.is_running:
            self.start_camera()
        else:
            self.stop_camera()
            
    def start_camera(self):
        """启动摄像头"""
        try:
            self.cap = cv2.VideoCapture(0)
            if self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
                self.cap.set(cv2.CAP_PROP_FPS, 30)
                
                self.is_running = True
                self.connect_btn.config(text="断开摄像头")
                self.status_label.config(text="状态: 已连接")
                
                # 启动视频线程
                self.video_thread = threading.Thread(target=self.update_video)
                self.video_thread.daemon = True
                self.video_thread.start()
                
                print("摄像头连接成功")
            else:
                self.status_label.config(text="状态: 连接失败")
                print("摄像头打开失败")
                
        except Exception as e:
            self.status_label.config(text=f"状态: 错误")
            print(f"摄像头连接失败: {e}")
            
    def stop_camera(self):
        """停止摄像头"""
        self.is_running = False
        
        if self.cap:
            self.cap.release()
            self.cap = None
            
        self.connect_btn.config(text="连接摄像头")
        self.status_label.config(text="状态: 未连接")
        self.hands_label.config(text="手部: 0")
        self.video_label.config(image='')
        
        print("摄像头已断开")
        
    def update_video(self):
        """更新视频流"""
        while self.is_running:
            try:
                if self.cap and self.cap.isOpened():
                    ret, frame = self.cap.read()
                    if ret:
                        # 处理帧
                        processed_frame = self.process_frame(frame)
                        
                        # 转换为tkinter格式
                        image = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                        image = cv2.resize(image, (300, 200))
                        
                        # 转换为PIL图像
                        pil_image = Image.fromarray(image)
                        
                        # 转换为PhotoImage
                        photo = ImageTk.PhotoImage(image=pil_image)
                        
                        # 更新标签
                        self.video_label.config(image=photo)
                        self.video_label.image = photo  # 保持引用
                        
            except Exception as e:
                print(f"视频更新错误: {e}")
                
            time.sleep(0.033)  # 约30fps
            
    def process_frame(self, frame):
        """处理视频帧"""
        # 水平翻转（镜像效果）
        frame = cv2.flip(frame, 1)
        
        # 转换为RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 手部检测
        results = self.hands.process(frame_rgb)
        
        hand_count = 0
        
        if results.multi_hand_landmarks:
            hand_count = len(results.multi_hand_landmarks)
            
            for hand_landmarks in results.multi_hand_landmarks:
                # 绘制手部关键点
                self.mp_drawing.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2),
                    self.mp_drawing.DrawingSpec(color=(0, 125, 255), thickness=2)
                )
                
        # 更新手部计数
        self.root.after(0, self.update_hand_count, hand_count)
        
        return frame
        
    def update_hand_count(self, count):
        """更新手部计数显示"""
        self.hands_label.config(text=f"手部: {count}")
        
    def on_closing(self):
        """窗口关闭处理"""
        self.stop_camera()
        self.root.quit()
        self.root.destroy()
        
    def run(self):
        """运行应用"""
        self.root.mainloop()

if __name__ == "__main__":
    app = FloatingHandWindow()
    app.run()