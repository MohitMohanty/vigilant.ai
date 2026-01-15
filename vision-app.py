import sys
import cv2
import threading
import ollama
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QTextEdit, QLineEdit, QLabel, QPushButton, QSizePolicy)
from PyQt6.QtCore import QTimer, Qt, pyqtSignal, QObject
from PyQt6.QtGui import QImage, QPixmap, QFont

class AISignals(QObject):
    description_ready = pyqtSignal(str)

class VisionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RTX 3050 Scene Intelligence")
        self.showMaximized() 

        self.current_prompt = "Describe what is happening in this scene in detail."
        self.is_analyzing = False
        self.signals = AISignals()
        self.signals.description_ready.connect(self.update_description)

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)

        # --- LEFT COLUMN: Video Feed ---
        self.video_label = QLabel("Loading Webcam...")
        self.video_label.setStyleSheet("background-color: black; border: 2px solid #333;")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # FIX: Ignore size hints from the image to stop "zooming"
        self.video_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        
        layout.addWidget(self.video_label, stretch=3)

        # --- RIGHT COLUMN: Controls ---
        right_panel = QVBoxLayout()
        
        prompt_label = QLabel("Enter Custom Prompt:")
        prompt_label.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        right_panel.addWidget(prompt_label)
        
        self.prompt_input = QLineEdit()
        self.prompt_input.setText(self.current_prompt)
        right_panel.addWidget(self.prompt_input)

        self.send_btn = QPushButton("Update AI Task")
        self.send_btn.clicked.connect(self.update_prompt_from_ui)
        right_panel.addWidget(self.send_btn)

        right_panel.addSpacing(20)

        desc_label = QLabel("AI Scene Analysis:")
        desc_label.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        right_panel.addWidget(desc_label)
        
        self.description_output = QTextEdit()
        self.description_output.setReadOnly(True)
        self.description_output.setFont(QFont("Segoe UI", 12))
        right_panel.addWidget(self.description_output, stretch=1)

        layout.addLayout(right_panel, stretch=1)

        # Setup Camera
        self.cap = cv2.VideoCapture(0)
        
        # High-speed UI Timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

        # AI Update Timer
        self.ai_timer = QTimer()
        self.ai_timer.timeout.connect(self.trigger_ai_analysis)
        self.ai_timer.start(4000)

    def update_prompt_from_ui(self):
        self.current_prompt = self.prompt_input.text()
        self.description_output.append("\n<b>[System]:</b> Prompt updated.")

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            self.last_frame = frame.copy()
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            
            # Create QImage from current frame
            qt_image = QImage(rgb_image.data, w, h, ch * w, QImage.Format.Format_RGB888)
            
            # FIX: Only scale the Pixmap to fit the label's *current* size
            # This prevents the label from expanding to match the image
            pixmap = QPixmap.fromImage(qt_image)
            scaled_pixmap = pixmap.scaled(
                self.video_label.width(), 
                self.video_label.height(), 
                Qt.AspectRatioMode.KeepAspectRatio, 
                Qt.TransformationMode.SmoothTransformation
            )
            self.video_label.setPixmap(scaled_pixmap)

    def trigger_ai_analysis(self):
        if not self.is_analyzing and hasattr(self, 'last_frame'):
            self.is_analyzing = True
            threading.Thread(target=self.run_ollama_inference, daemon=True).start()

    def run_ollama_inference(self):
        try:
            _, buffer = cv2.imencode('.jpg', self.last_frame)
            response = ollama.generate(model='llava', prompt=self.current_prompt, images=[buffer.tobytes()])
            self.signals.description_ready.emit(response['response'])
        except Exception as e:
            self.signals.description_ready.emit(f"AI Error: {e}")
        finally:
            self.is_analyzing = False

    def update_description(self, text):
        self.description_output.setPlainText(text)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VisionApp()
    window.show()
    sys.exit(app.exec())