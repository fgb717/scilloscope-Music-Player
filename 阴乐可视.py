import sys
import wave
import numpy as np
import os
from PyQt6 import QtWidgets, uic, QtCore
from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from PyQt6.QtGui import QPainter, QColor, QPen, QBrush
from pyaudio import PyAudio
from pydub import AudioSegment
from io import BytesIO
import time

# 获取当前脚本所在目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# ffmpeg 路径：脚本同目录/ffmpeg/ffmpeg.exe
FFMPEG_PATH = os.path.join(BASE_DIR, "ffmpeg", "ffmpeg.exe")
# UI 文件路径：脚本同目录/ffmpeg/untitled.ui
UI_PATH = os.path.join(BASE_DIR, "ffmpeg", "untitled.ui")

# 配置 pydub 使用本地 ffmpeg
AudioSegment.converter = FFMPEG_PATH

class OscilloscopeXY(QOpenGLWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.xy_data = np.zeros((512, 2))
        self.dot_mode = False

        # 翻转控制
        self.flip_vertical = False
        self.flip_horizontal = False

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update)
        self.timer.start(16)

    def update_xy(self, stereo_data):
        self.xy_data = stereo_data

    def set_dot_mode(self, enable):
        self.dot_mode = enable

    def set_flip_vertical(self, enable):
        self.flip_vertical = enable

    def set_flip_horizontal(self, enable):
        self.flip_horizontal = enable

    def paintGL(self):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(0, 0, 0))
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w = self.width()
        h = self.height()
        cx = w / 2
        cy = h / 2
        scale = min(w, h) * 0.45

        points = []
        for x, y in self.xy_data:
            if self.flip_horizontal:
                x = -x
            if self.flip_vertical:
                y = -y
            points.append(QtCore.QPointF(cx + x * scale, cy - y * scale))

        if self.dot_mode:
            painter.setPen(QPen(QColor(0, 255, 0), 2))
            painter.setBrush(QBrush(QColor(0, 255, 0)))
            for p in points:
                painter.drawEllipse(p, 1.5, 1.5)
        else:
            painter.setPen(QPen(QColor(0, 255, 0), 1))
            if len(points) > 1:
                painter.drawPolyline(points)

class AudioXYThread(QtCore.QThread):
    xy_ready = QtCore.pyqtSignal(np.ndarray)
    progress_update = QtCore.pyqtSignal(int)
    total_ms_ready = QtCore.pyqtSignal(int)
    finished = QtCore.pyqtSignal()

    def __init__(self, path, chunk_size=1024):
        super().__init__()
        self.path = path
        self.run_flag = True
        self.pause_flag = False
        self.mutex = QtCore.QMutex()

        self.wf = None
        self.sample_rate = 44100
        self.total_frames = 0
        self.duration_ms = 0
        self.current_frame = 0
        self.chunk = chunk_size
        self.audio = None

    def load_audio(self):
        ext = self.path.lower().split('.')[-1]
        self.audio = AudioSegment.from_file(self.path, format=ext)
        self.audio = self.audio.set_channels(2)
        
        self.duration_ms = len(self.audio)
        
        wav_io = BytesIO()
        self.audio.export(wav_io, format='wav')
        wav_io.seek(0)
        self.wf = wave.open(wav_io, 'rb')

        self.sample_rate = self.wf.getframerate()
        self.total_frames = self.wf.getnframes()
        
        self.total_ms_ready.emit(self.duration_ms)

    def set_chunk_size(self, new_chunk):
        self.mutex.lock()
        self.chunk = new_chunk
        self.mutex.unlock()

    def run(self):
        try:
            self.load_audio()
            p = PyAudio()
            stream = p.open(
                format=p.get_format_from_width(self.wf.getsampwidth()),
                channels=2,
                rate=self.sample_rate,
                output=True
            )

            while self.run_flag:
                self.mutex.lock()
                if self.pause_flag:
                    self.mutex.unlock()
                    time.sleep(0.01)
                    continue
                current_chunk = self.chunk
                self.mutex.unlock()

                data = self.wf.readframes(current_chunk)
                if not data:
                    break

                stream.write(data)
                self.current_frame += current_chunk
                progress = int(self.current_frame / self.total_frames * 1000)
                self.progress_update.emit(progress)

                arr = np.frombuffer(data, dtype=np.int16).reshape(-1, 2)
                arr = arr / 32768.0
                self.xy_ready.emit(arr)

            stream.stop_stream()
            stream.close()
            p.terminate()
            self.wf.close()
            self.finished.emit()
        except Exception as e:
            print("播放错误:", e)

    def pause(self):
        self.mutex.lock()
        self.pause_flag = True
        self.mutex.unlock()

    def resume(self):
        self.mutex.lock()
        self.pause_flag = False
        self.mutex.unlock()

    def stop(self):
        self.run_flag = False
        self.wait()

class OscilloscopeApp:
    def __init__(self, ui_path):
        self.ui = uic.loadUi(ui_path)
        self.gl_widget = OscilloscopeXY()
        layout = QtWidgets.QVBoxLayout(self.ui.xianshi)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.gl_widget)

        self.ui.xuanze.clicked.connect(self.select_file)
        self.ui.kaishi.clicked.connect(self.toggle_play_pause)
        self.ui.jindu.sliderPressed.connect(self.on_slider_pressed)
        self.ui.jindu.sliderReleased.connect(self.on_slider_released)
        self.ui.dian.toggled.connect(self.on_dot_mode_toggled)

        self.ui.shangxia.setAutoExclusive(False)
        self.ui.zuoyou.setAutoExclusive(False)
        self.ui.shangxia.clicked.connect(self.on_shangxia_click)
        self.ui.zuoyou.clicked.connect(self.on_zuoyou_click)

        self.ui.huanchong.setRange(128, 4096)
        self.ui.huanchong.setValue(1024)
        self.ui.huanchong.valueChanged.connect(self.on_chunk_changed)

        self.audio_thread = None
        self.is_playing = False
        self.is_sliding = False
        self.total_time_str = "00:00"
        self.current_audio_path = ""

    def on_shangxia_click(self):
        self.gl_widget.set_flip_vertical(self.ui.shangxia.isChecked())

    def on_zuoyou_click(self):
        self.gl_widget.set_flip_horizontal(self.ui.zuoyou.isChecked())

    def on_chunk_changed(self, value):
        if self.audio_thread:
            self.audio_thread.set_chunk_size(value)

    def on_dot_mode_toggled(self, checked):
        self.gl_widget.set_dot_mode(checked)

    def format_time(self, ms):
        seconds = ms // 1000
        minutes = seconds // 60
        secs = seconds % 60
        return f"{minutes:02d}:{secs:02d}"

    def set_total_time(self, ms):
        self.total_time_str = self.format_time(ms)
        self.ui.shichang.setText(f"00:00/{self.total_time_str}")

    def update_progress(self, progress):
        if not self.is_sliding and self.audio_thread:
            self.ui.jindu.setValue(progress)
            current_ms = int(progress / 1000 * self.audio_thread.duration_ms)
            current_str = self.format_time(current_ms)
            self.ui.shichang.setText(f"{current_str}/{self.total_time_str}")

    def select_file(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self.ui, "选择示波器音乐", "", "音频文件 (*.wav *.mp3 *.flac)"
        )
        if not path:
            return

        if self.audio_thread:
            self.audio_thread.stop()

        self.current_audio_path = path
        self._start_new_audio()

    def _start_new_audio(self):
        chunk = self.ui.huanchong.value()
        self.audio_thread = AudioXYThread(self.current_audio_path, chunk)
        self.audio_thread.total_ms_ready.connect(self.set_total_time)
        self.audio_thread.xy_ready.connect(self.gl_widget.update_xy)
        self.audio_thread.progress_update.connect(self.update_progress)
        self.audio_thread.finished.connect(self.on_audio_finished)
        self.audio_thread.start()

        self.ui.jindu.setRange(0, 1000)
        self.ui.jindu.setValue(0)
        self.is_playing = True
        self.ui.kaishi.setText("暂停")

    def toggle_play_pause(self):
        if not self.current_audio_path:
            return

        if not self.is_playing:
            if self.audio_thread and not self.audio_thread.isRunning():
                self._start_new_audio()
                return

        if self.is_playing:
            self.audio_thread.pause()
            self.ui.kaishi.setText("播放")
        else:
            self.audio_thread.resume()
            self.ui.kaishi.setText("暂停")
        self.is_playing = not self.is_playing

    def on_slider_pressed(self):
        self.is_sliding = True

    def on_slider_released(self):
        if self.audio_thread and self.audio_thread.isRunning():
            val = self.ui.jindu.value()
            target_frame = int(val / 1000 * self.audio_thread.total_frames)
            self.audio_thread.current_frame = target_frame
            self.audio_thread.wf.setpos(target_frame)
        self.is_sliding = False

    def on_audio_finished(self):
        self.is_playing = False
        self.ui.kaishi.setText("播放")
        self.ui.jindu.setValue(0)
        self.ui.shichang.setText(f"00:00/{self.total_time_str}")

    def show(self):
        self.ui.show()

def main():
    app = QtWidgets.QApplication(sys.argv)
    # 使用自动配置的UI路径
    window = OscilloscopeApp(UI_PATH)
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()