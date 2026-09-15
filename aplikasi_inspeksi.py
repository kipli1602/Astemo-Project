import sys
import time
import os
import re
import multiprocessing
import threading
import queue

# Matikan PIR API dan MKLDNN default dari Paddle (jika perlu untuk menghindari konflik)
os.environ["FLAGS_enable_pir_api"] = "0"
os.environ["FLAGS_use_mkldnn"] = "0"

import torch
torch.set_num_threads(4)
import cv2
import numpy as np
from ultralytics import YOLO

import json
# [IOT DISABLED] import urllib.request
# [IOT DISABLED] import threading

# Konfigurasi IP IoT (NodeMCU ESP8266)
# [IOT DISABLED] IOT_ESP_IP = "192.168.100.8"

# ============================================================
#  WORKER PADDLEOCR (MULTIPROCESSING)
# ============================================================
_OCR_BACKEND = None
_OCR_BACKEND_ERROR = None
_NVIDIA_DLL_HANDLES = []

def _add_nvidia_dll_dirs():
    """Tambahkan direktori DLL NVIDIA ke PATH agar PaddlePaddle-GPU dapat menemukan cuDNN"""
    venv_site_packages = os.path.join(
        os.path.dirname(__file__), ".venv", "Lib", "site-packages"
    )
    fallback_site = os.path.join(
        os.path.dirname(__file__), "Lib", "site-packages"
    )
    for base in [venv_site_packages, fallback_site]:
        for pattern in [
            "nvidia/cublas/bin",
            "nvidia/cuda_nvrtc/bin",
            "nvidia/cudnn/bin",
        ]:
            d = os.path.join(base, pattern)
            if os.path.isdir(d):
                _NVIDIA_DLL_HANDLES.append(os.add_dll_directory(d))
                os.environ["PATH"] = d + os.pathsep + os.environ["PATH"]

def _load_ocr_backend(use_gpu):
    if use_gpu:
        _add_nvidia_dll_dirs()
    import paddle
    from paddleocr import PaddleOCR
    return paddle, PaddleOCR


def _ocr_text(result):
    if not result or result[0] is None:
        return ''
    return ' '.join(line[1][0] for line in result[0])


def _matches_ocr_target(result, target):
    # Gunakan aturan keputusan akhir yang sama, termasuk format part number.
    return bool(target and fuzzy_match(_ocr_text(result), target, toleransi=1)[0])


def read_ocr_crop(ocr, img, target, prefer_processed=False):
    started = time.perf_counter()
    first = None
    second = None
    raw_seconds = 0.0
    prep_seconds = 0.0
    fallback_seconds = 0.0
    order = ('processed', 'raw') if prefer_processed else ('raw', 'processed')
    for variant in order:
        stage_started = time.perf_counter()
        if variant == 'raw':
            first = ocr.ocr(img, cls=False)
            raw_seconds = time.perf_counter() - stage_started
            result = first
        else:
            processed = preprocessing_gambar(img)
            prep_done = time.perf_counter()
            second = ocr.ocr(processed, cls=False)
            prep_seconds = prep_done - stage_started
            fallback_seconds = time.perf_counter() - prep_done
            result = second
        if _matches_ocr_target(result, target):
            break
    timings = dict(raw=raw_seconds, preprocessing=prep_seconds,
                   fallback=fallback_seconds, total=time.perf_counter() - started)
    return first, second, timings


def _ocr_worker(image_queue, result_queue, use_gpu):
    """OCR dan preprocessing tidak menahan thread kamera maupun YOLO."""
    try:
        if _OCR_BACKEND_ERROR:
            raise RuntimeError(_OCR_BACKEND_ERROR)
        paddle, PaddleOCR = _OCR_BACKEND or _load_ocr_backend(use_gpu)
        use_gpu = use_gpu and paddle.is_compiled_with_cuda()
        ocr = PaddleOCR(
            lang='en', use_angle_cls=False, det_db_thresh=0.3,
            rec_algorithm='SVTR_LCNet', show_log=False,
            use_gpu=use_gpu, enable_mkldnn=not use_gpu, cpu_threads=4,
        )
        result_queue.put(('ready', 'GPU' if use_gpu else 'CPU'))
    except Exception as exc:
        result_queue.put(('error', f'Inisialisasi OCR gagal: {exc}'))
        return

    preferred_processed = False
    previous_target = None
    while True:
        item = image_queue.get()
        if item is None:
            break
        job_id, img, target = item
        try:
            if target != previous_target:
                preferred_processed = False
                previous_target = target
            first, second, timings = read_ocr_crop(ocr, img, target, preferred_processed)
            if _matches_ocr_target(second, target):
                preferred_processed = True
            elif _matches_ocr_target(first, target):
                preferred_processed = False
            print(f"[OCR #{job_id}] " + ' '.join(
                f'{stage}={seconds:.3f}s' for stage, seconds in timings.items()), flush=True)
            print(f"[OCR #{job_id}] target={target!r} raw_text={_ocr_text(first)!r} "
                  f"processed_text={_ocr_text(second)!r} "
                  f"next_first={'processed' if preferred_processed else 'raw'}", flush=True)
            result_queue.put(('result', job_id, first, second, None))
        except Exception as exc:
            result_queue.put(('result', job_id, None, None, str(exc)))


class OcrManager:
    def __init__(self, use_gpu):
        ctx = multiprocessing.get_context('spawn')
        self.image_queue = ctx.Queue(maxsize=1)
        self.result_queue = ctx.Queue(maxsize=2)
        self.process = ctx.Process(
            target=_ocr_worker, name="astemo-ocr",
            args=(self.image_queue, self.result_queue, use_gpu), daemon=True,
        )
        self.process.start()

    def submit(self, job_id, img, target=''):
        try:
            self.image_queue.put_nowait((job_id, img.copy(), target))
            return True
        except queue.Full:
            return False

    def poll(self):
        try:
            return self.result_queue.get_nowait()
        except queue.Empty:
            return None

    def shutdown(self):
        try:
            self.image_queue.put_nowait(None)
        except queue.Full:
            pass
        self.process.join(timeout=2)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=2)
        for channel in (self.image_queue, self.result_queue):
            channel.cancel_join_thread()
            channel.close()

# ============================================================
#  FUNGSI UTAMA AI (YOLO) & UTILITAS
# ============================================================
def setup_device_ai():
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"\n[INFO] GPU Terdeteksi & Aktif: {gpu_name}")
        return True, 0  
    else:
        print("\n[INFO] Peringatan: GPU tidak tersedia, sistem beralih menggunakan CPU.")
        return False, 'cpu'

def preprocessing_gambar(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    gray = cv2.resize(gray, (w*2, h*2), interpolation=cv2.INTER_CUBIC)
    gray = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                cv2.THRESH_BINARY, 11, 2)
    kernel = np.ones((2,2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    return binary

def fuzzy_match(teks, target, toleransi=1):
    teks_bersih = re.sub(r'[^A-Z0-9]', '', teks.upper())
    target_bersih = target.upper()
    
    # 1. Exact match (Selalu dicek pertama)
    if target_bersih in teks_bersih:
        return True, 100
        
    # 2. Strict Mode untuk kode pendek (K80, K81, dll)
    # Jika panjang kode <= 4, JANGAN pakai toleransi kesalahan.
    # Karena beda 1 angka saja (K80 vs K81) artinya itu barang yang beda!
    # Lebih baik OCR mengulang bacaan daripada sistem meloloskan barang salah (False Positive).
    if len(target_bersih) <= 4:
        return False, 0
        
    # 3. Fuzzy Match untuk teks panjang (Toleransi Typo OCR)
    for i in range(len(teks_bersih) - len(target_bersih) + 1):
        substring = teks_bersih[i:i+len(target_bersih)]
        kesalahan = sum(1 for a, b in zip(substring, target_bersih) if a != b)
        if kesalahan <= toleransi:
            confidence = int(((len(target_bersih) - kesalahan) / len(target_bersih)) * 100)
            return True, confidence
            
    return False, 0


def enumerate_cameras(max_scan=10):
    """Scan index 0..max_scan dan kembalikan list of (index, nama_device) untuk kamera yang bisa dibuka."""
    cameras = []
    for i in range(max_scan):
        try:
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    name = cap.getBackendName()
                    try:
                        device_name = cap.get(cv2.CAP_PROP_DEVICE_NAME)
                        if device_name and str(device_name).strip():
                            name = str(device_name).strip()
                    except Exception:
                        pass
                    cameras.append((i, name if name else f"Camera {i}"))
                cap.release()
        except Exception:
            pass
    return cameras


# Windows spawn mengimpor ulang modul aplikasi. Muat backend OCR sebelum Qt,
# agar hook import PySide tidak memperlambat pemuatan SciPy/PaddleOCR.
if multiprocessing.current_process().name == 'astemo-ocr':
    try:
        _OCR_BACKEND = _load_ocr_backend(torch.cuda.is_available())
    except Exception as exc:
        _OCR_BACKEND_ERROR = str(exc)

from PySide6.QtCore import Qt, QThread, Signal, QTimer
from PySide6.QtGui import QImage, QPixmap, QFont, QColor
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QVBoxLayout, QHBoxLayout,
    QPushButton, QScrollArea, QFrame, QSizePolicy, QGraphicsDropShadowEffect,
    QInputDialog, QMessageBox, QComboBox
)

# ============================================================
#  THEME / QSS - Astemo Light Industrial Identity
# ============================================================
# --- Konstanta Warna UI ---
BACKGROUND       = "#F3F4F6"
SIDEBAR          = "#FFFFFF"
CARD             = "#FFFFFF"
BORDER           = "#D1D5DB"

ASTEMO_RED       = "#C9002B"
ASTEMO_RED_SOFT  = "#FDECEF"

TEXT_MAIN        = "#1F2937"
TEXT_MUTED       = "#6B7280"

MATCH_GREEN      = "#16A34A"
SCAN_YELLOW      = "#D97706"
NG_RED           = "#DC2626"

# --- QSS Styling ---
ASTEMO_QSS = f"""
QMainWindow, QDialog, QMessageBox, QInputDialog {{
    background-color: {BACKGROUND};
    color: {TEXT_MAIN};
    font-family: 'Segoe UI', 'Arial', sans-serif;
}}
QWidget {{
    color: {TEXT_MAIN};
    font-family: 'Segoe UI', 'Arial', sans-serif;
}}
QLineEdit {{
    background-color: #FFFFFF;
    border: 1px solid {BORDER};
    border-radius: 4px;
    padding: 6px;
    color: {TEXT_MAIN};
    font-size: 14px;
}}
QLineEdit:focus {{
    border: 1px solid {ASTEMO_RED};
}}
QDialog QPushButton, QMessageBox QPushButton {{
    background-color: #FFFFFF;
    border: 1px solid {BORDER};
    border-radius: 4px;
    padding: 6px 14px;
    color: {TEXT_MAIN};
    font-weight: bold;
}}
QDialog QPushButton:hover, QMessageBox QPushButton:hover {{
    background-color: {BACKGROUND};
}}
QLabel {{
    background-color: transparent;
}}
QScrollArea, QScrollArea > QWidget > QWidget {{
    background-color: transparent;
    border: none;
}}
QScrollBar:vertical {{
    border: none;
    background: transparent;
    width: 6px;
    margin: 0px;
}}
QScrollBar::handle:vertical {{
    background: {BORDER};
    border-radius: 3px;
    min-height: 20px;
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical,
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
    border: none;
    background: none;
    height: 0px;
}}
#CentralWidget {{
    background-color: {BACKGROUND};
}}
#Sidebar {{
    background-color: {SIDEBAR};
    border-right: 1px solid {BORDER};
}}
#LogoBadge {{
    background-color: {ASTEMO_RED};
    border-radius: 8px;
    color: #FFFFFF;
    font-size: 20px;
    font-weight: bold;
}}
#SidebarTitle {{
    color: {TEXT_MAIN};
    font-size: 15px;
    font-weight: 800;
    letter-spacing: 0.5px;
}}
#SidebarSubtitle {{
    color: {TEXT_MUTED};
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1px;
}}
#SectionLabel {{
    color: {TEXT_MUTED};
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 1.5px;
    padding: 10px 16px 6px 16px;
}}
QPushButton#TargetBtn {{
    background-color: transparent;
    border: none;
    border-left: 4px solid transparent;
    border-radius: 0px;
    color: {TEXT_MUTED};
    text-align: left;
    padding: 12px 16px;
    margin: 0px;
    font-size: 14px;
    font-weight: 700;
}}
QPushButton#TargetBtn:hover {{
    background-color: {BORDER};
    color: {TEXT_MAIN};
}}
QPushButton#TargetBtn[active="true"] {{
    background-color: #FBD5DB;
    border-left: 4px solid {ASTEMO_RED};
    color: {ASTEMO_RED};
}}
QPushButton#ExitBtn {{
    background-color: {ASTEMO_RED};
    color: #FFFFFF;
    border: none;
    border-radius: 6px;
    padding: 12px;
    font-size: 14px;
    font-weight: bold;
    margin: 16px;
}}
QPushButton#ExitBtn:hover {{
    background-color: #E60032;
}}
QPushButton#ManageBtn {{
    background-color: transparent;
    border: 1px solid {BORDER};
    border-radius: 6px;
    color: {TEXT_MUTED};
    padding: 8px;
    font-size: 12px;
    font-weight: bold;
}}
QPushButton#ManageBtn:hover {{
    background-color: {BORDER};
    color: {TEXT_MAIN};
}}
#TopBar {{
    background-color: {SIDEBAR};
    border-bottom: 1px solid {BORDER};
}}
#TopBarTitle {{
    color: {TEXT_MAIN};
    font-size: 16px;
    font-weight: bold;
}}
#KpiCard {{
    background-color: {CARD};
    border: none;
    border-radius: 8px;
}}
#KpiLabel {{
    color: {TEXT_MUTED};
    font-size: 13px;
    font-weight: bold;
    letter-spacing: 1px;
}}
#KpiValueTarget, #KpiValueOk, #KpiValueNg {{
    font-size: 34px;
    font-weight: bold;
}}
#KpiValueTarget {{ color: {TEXT_MAIN}; }}
#KpiValueOk {{ color: {MATCH_GREEN}; }}
#KpiValueNg {{ color: {NG_RED}; }}
#VideoFrame {{
    background-color: #000000;
    border: 2px solid {BORDER};
    border-radius: 8px;
}}
#StatusBar {{
    background-color: {SIDEBAR};
    border-top: 1px solid {BORDER};
    color: {TEXT_MUTED};
    padding: 10px 16px;
    font-size: 13px;
}}
QComboBox#CameraSelector {{
    background-color: #FFFFFF;
    border: 1px solid {BORDER};
    border-radius: 6px;
    color: {TEXT_MAIN};
    padding: 6px 10px;
    font-size: 12px;
    font-weight: bold;
}}
QComboBox#CameraSelector:hover {{
    border: 1px solid {ASTEMO_RED};
}}
QComboBox#CameraSelector::drop-down {{
    border: none;
    width: 24px;
}}
QComboBox#CameraSelector::down-arrow {{
    image: none;
    border-left: 5px solid transparent;
    border-right: 5px solid transparent;
    border-top: 6px solid {TEXT_MUTED};
    margin-right: 8px;
}}
QComboBox#CameraSelector QAbstractItemView {{
    background-color: #FFFFFF;
    border: 1px solid {BORDER};
    color: {TEXT_MAIN};
    selection-background-color: {ASTEMO_RED_SOFT};
    selection-color: {ASTEMO_RED};
    padding: 4px;
}}
"""


# ============================================================
#  VIDEO / AI THREAD
# ============================================================
class VideoCaptureThread(QThread):
    frame_ready = Signal(object)              
    stats_updated = Signal(int, int, int)          
    log_updated = Signal(str, str) # Modifikasi Signal: Menambah tipe status (MATCH/NG/INFO) untuk warna UI               

    def __init__(self, camera_index=2, loop_interval_ms=10, parent=None):
        super().__init__(parent)
        self.camera_index = camera_index
        self.loop_interval_ms = loop_interval_ms
        self._running = False
        
        # State Internal
        self.target_kode = "K81"
        self.memori_kardus = {}
        self.total_terdeteksi = 0
        self.frame_counter = 0

        self._stop_event = threading.Event()
        self._frames = queue.Queue(maxsize=1)
        self._state_lock = threading.Lock()
        self._target_revision = 0
        self._ai_revision = 0
        self._overlays = (0.0, 0, [])
        self._preview_pending = False
        self.ocr_manager = None
        self._ocr_ready = False
        self._ocr_failed = False
        self._pending_ocr = None
        self._job_counter = 0

    def set_target(self, kode: str):
        with self._state_lock:
            self.target_kode = kode
            self._target_revision += 1
            self._overlays = (0.0, self._target_revision, [])
        self.log_updated.emit(f"Target sistem diubah secara real-time ke: {kode}", "INFO")

    def run(self):
        cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        ai_thread = None
        try:
            if not cap.isOpened() and self.camera_index != 0:
                cap.release()
                cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            if not cap.isOpened():
                self.log_updated.emit("Kamera tidak dapat dibuka.", "INFO")
                return
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            ai_thread = threading.Thread(target=self._run_ai, daemon=True)
            ai_thread.start()
            self._running = True
            while not self._stop_event.is_set():
                ok, frame = cap.read()
                if not ok:
                    self.msleep(20)
                    continue
                # Satu frame terbaru: frame lama tidak menumpuk di belakang AI.
                try:
                    self._frames.get_nowait()
                except queue.Empty:
                    pass
                self._frames.put_nowait((time.monotonic(), frame.copy()))
                with self._state_lock:
                    timestamp, revision, overlays = self._overlays
                    revision_now = self._target_revision
                if revision == revision_now and time.monotonic() - timestamp < 0.5:
                    for (x1, y1, x2, y2), color, label in overlays:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                        cv2.putText(frame, label, (x1, max(20, y1 - 10)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                # Maksimal satu update UI tertunda.
                if not self._preview_pending:
                    self._preview_pending = True
                    self.frame_ready.emit(frame)
        finally:
            cap.release()
            self._running = False
            self._stop_event.set()
            if ai_thread is not None:
                ai_thread.join()

    def _run_ai(self):
        try:
            self.gunakan_gpu, self.device_yolo = setup_device_ai()
            self.log_updated.emit(f"YOLO: {'GPU' if self.gunakan_gpu else 'CPU'}; memuat OCR...", "INFO")
            self.model = YOLO(os.path.join(os.path.dirname(__file__), 'best.pt'))
            self.ocr_manager = OcrManager(use_gpu=self.gunakan_gpu)
            while not self._stop_event.is_set():
                try:
                    captured_at, frame = self._frames.get(timeout=0.1)
                except queue.Empty:
                    continue
                with self._state_lock:
                    revision, target = self._target_revision, self.target_kode
                if revision != self._ai_revision:
                    self.memori_kardus.clear()
                    self._ai_revision = revision
                self._active_target = target
                overlays = self.process_frame(frame)
                with self._state_lock:
                    if revision == self._target_revision:
                        self._overlays = (captured_at, revision, overlays)
                self.stats_updated.emit(
                    sum(v['status'] == 'Sesuai' for v in self.memori_kardus.values()),
                    sum(v['status'] == 'Tidak Sesuai' for v in self.memori_kardus.values()),
                    self.total_terdeteksi,
                )
        except Exception as exc:
            self.log_updated.emit(f"Proses AI gagal: {exc}", "INFO")
        finally:
            if self.ocr_manager is not None:
                self.ocr_manager.shutdown()

    def _poll_ocr(self):
        while True:
            message = self.ocr_manager.poll()
            if message is None:
                break
            if message[0] == 'ready':
                self._ocr_ready = True
                self.log_updated.emit(f"OCR siap: {message[1]}", "INFO")
            elif message[0] == 'error':
                self._ocr_failed = True
                self.log_updated.emit(message[1], "INFO")
            elif message[0] == 'result':
                _, job_id, first, second, error = message
                pending = self._pending_ocr
                if pending is None or pending[0] != job_id:
                    continue
                self._pending_ocr = None
                _, revision, box_id, state, submitted_at = pending
                print(f"[OCR #{job_id}] box={box_id} revision={revision} "
                      f"elapsed={time.monotonic()-submitted_at:.3f}s "
                      f"mismatch_count={state.get('gagal_hitung', 0)}", flush=True)
                if error:
                    self.log_updated.emit(f"OCR gagal: {error}", "INFO")
                    continue
                if (revision != self._ai_revision or revision != self._target_revision
                        or self.memori_kardus.get(box_id) is not state):
                    continue
                words = []
                for result in (first, second):
                    if result and result[0] is not None:
                        words.extend(line[1][0] for line in result[0])
                text = ' '.join(words)
                if text.strip():
                    match, confidence = fuzzy_match(text, self._active_target, toleransi=1)
                    state.update(teks=text, confidence=confidence)
                    if match:
                        state['status'] = 'Sesuai'
                        self.log_updated.emit(f"Kardus #{box_id} -> OCR: {text} -> MATCH", "MATCH")
                    else:
                        state['gagal_hitung'] += 1
                        if state['gagal_hitung'] > 4:
                            state['status'] = 'Tidak Sesuai'
                            self.log_updated.emit(f"Kardus #{box_id} -> OCR: {text} -> NG", "NG")
        if not self._ocr_failed:
            timed_out = self._pending_ocr and time.monotonic() - self._pending_ocr[4] > 30
            if not self.ocr_manager.process.is_alive() or timed_out:
                self._ocr_failed = True
                self._ocr_ready = False
                self.log_updated.emit("OCR berhenti atau melewati batas 30 detik. Mulai ulang kamera untuk mencoba lagi.", "INFO")

    def process_frame(self, frame: np.ndarray) -> list:
        self.frame_counter += 1
        self._poll_ocr()
        overlays = []
        
        # 1. Jalankan Tracking YOLO
        results = self.model.track(frame, persist=True, conf=0.6, tracker="botsort.yaml", device=self.device_yolo, verbose=False)
        id_aktif_di_frame = set()

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            ids = results[0].boxes.id.cpu().numpy()
            
            # Dahulukan kardus yang paling lama belum mendapat giliran OCR.
            candidates = sorted(zip(boxes, ids), key=lambda item:
                self.memori_kardus.get(int(item[1]), {}).get("last_ocr", 0))
            for box, id_kardus in candidates:
                x1, y1, x2, y2 = map(int, box)
                id_kardus = int(id_kardus)
                id_aktif_di_frame.add(id_kardus)
                
                # Registrasi kardus baru ke memori
                if id_kardus not in self.memori_kardus:
                    self.memori_kardus[id_kardus] = {
                        "status": "Scanning...", 
                        "teks": "", 
                        "confidence": 0, 
                        "gagal_hitung": 0,
                    }
                
                # Jadwalkan satu crop; hasil dibaca pada iterasi AI berikutnya.
                state = self.memori_kardus[id_kardus]
                if (state['status'] == 'Scanning...' and self._ocr_ready
                        and not self._ocr_failed and self._pending_ocr is None):
                    h, w = frame.shape[:2]
                    crop = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
                    if crop.size > 0:
                        self._job_counter += 1
                        if self.ocr_manager.submit(self._job_counter, crop, self._active_target):
                            state["last_ocr"] = self._job_counter
                            self._pending_ocr = (self._job_counter, self._ai_revision,
                                                 id_kardus, state, time.monotonic())

                # 3. Visualisasi Bounding Box (Warna sesuai request BGR: OpenCV)
                warna_kotak = (6, 119, 217) # BGR untuk SCAN_YELLOW (#D97706)
                status = self.memori_kardus[id_kardus]["status"]
                teks_ocr = self.memori_kardus[id_kardus]["teks"]
                confidence = self.memori_kardus[id_kardus]["confidence"]

                if status == "Sesuai":
                    warna_kotak = (74, 163, 22) # BGR untuk MATCH_GREEN (#16A34A)
                    teks_label = f"ID:{id_kardus} | {self._active_target} OK ({confidence}%)"
                elif status == "Tidak Sesuai":
                    warna_kotak = (38, 38, 220) # BGR untuk NG_RED (#DC2626)
                    teks_pendek = teks_ocr[:15] if len(teks_ocr) > 15 else teks_ocr
                    teks_label = f"ID:{id_kardus} | SALAH: {teks_pendek}"
                else:
                    teks_label = f"ID:{id_kardus} | Scanning..."

                overlays.append(((x1, y1, x2, y2), warna_kotak, teks_label))

        # Hapus kardus dari memori jika sudah hilang dari frame (agar memori tidak bocor)
        id_dalam_memori = list(self.memori_kardus.keys())
        for id_kardus in id_dalam_memori:
            if id_kardus not in id_aktif_di_frame:
                del self.memori_kardus[id_kardus]

        self.total_terdeteksi = len(id_aktif_di_frame)
        return overlays

    def stop(self):
        self._stop_event.set()
        self.wait()

# ============================================================
#  MAIN WINDOW (PySide6 Redesign)
# ============================================================
class InspectionDashboard(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Astemo — Industrial Visual Inspection System")
        self.showMaximized()

        # Database Setup
        self.db_path = os.path.join(os.path.dirname(__file__), "targets.json")
        self._load_targets()

        self.target_buttons = {}

        root = QWidget()
        root.setObjectName("CentralWidget")
        root_layout = QHBoxLayout(root)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)
        self.setCentralWidget(root)

        root_layout.addWidget(self._build_sidebar())
        root_layout.addWidget(self._build_main_area(), stretch=1)

        # --- start AI/video thread ---
        # Ambil index kamera pertama yang terdeteksi, fallback ke 0
        initial_cam = 0
        if self._camera_map:
            initial_cam = next(iter(self._camera_map.values()))

        self.video_thread = VideoCaptureThread(camera_index=initial_cam, loop_interval_ms=10)
        self.video_thread.frame_ready.connect(self.on_frame_ready)
        self.video_thread.stats_updated.connect(self.on_stats_updated)
        self.video_thread.log_updated.connect(self.on_log_updated)
        
        # SINKRONISASI: Pastikan AI mengecek kardus sesuai database target pertama (bukan hardcode K81)
        if self.TARGET_LIST:
            self.video_thread.set_target(self.TARGET_LIST[0])
            
        self.video_thread.start()

        # Jam real-time di top bar
        self.clock_timer = QTimer(self)
        self.clock_timer.timeout.connect(self._update_clock)
        self.clock_timer.start(1000)

    # ---------------- SIDEBAR ----------------
    def _build_sidebar(self) -> QWidget:
        sidebar = QFrame()
        sidebar.setObjectName("Sidebar")
        sidebar.setFixedWidth(220)

        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # --- Brand header: logo gambar Astemo ---
        brand = QWidget()
        brand_layout = QVBoxLayout(brand)
        brand_layout.setContentsMargins(16, 16, 16, 4)

        logo = QLabel()
        logo_path = os.path.join(os.path.dirname(__file__), "assets", "Logo.png")
        pixmap = QPixmap(logo_path)
        
        if not pixmap.isNull():
            # Lebarkan skala hingga 200px (mendekati lebar sidebar 232px) agar kualitas HD asli gambarnya bisa lebih nendang & tajam
            scaled_pixmap = pixmap.scaledToWidth(200, Qt.SmoothTransformation)
            logo.setPixmap(scaled_pixmap)
        else:
            logo.setText("ASTEMO")
            logo.setStyleSheet(f"color: {ASTEMO_RED}; font-size: 20px; font-weight: bold;")
            
        logo.setAlignment(Qt.AlignCenter)
        brand_layout.addWidget(logo)
        layout.addWidget(brand)

        # Garis pemisah (Separator)
        separator = QFrame()
        separator.setStyleSheet(f"background-color: {BORDER}; margin: 0px 16px;") 
        separator.setFixedHeight(1)
        layout.addWidget(separator)
        
        layout.addSpacing(4) # Memberikan jarak antara garis dengan teks TARGET INSPEKSI di bawahnya

        section_lbl = QLabel("TARGET INSPEKSI")
        section_lbl.setObjectName("SectionLabel")
        layout.addWidget(section_lbl)

        self.list_container = QWidget()
        self.list_layout = QVBoxLayout(self.list_container)
        self.list_layout.setContentsMargins(0, 4, 0, 0)
        self.list_layout.setSpacing(2)
        
        self.target_buttons = {}
        
        # Populate Target UI First Time
        # It needs video_thread to be none initially or we check hasattr
        self._refresh_target_ui()

        layout.addWidget(self.list_container)
        
        # Stretch diletakkan di bawah list agar list menempel di atas
        layout.addStretch()

        # --- KAMERA SELECTOR ---
        cam_section_lbl = QLabel("KAMERA")
        cam_section_lbl.setObjectName("SectionLabel")
        layout.addWidget(cam_section_lbl)

        cam_row = QHBoxLayout()
        cam_row.setContentsMargins(16, 0, 16, 4)

        self.camera_combo = QComboBox()
        self.camera_combo.setObjectName("CameraSelector")
        self.camera_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.camera_combo.setMinimumHeight(36)

        detected = enumerate_cameras()
        self._camera_map = {}  # displayName -> index
        for idx, name in detected:
            display = f"[{idx}] {name}"
            self.camera_combo.addItem(display)
            self._camera_map[display] = idx

        if not detected:
            self.camera_combo.addItem("Tidak ada kamera")
            self._camera_map = {}

        self.camera_combo.currentTextChanged.connect(self._on_camera_changed)
        cam_row.addWidget(self.camera_combo)
        layout.addLayout(cam_row)

        layout.addSpacing(8)

        # Tombol Manage Target (Tambah / Hapus)
        manage_layout = QHBoxLayout()
        manage_layout.setContentsMargins(16, 8, 16, 8)
        
        add_btn = QPushButton("[+] Tambah")
        add_btn.setObjectName("ManageBtn")
        add_btn.setCursor(Qt.PointingHandCursor)
        add_btn.clicked.connect(self.on_add_target)
        
        del_btn = QPushButton("[-] Hapus")
        del_btn.setObjectName("ManageBtn")
        del_btn.setCursor(Qt.PointingHandCursor)
        del_btn.clicked.connect(self.on_delete_target)
        
        manage_layout.addWidget(add_btn)
        manage_layout.addWidget(del_btn)
        
        layout.addLayout(manage_layout)

        exit_btn = QPushButton("EXIT")
        exit_btn.setObjectName("ExitBtn")
        exit_btn.setCursor(Qt.PointingHandCursor)
        exit_btn.clicked.connect(self.close)
        layout.addWidget(exit_btn)

        return sidebar

    # ---------------- MAIN AREA ----------------
    def _build_main_area(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(16)

        layout.addWidget(self._build_top_bar())
        layout.addLayout(self._build_kpi_row())
        layout.addWidget(self._build_video_panel(), stretch=1)
        layout.addWidget(self._build_status_bar())

        return container

    def _build_top_bar(self) -> QWidget:
        bar = QFrame()
        bar.setObjectName("TopBar")
        bar.setFixedHeight(50)
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(16, 0, 16, 0)

        # Perubahan UI: Menambah judul aplikasi di Top Bar
        title = QLabel("ASTEMO VISUAL INSPECTION SYSTEM")
        title.setObjectName("TopBarTitle")
        layout.addWidget(title)

        layout.addSpacing(20)

        live_dot = QLabel("● LIVE")
        live_dot.setStyleSheet(f"color:{MATCH_GREEN}; font-weight:bold; font-size:14px;")
        layout.addWidget(live_dot)
        
        layout.addStretch()

        self.clock_label = QLabel("")
        self.clock_label.setStyleSheet(f"color:{TEXT_MUTED}; font-weight:bold; font-size:16px;")
        layout.addWidget(self.clock_label)

        return bar

    def _build_kpi_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(16)

        self.kpi_target = self._make_kpi_card("TARGET HARI INI", self.TARGET_LIST[0], "KpiValueTarget")
        self.kpi_terdeteksi = self._make_kpi_card("Total Box", "0", "KpiValueTarget")
        self.kpi_ok = self._make_kpi_card("Total Benar", "0", "KpiValueOk")
        self.kpi_ng = self._make_kpi_card("Total Salah", "0", "KpiValueNg")

        row.addWidget(self.kpi_target[0])
        row.addWidget(self.kpi_terdeteksi[0])
        row.addWidget(self.kpi_ok[0])
        row.addWidget(self.kpi_ng[0])
        return row

    def _make_kpi_card(self, label_text, value_text, value_style_name):
        card = QFrame()
        card.setObjectName("KpiCard")
        card.setFixedHeight(90)
        
        shadow = QGraphicsDropShadowEffect(card)
        shadow.setBlurRadius(15)
        shadow.setColor(QColor(0, 0, 0, 15))
        shadow.setOffset(0, 4)
        card.setGraphicsEffect(shadow)
        
        layout = QVBoxLayout(card)
        layout.setContentsMargins(20, 12, 20, 12)

        lbl = QLabel(label_text)
        lbl.setObjectName("KpiLabel")
        val = QLabel(value_text)
        val.setObjectName(value_style_name)

        layout.addWidget(lbl)
        layout.addWidget(val)
        return card, val   

    def _build_video_panel(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        # Header kecil di atas kamera (Sesuai request)
        header = QLabel("LIVE INSPECTION")
        header.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 12px; font-weight: bold; letter-spacing: 1px;")
        layout.addWidget(header)

        self.video_label = QLabel()
        self.video_label.setObjectName("VideoFrame")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_label.setMinimumSize(320, 240)  
        self.video_label.setText("Menunggu Inisialisasi Kamera & AI...")
        self.video_label.setStyleSheet(self.video_label.styleSheet() + f"color:{TEXT_MUTED};")
        layout.addWidget(self.video_label, stretch=1)
        
        return container

    def _build_status_bar(self) -> QWidget:
        self.status_label = QLabel("SYSTEM READY")
        self.status_label.setObjectName("StatusBar")
        return self.status_label

    # ---------------- DATABASE & UI REFRESH ----------------
    def _load_targets(self):
        print(f"[DEBUG] Memuat database target dari: {self.db_path}")
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, "r") as f:
                    self.TARGET_LIST = json.load(f)
                print(f"[DEBUG] Target berhasil dimuat: {self.TARGET_LIST}")
            except Exception as e:
                print(f"[DEBUG] Error loading database: {e}")
                self.TARGET_LIST = ["K81", "K80", "K59", "K93"]
        else:
            self.TARGET_LIST = ["K81", "K80", "K59", "K93"]
            self._save_targets()

        if not self.TARGET_LIST:
            self.TARGET_LIST = ["TARGET_KOSONG"]

    def _save_targets(self):
        print(f"[DEBUG] Menyimpan database target... Isi: {self.TARGET_LIST}")
        try:
            with open(self.db_path, "w") as f:
                json.dump(self.TARGET_LIST, f, indent=4)
            print("[DEBUG] Database berhasil disimpan.")
        except Exception as e:
            print(f"[DEBUG] Error saving database: {e}")

    def _refresh_target_ui(self):
        # Bersihkan list tombol yang lama
        while self.list_layout.count():
            item = self.list_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
        
        self.target_buttons.clear()
        
        # Dapatkan target aktif saat ini
        active_target = self.video_thread.target_kode if hasattr(self, 'video_thread') else self.TARGET_LIST[0]
        
        # Buat ulang tombol-tombol target
        for kode in self.TARGET_LIST:
            btn = QPushButton(f"   {kode}")
            btn.setObjectName("TargetBtn")
            btn.setCursor(Qt.PointingHandCursor)
            btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            btn.setMinimumHeight(42)
            btn.setProperty("active", kode == active_target)
            btn.clicked.connect(lambda checked=False, k=kode: self.on_target_selected(k))
            self.target_buttons[kode] = btn
            self.list_layout.addWidget(btn)

    # ---------------- SLOTS / EVENT HANDLERS ----------------
    def on_add_target(self):
        print("[DEBUG] Membuka dialog Tambah Target...")
        text, ok = QInputDialog.getText(self, "Tambah Target", "Masukkan ID Target Baru (Contoh: K99):")
        if ok and text:
            kode = text.strip().upper()
            print(f"[DEBUG] Input user diterima: {kode}")
            if kode and kode not in self.TARGET_LIST:
                self.TARGET_LIST.append(kode)
                print(f"[DEBUG] Target '{kode}' ditambahkan ke list.")
                self._save_targets()
                self._refresh_target_ui()
            else:
                print(f"[DEBUG] Target '{kode}' kosong atau sudah ada di list.")
        else:
            print("[DEBUG] User membatalkan dialog Tambah Target.")

    def on_delete_target(self):
        if not hasattr(self, 'video_thread'): return
        kode = self.video_thread.target_kode
        print(f"[DEBUG] Meminta konfirmasi penghapusan target: {kode}")
        reply = QMessageBox.question(self, "Konfirmasi Hapus", 
                                     f"Apakah Anda yakin ingin menghapus target '{kode}' secara permanen?",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            print(f"[DEBUG] User mengonfirmasi penghapusan target '{kode}'.")
            if kode in self.TARGET_LIST:
                self.TARGET_LIST.remove(kode)
                self._save_targets()
                
                # Otomatis pindah ke target pertama jika ada
                if self.TARGET_LIST:
                    print(f"[DEBUG] Beralih ke target berikutnya: {self.TARGET_LIST[0]}")
                    self.on_target_selected(self.TARGET_LIST[0])
                else:
                    print("[DEBUG] Database kosong. Mengalihkan ke mode 'KOSONG'.")
                    self.TARGET_LIST = ["KOSONG"]
                    self.on_target_selected("KOSONG")
                    
                self._refresh_target_ui()
        else:
            print("[DEBUG] User membatalkan penghapusan.")

    def on_target_selected(self, kode: str):
        print(f"[DEBUG] Target dipilih dari UI: {kode}")
        if hasattr(self, 'video_thread'):
            self.video_thread.set_target(kode)
        if hasattr(self, 'kpi_target'):
            self.kpi_target[1].setText(kode)
            
        for k, btn in self.target_buttons.items():
            btn.setProperty("active", k == kode)
            btn.style().unpolish(btn)
            btn.style().polish(btn)

    def _on_camera_changed(self, display_name: str):
        if display_name not in self._camera_map:
            return
        new_index = self._camera_map[display_name]
        print(f"[DEBUG] Pengguna memilih kamera: index {new_index} ({display_name})")
        self._switch_camera(new_index)

    def _switch_camera(self, new_index: int):
        if not hasattr(self, 'video_thread'):
            return
        if self.video_thread.camera_index == new_index:
            return

        current_target = self.video_thread.target_kode

        self.video_thread.stop()

        self.video_thread = VideoCaptureThread(camera_index=new_index, loop_interval_ms=10)
        self.video_thread.frame_ready.connect(self.on_frame_ready)
        self.video_thread.stats_updated.connect(self.on_stats_updated)
        self.video_thread.log_updated.connect(self.on_log_updated)
        if current_target:
            self.video_thread.set_target(current_target)
        self.video_thread.start()

        self.status_label.setText(f"Beralih ke kamera index {new_index}")
        self.status_label.setStyleSheet(f"color: {TEXT_MUTED}; font-weight: normal;")

    def on_frame_ready(self, frame: np.ndarray):
        source = self.sender()
        if source is not None:
            source._preview_pending = False
            if source is not self.video_thread:
                return
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg).scaled(
            self.video_label.width(), self.video_label.height(),
            Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.video_label.setPixmap(pixmap)

    def on_stats_updated(self, total_ok: int, total_ng: int, total_terdeteksi: int):
        self.kpi_ok[1].setText(str(total_ok))
        self.kpi_ng[1].setText(str(total_ng))
        self.kpi_terdeteksi[1].setText(str(total_terdeteksi))

    def on_log_updated(self, message: str, msg_type: str):
        # Perubahan UI: Memberikan warna pada log status bar sesuai kejadian
        if msg_type == "MATCH":
            self.status_label.setStyleSheet(f"color: {MATCH_GREEN}; font-weight: bold;")
            # [IOT DISABLED] threading.Thread(target=self._trigger_iot_alarm, args=("/hijau",), daemon=True).start()
        elif msg_type == "NG":
            self.status_label.setStyleSheet(f"color: {NG_RED}; font-weight: bold;")
            # [IOT DISABLED] threading.Thread(target=self._trigger_iot_alarm, args=("/merah",), daemon=True).start()
        else:
            self.status_label.setStyleSheet(f"color: {TEXT_MUTED}; font-weight: normal;")
            
        self.status_label.setText(message)

    # [IOT DISABLED] def _trigger_iot_alarm(self, endpoint):
        # [IOT DISABLED] """Menembak sinyal IoT HTTP GET ke NodeMCU tanpa membuat antarmuka (UI) freeze/lag."""
        # [IOT DISABLED] url = f"http://{IOT_ESP_IP}{endpoint}"
        # [IOT DISABLED] try:
            # [IOT DISABLED] # Timeout 1 detik sudah cukup untuk kirim perintah ringan di jaringan lokal
            # [IOT DISABLED] urllib.request.urlopen(url, timeout=1.0)
            # [IOT DISABLED] print(f"[DEBUG] IoT Relay Sukses dieksekusi: {url}")
        # [IOT DISABLED] except Exception as e:
            # [IOT DISABLED] print(f"[DEBUG] IoT Relay Gagal terhubung ke {url} -> {e}")

    def _update_clock(self):
        self.clock_label.setText(time.strftime("%H:%M:%S"))

    def closeEvent(self, event):
        self.video_thread.stop()
        event.accept()

# ============================================================
#  ENTRY POINT
# ============================================================
if __name__ == "__main__":
    # SANGAT PENTING: Mencegah infinite loop spawning di OS Windows saat pakai Multiprocessing
    multiprocessing.freeze_support()
    
    app = QApplication(sys.argv)
    
    # Terapkan QSS ke seluruh aplikasi (termasuk popup/dialog)
    app.setStyleSheet(ASTEMO_QSS)
    
    window = InspectionDashboard()
    window.show()
    sys.exit(app.exec())
