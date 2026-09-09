import sys
import time
import os
import re
import multiprocessing

# Matikan PIR API dan MKLDNN default dari Paddle (jika perlu untuk menghindari konflik)
os.environ["FLAGS_enable_pir_api"] = "0"
os.environ["FLAGS_use_mkldnn"] = "0"

import torch
torch.set_num_threads(12)
import cv2
import numpy as np
from ultralytics import YOLO

from PySide6.QtCore import Qt, QThread, Signal, QTimer
from PySide6.QtGui import QImage, QPixmap, QFont, QColor
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QVBoxLayout, QHBoxLayout,
    QPushButton, QScrollArea, QFrame, QSizePolicy, QGraphicsDropShadowEffect,
    QInputDialog, QMessageBox
)
import json
import urllib.request
import threading

# Konfigurasi IP IoT (NodeMCU ESP8266)
IOT_ESP_IP = "192.168.100.8"

# ============================================================
#  WORKER PADDLEOCR (MULTIPROCESSING)
# ============================================================
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
                os.add_dll_directory(d)
                os.environ["PATH"] = d + os.pathsep + os.environ["PATH"]

def _ocr_worker(image_queue, result_queue, use_gpu):
    """Worker process untuk PaddleOCR — berjalan di proses terpisah agar tidak bertabrakan DLL dengan PyTorch"""
    if use_gpu:
        _add_nvidia_dll_dirs()
        
    from paddleocr import PaddleOCR

    try:
        ocr = PaddleOCR(
            use_textline_orientation=False,
            lang='en',
            use_angle_cls=True,
            det_db_thresh=0.3,
            rec_algorithm='SVTR_LCNet',
            show_log=False,
            use_gpu=use_gpu,
            enable_mkldnn=not use_gpu,  # Tetap aktifkan MKL-DNN jika pakai CPU
            cpu_threads=12
        )
    except Exception:
        ocr = PaddleOCR(use_textline_orientation=False, lang='en', use_angle_cls=True, show_log=False)

    while True:
        item = image_queue.get()
        if item is None:
            break

        img, img_processed = item
        try:
            hasil_ocr1 = ocr.ocr(img, cls=False)
            hasil_ocr2 = ocr.ocr(img_processed, cls=False)
            result_queue.put((hasil_ocr1, hasil_ocr2))
        except Exception:
            result_queue.put((None, None))

class OcrManager:
    """Mengelola proses worker PaddleOCR secara terpisah untuk menghindari konflik DLL pada Windows"""
    def __init__(self, use_gpu):
        ctx = multiprocessing.get_context('spawn')
        self.image_queue = ctx.Queue()
        self.result_queue = ctx.Queue()
        self.process = ctx.Process(
            target=_ocr_worker,
            args=(self.image_queue, self.result_queue, use_gpu),
            daemon=True
        )
        self.process.start()

    def ocr(self, img, img_processed):
        """Kirim gambar ke worker proses, terima hasil OCR"""
        self.image_queue.put((img, img_processed))
        return self.result_queue.get()

    def shutdown(self):
        self.image_queue.put(None)
        self.process.join(timeout=5)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=2)

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
        self.total_sesuai = 0
        self.total_nyasar = 0
        self.total_terdeteksi = 0
        self.frame_counter = 0

        # Inisialisasi Model AI (YOLO + OcrManager)
        self.log_updated.emit("Memuat model YOLO dan OCR (Multiprocessing)... Harap tunggu.", "INFO")
        self.gunakan_gpu, self.device_yolo = setup_device_ai()
        self.model = YOLO("best.pt")
        
        # Menggunakan OCR Manager (Worker Terpisah)
        self.ocr_manager = OcrManager(use_gpu=self.gunakan_gpu)
        
        self.log_updated.emit("SYSTEM READY", "INFO")

    def set_target(self, kode: str):
        self.target_kode = kode
        self.memori_kardus.clear()
        self.total_sesuai = 0
        self.total_nyasar = 0
        self.log_updated.emit(f"Target sistem diubah secara real-time ke: {kode}", "INFO")

    def run(self):
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            print(f"[PERINGATAN] USB Webcam index {self.camera_index} tidak ditemukan. Beralih ke 0.")
            cap = cv2.VideoCapture(0)
            
        # Atur resolusi kamera ke 720p (16:9) agar layar penuh
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            
        self._running = True

        while self._running:
            loop_start = time.time()
            ok, frame = cap.read()
            if not ok:
                continue

            frame = self.process_frame(frame)
            
            self.frame_ready.emit(frame.copy())
            self.stats_updated.emit(self.total_sesuai, self.total_nyasar, self.total_terdeteksi)

            elapsed_ms = (time.time() - loop_start) * 1000
            sleep_ms = max(0, self.loop_interval_ms - elapsed_ms)
            self.msleep(int(sleep_ms))

        cap.release()

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        self.frame_counter += 1
        
        # 1. Jalankan Tracking YOLO
        results = self.model.track(frame, persist=True, conf=0.6, tracker="botsort.yaml", device=self.device_yolo, verbose=False)
        id_aktif_di_frame = set()

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            ids = results[0].boxes.id.cpu().numpy()
            
            for box, id_kardus in zip(boxes, ids):
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
                        "counted": False
                    }
                
                # 2. Lakukan OCR setiap 5 frame jika masih "Scanning..."
                if self.memori_kardus[id_kardus]["status"] == "Scanning..." and (self.frame_counter % 5 == 0):
                    potongan_stiker = frame[y1:y2, x1:x2]
                    if potongan_stiker.size > 0:
                        img_processed = preprocessing_gambar(potongan_stiker)
                        
                        # Memanggil OCR melalui Manager (Worker Process)
                        hasil_ocr1, hasil_ocr2 = self.ocr_manager.ocr(potongan_stiker, img_processed)
                        
                        semua_teks = []
                        if hasil_ocr1 and hasil_ocr1[0] is not None:
                            semua_teks.extend([baris[1][0] for baris in hasil_ocr1[0]])
                        if hasil_ocr2 and hasil_ocr2[0] is not None:
                            semua_teks.extend([baris[1][0] for baris in hasil_ocr2[0]])
                        
                        teks_lengkap = " ".join(semua_teks)
                        
                        if teks_lengkap.strip():
                            cocok, confidence = fuzzy_match(teks_lengkap, self.target_kode, toleransi=1)
                            self.memori_kardus[id_kardus]["teks"] = teks_lengkap
                            self.memori_kardus[id_kardus]["confidence"] = confidence
                            
                            if cocok:
                                self.memori_kardus[id_kardus]["status"] = "Sesuai"
                                if not self.memori_kardus[id_kardus]["counted"]:
                                    self.total_sesuai += 1
                                    self.memori_kardus[id_kardus]["counted"] = True
                                    self.log_updated.emit(f"Kardus #{id_kardus} → OCR: {teks_lengkap} → MATCH", "MATCH")
                            else:
                                self.memori_kardus[id_kardus]["gagal_hitung"] += 1
                                if self.memori_kardus[id_kardus]["gagal_hitung"] > 4:
                                    self.memori_kardus[id_kardus]["status"] = "Tidak Sesuai"
                                    if not self.memori_kardus[id_kardus]["counted"]:
                                        self.total_nyasar += 1
                                        self.memori_kardus[id_kardus]["counted"] = True
                                        self.log_updated.emit(f"Kardus #{id_kardus} → OCR: {teks_lengkap} → NG", "NG")

                # 3. Visualisasi Bounding Box (Warna sesuai request BGR: OpenCV)
                warna_kotak = (6, 119, 217) # BGR untuk SCAN_YELLOW (#D97706)
                status = self.memori_kardus[id_kardus]["status"]
                teks_ocr = self.memori_kardus[id_kardus]["teks"]
                confidence = self.memori_kardus[id_kardus]["confidence"]

                if status == "Sesuai":
                    warna_kotak = (74, 163, 22) # BGR untuk MATCH_GREEN (#16A34A)
                    teks_label = f"ID:{id_kardus} | {self.target_kode} OK ({confidence}%)"
                elif status == "Tidak Sesuai":
                    warna_kotak = (38, 38, 220) # BGR untuk NG_RED (#DC2626)
                    teks_pendek = teks_ocr[:15] if len(teks_ocr) > 15 else teks_ocr
                    teks_label = f"ID:{id_kardus} | SALAH: {teks_pendek}"
                else:
                    teks_label = f"ID:{id_kardus} | Scanning..."

                cv2.rectangle(frame, (x1, y1), (x2, y2), warna_kotak, 3)
                cv2.putText(frame, teks_label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, warna_kotak, 2)

        # Hapus kardus dari memori jika sudah hilang dari frame (agar memori tidak bocor)
        id_dalam_memori = list(self.memori_kardus.keys())
        for id_kardus in id_dalam_memori:
            if id_kardus not in id_aktif_di_frame:
                del self.memori_kardus[id_kardus]

        self.total_terdeteksi = len(id_aktif_di_frame)
        return frame

    def stop(self):
        self._running = False
        self.wait()
        # Matikan process OCR saat thread ini dihentikan
        self.ocr_manager.shutdown()

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
        self.video_thread = VideoCaptureThread(camera_index=1, loop_interval_ms=10)
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

    def on_frame_ready(self, frame: np.ndarray):
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
            threading.Thread(target=self._trigger_iot_alarm, args=("/hijau",), daemon=True).start()
        elif msg_type == "NG":
            self.status_label.setStyleSheet(f"color: {NG_RED}; font-weight: bold;")
            threading.Thread(target=self._trigger_iot_alarm, args=("/merah",), daemon=True).start()
        else:
            self.status_label.setStyleSheet(f"color: {TEXT_MUTED}; font-weight: normal;")
            
        self.status_label.setText(message)

    def _trigger_iot_alarm(self, endpoint):
        """Menembak sinyal IoT HTTP GET ke NodeMCU tanpa membuat antarmuka (UI) freeze/lag."""
        url = f"http://{IOT_ESP_IP}{endpoint}"
        try:
            # Timeout 1 detik sudah cukup untuk kirim perintah ringan di jaringan lokal
            urllib.request.urlopen(url, timeout=1.0)
            print(f"[DEBUG] IoT Relay Sukses dieksekusi: {url}")
        except Exception as e:
            print(f"[DEBUG] IoT Relay Gagal terhubung ke {url} -> {e}")

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