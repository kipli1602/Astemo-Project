import os
os.environ["FLAGS_enable_pir_api"] = "0" 
os.environ["FLAGS_use_mkldnn"] = "0"      

import torch
import cv2
import tkinter as tk
from tkinter import ttk
from ultralytics import YOLO
from paddleocr import PaddleOCR
import numpy as np
import re

def setup_device_ai():
    """Mendeteksi dan mengonfigurasi GPU secara otomatis"""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"\n[INFO] GPU Terdeteksi & Aktif: {gpu_name}")
        return True, 0  
    else:
        print("\n[INFO] Peringatan: GPU tidak tersedia, sistem beralih menggunakan CPU.")
        return False, 'cpu'

def preprocessing_gambar(img):
    """Preprocessing gambar untuk meningkatkan akurasi OCR"""
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
    """Cek apakah target ada di teks dengan toleransi kesalahan"""
    teks_bersih = re.sub(r'[^A-Z0-9]', '', teks.upper())
    target_bersih = target.upper()
    
    if target_bersih in teks_bersih:
        return True, 100
    
    if len(target_bersih) <= 3:  
        for i in range(len(teks_bersih) - len(target_bersih) + 1):
            substring = teks_bersih[i:i+len(target_bersih)]
            kesalahan = sum(1 for a, b in zip(substring, target_bersih) if a != b)
            if kesalahan <= toleransi:
                confidence = int(((len(target_bersih) - kesalahan) / len(target_bersih)) * 100)
                return True, confidence
    
    return False, 0

def jalankan_inspeksi(target_kode):
    print(f"\nSistem Aktif! Menyeleksi Kardus: {target_kode}")
    
    gunakan_gpu, device_yolo = setup_device_ai()
    model = YOLO("best.pt") 
    
    try:
        ocr = PaddleOCR(
            use_textline_orientation=False, 
            lang='en',
            use_angle_cls=True, 
            det_db_thresh=0.3,   
            rec_algorithm='SVTR_LCNet', 
            show_log=False,
            use_gpu=gunakan_gpu
        )
    except Exception:
        ocr = PaddleOCR(use_textline_orientation=False, lang='en', use_angle_cls=True, show_log=False)
    
    memori_kardus = {} 
    
    index_kamera_usb = 2
    cap = cv2.VideoCapture(index_kamera_usb)
    if not cap.isOpened():
        print(f"[PERINGATAN] USB Webcam pada index {index_kamera_usb} tidak ditemukan. Mencoba beralih ke index 0...")
        cap = cv2.VideoCapture(0)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280) 
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    frame_counter = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: 
            print("[ERROR] Gagal membaca frame dari kamera. Pastikan kabel USB webcam terhubung dengan benar.")
            break

        frame_counter += 1
        results = model.track(frame, persist=True, conf=0.6, tracker="botsort.yaml", device=device_yolo, verbose=False)
        
        id_aktif_di_frame = set()

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            ids = results[0].boxes.id.cpu().numpy()
            
            for box, id_kardus in zip(boxes, ids):
                x1, y1, x2, y2 = map(int, box)
                id_kardus = int(id_kardus)
                id_aktif_di_frame.add(id_kardus)
                
                if id_kardus not in memori_kardus:
                    memori_kardus[id_kardus] = {"status": "Scanning...", "teks": "", "confidence": 0, "gagal_hitung": 0}
                
                if memori_kardus[id_kardus]["status"] == "Scanning..." and (frame_counter % 5 == 0):
                    potongan_stiker = frame[y1:y2, x1:x2]
                    if potongan_stiker.size > 0:
                        img_processed = preprocessing_gambar(potongan_stiker)
                        
                        hasil_ocr1 = ocr.ocr(potongan_stiker, cls=False)
                        hasil_ocr2 = ocr.ocr(img_processed, cls=False)
                        
                        semua_teks = []
                        if hasil_ocr1 and hasil_ocr1[0] is not None:
                            semua_teks.extend([baris[1][0] for baris in hasil_ocr1[0]])
                        if hasil_ocr2 and hasil_ocr2[0] is not None:
                            semua_teks.extend([baris[1][0] for baris in hasil_ocr2[0]])
                        
                        teks_lengkap = " ".join(semua_teks)
                        
                        if teks_lengkap.strip():
                            cocok, confidence = fuzzy_match(teks_lengkap, target_kode, toleransi=1)
                            memori_kardus[id_kardus]["teks"] = teks_lengkap
                            memori_kardus[id_kardus]["confidence"] = confidence
                            
                            if cocok:
                                memori_kardus[id_kardus]["status"] = "Sesuai"
                            else:
                                memori_kardus[id_kardus]["gagal_hitung"] += 1
                                if memori_kardus[id_kardus]["gagal_hitung"] > 4:
                                    memori_kardus[id_kardus]["status"] = "Tidak Sesuai"

                warna_kotak = (0, 255, 255) 
                status = memori_kardus[id_kardus]["status"]
                teks_ocr = memori_kardus[id_kardus]["teks"]
                confidence = memori_kardus[id_kardus]["confidence"]

                if status == "Sesuai":
                    warna_kotak = (0, 255, 0) 
                    teks_label = f"ID:{id_kardus} | {target_kode} OK ({confidence}%)"
                elif status == "Tidak Sesuai":
                    warna_kotak = (0, 0, 255) 
                    teks_pendek = teks_ocr[:15] if len(teks_ocr) > 15 else teks_ocr
                    teks_label = f"ID:{id_kardus} | SALAH: {teks_pendek}"
                else:
                    teks_label = f"ID:{id_kardus} | Scanning..."

                cv2.rectangle(frame, (x1, y1), (x2, y2), warna_kotak, 3)
                cv2.putText(frame, teks_label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, warna_kotak, 2)

        id_dalam_memori = list(memori_kardus.keys())
        for id_kardus in id_dalam_memori:
            if id_kardus not in id_aktif_di_frame:
                del memori_kardus[id_kardus]

        total_ok = sum(1 for k, v in memori_kardus.items() if v["status"] == "Sesuai")
        total_salah = sum(1 for k, v in memori_kardus.items() if v["status"] == "Tidak Sesuai")
        
        cv2.rectangle(frame, (10, 10), (350, 100), (0, 0, 0), -1)
        cv2.putText(frame, f"TARGET HARI INI: {target_kode}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"Total Sesuai : {total_ok} Kardus", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Total Nyasar : {total_salah} Kardus", (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Dashboard Inspeksi Honda Real-Time", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def mulai_program():
    target_dipilih = combo_target.get()
    if target_dipilih:
        root.destroy() 
        jalankan_inspeksi(target_dipilih) 

root = tk.Tk()
root.title("Menu Operator Inspeksi")
root.geometry("300x200")
root.eval('tk::PlaceWindow . center')

tk.Label(root, text="Pilih Target Hari Ini:", font=("Arial", 12)).pack(pady=20)

daftar_kode = ["K81", "K80", "K59", "K93"]
combo_target = ttk.Combobox(root, values=daftar_kode, font=("Arial", 14), state="readonly")
combo_target.current(0) 
combo_target.pack(pady=10)

tk.Button(root, text="Mulai Kamera Inspeksi", command=mulai_program, bg="green", fg="white", font=("Arial", 12)).pack(pady=20)

root.mainloop()