"""Benchmark OCR sintetis; tidak mengukur akurasi kardus atau FPS kamera fisik."""
import os
import sys
import time

os.environ['FLAGS_enable_pir_api'] = '0'
os.environ['FLAGS_use_mkldnn'] = '0'

import cv2
import numpy as np
def main():
    # Urutan DLL Windows: PyTorch sebelum Paddle; Paddle sebelum hook import Qt.
    import torch
    from paddleocr import PaddleOCR
    from aplikasi_inspeksi import preprocessing_gambar, read_ocr_crop

    ocr = PaddleOCR(lang='en', use_angle_cls=False, det_db_thresh=0.3,
                    rec_algorithm='SVTR_LCNet', show_log=False,
                    use_gpu=False, enable_mkldnn=True, cpu_threads=4)
    images = []
    for code in ('K81', 'K81', 'K81', 'K81'):
        img = np.full((160, 400, 3), 255, dtype=np.uint8)
        cv2.putText(img, code, (35, 115), cv2.FONT_HERSHEY_SIMPLEX,
                    3, (0, 0, 0), 6, cv2.LINE_AA)
        images.append(img)
    ocr.ocr(images[0], cls=False)
    started = time.perf_counter()
    for img in images:
        ocr.ocr(img, cls=False)
        ocr.ocr(preprocessing_gambar(img), cls=False)
    baseline = time.perf_counter() - started
    started = time.perf_counter()
    for img in images:
        first, second, timing = read_ocr_crop(ocr, img, 'K81')
        print('READING', first, 'FALLBACK', second is not None, 'TIMING', timing, flush=True)
    optimized = time.perf_counter() - started
    print(f'FOUR_SYNTHETIC_CROPS baseline={baseline:.3f}s optimized={optimized:.3f}s', flush=True)
    started = time.perf_counter()
    for img in images:
        first, second, timing = read_ocr_crop(ocr, img, 'K81', prefer_processed=True)
        assert second and second[0], second
        print('PROCESSED_FIRST', timing, flush=True)
    print(f'FOUR_PROCESSED_FIRST {time.perf_counter()-started:.3f}s', flush=True)


def worker_smoke():
    from aplikasi_inspeksi import OcrManager

    started = time.perf_counter()
    manager = OcrManager(True)
    try:
        deadline = time.monotonic() + 120
        finished = 0
        scanning_started = None
        image = np.full((160, 400, 3), 255, dtype=np.uint8)
        cv2.putText(image, 'K81', (35, 115), cv2.FONT_HERSHEY_SIMPLEX,
                    3, (0, 0, 0), 6, cv2.LINE_AA)
        while time.monotonic() < deadline:
            message = manager.poll()
            if message:
                if message[0] == 'ready':
                    print(f'WORKER_READY {message[1]} init={time.perf_counter()-started:.3f}s', flush=True)
                    scanning_started = time.perf_counter()
                    assert manager.submit(1, image, 'K81')
                elif message[0] == 'error':
                    raise RuntimeError(message[1])
                elif message[0] == 'result':
                    assert message[-1] is None, message
                    assert message[2] and message[2][0], message
                    finished += 1
                    if finished == 4:
                        print(f'WORKER_FOUR_CROPS {time.perf_counter()-scanning_started:.3f}s', flush=True)
                        return
                    assert manager.submit(finished + 1, image, 'K81')
            time.sleep(0.01)
        raise RuntimeError('Worker benchmark timeout')
    finally:
        manager.shutdown()


if __name__ == '__main__':
    if '--worker' in sys.argv:
        worker_smoke()
    else:
        main()
