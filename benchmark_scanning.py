"""Benchmark OCR sintetis; tidak mengukur akurasi kardus atau FPS kamera fisik."""
import os
import sys
import time

os.environ['FLAGS_enable_pir_api'] = '0'
os.environ['FLAGS_use_mkldnn'] = '0'
os.environ.setdefault('PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION', 'python')

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


def worker_smoke(small_crop=False):
    from aplikasi_inspeksi import OcrManager

    started = time.perf_counter()
    manager = OcrManager(True, worker_count=1)
    try:
        deadline = time.monotonic() + 120
        finished = 0
        scanning_started = None
        if small_crop:
            image = np.full((45, 90, 3), 255, dtype=np.uint8)
            cv2.putText(image, 'K81', (2, 35), cv2.FONT_HERSHEY_SIMPLEX,
                        1.15, (0, 0, 0), 2, cv2.LINE_AA)
        else:
            image = np.full((160, 400, 3), 255, dtype=np.uint8)
            cv2.putText(image, 'K81', (35, 115), cv2.FONT_HERSHEY_SIMPLEX,
                        3, (0, 0, 0), 6, cv2.LINE_AA)
        while time.monotonic() < deadline:
            message = manager.poll()
            if message:
                if message[0] == 'ready':
                    print(f'WORKER_READY {message[1]} init={time.perf_counter()-started:.3f}s', flush=True)
                    scanning_started = time.perf_counter()
                    assert manager.submit(
                        1, image, 'K81', prefer_processed=small_crop)
                elif message[0] == 'error':
                    raise RuntimeError(message[1])
                elif message[0] == 'result':
                    assert message[-1] is None, message
                    assert ((message[2] and message[2][0]) or
                            (message[3] and message[3][0])), message
                    finished += 1
                    if finished == 4:
                        print(f'WORKER_FOUR_CROPS {time.perf_counter()-scanning_started:.3f}s', flush=True)
                        return
                    assert manager.submit(
                        finished + 1, image, 'K81',
                        prefer_processed=small_crop)
            time.sleep(0.01)
        raise RuntimeError('Worker benchmark timeout')
    finally:
        manager.shutdown()


def parallel_worker_smoke(worker_count=2):
    from aplikasi_inspeksi import OcrManager

    managers = [OcrManager(False, worker_count=1)
                for _ in range(worker_count)]
    image = np.full((45, 90, 3), 255, dtype=np.uint8)
    cv2.putText(image, 'K81', (2, 35), cv2.FONT_HERSHEY_SIMPLEX,
                1.15, (0, 0, 0), 2, cv2.LINE_AA)
    try:
        deadline = time.monotonic() + 120
        ready = set()
        while len(ready) < worker_count and time.monotonic() < deadline:
            for index, manager in enumerate(managers):
                message = manager.poll()
                if message and message[0] == 'ready':
                    ready.add(index)
                elif message and message[0] == 'error':
                    raise RuntimeError(message[1])
            time.sleep(0.01)
        assert len(ready) == worker_count, ready

        started = time.perf_counter()
        for index, manager in enumerate(managers):
            assert manager.submit(index + 1, image, 'K81', prefer_processed=True)
        finished = 0
        next_job = worker_count + 1
        while finished < 4 and time.monotonic() < deadline:
            for manager in managers:
                message = manager.poll()
                if not message or message[0] != 'result':
                    continue
                assert message[-1] is None, message
                finished += 1
                if next_job <= 4:
                    assert manager.submit(
                        next_job, image, 'K81', prefer_processed=True)
                    next_job += 1
            time.sleep(0.005)
        assert finished == 4, finished
        print(f'PARALLEL_{worker_count}_FOUR_CROPS '
              f'{time.perf_counter()-started:.3f}s', flush=True)
    finally:
        for manager in managers:
            manager.shutdown()


def pooled_worker_smoke(worker_count=2):
    from aplikasi_inspeksi import OcrManager

    manager = OcrManager(False, worker_count=worker_count)
    image = np.full((45, 90, 3), 255, dtype=np.uint8)
    cv2.putText(image, 'K81', (2, 35), cv2.FONT_HERSHEY_SIMPLEX,
                1.15, (0, 0, 0), 2, cv2.LINE_AA)
    try:
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            message = manager.poll()
            if message and message[0] == 'ready':
                break
            if message and message[0] == 'error':
                raise RuntimeError(message[1])
            time.sleep(0.01)
        else:
            raise RuntimeError('Pool OCR tidak siap')

        try:
            import psutil
            parent = psutil.Process()
            rss = parent.memory_info().rss + sum(
                child.memory_info().rss for child in parent.children(recursive=True))
            print(f'POOL_{worker_count}_RAM {rss / (1024 ** 3):.2f}GB', flush=True)
        except Exception:
            pass

        started = time.perf_counter()
        next_job = 1
        inflight = 0
        finished = 0
        while next_job <= min(worker_count, 4):
            assert manager.submit(
                next_job, image, 'K81', prefer_processed=True)
            next_job += 1
            inflight += 1
        while finished < 4 and time.monotonic() < deadline:
            message = manager.poll()
            if message and message[0] == 'result':
                assert message[-1] is None, message
                finished += 1
                inflight -= 1
                if next_job <= 4:
                    assert manager.submit(
                        next_job, image, 'K81', prefer_processed=True)
                    next_job += 1
                    inflight += 1
            time.sleep(0.005)
        assert finished == 4, (finished, inflight)
        print(f'POOL_{worker_count}_FOUR_CROPS '
              f'{time.perf_counter()-started:.3f}s', flush=True)
    finally:
        manager.shutdown()


if __name__ == '__main__':
    parallel_arg = next(
        (arg for arg in sys.argv if arg.startswith('--parallel=')), None)
    pool_arg = next(
        (arg for arg in sys.argv if arg.startswith('--pool=')), None)
    if pool_arg:
        pooled_worker_smoke(int(pool_arg.split('=', 1)[1]))
    elif parallel_arg:
        parallel_worker_smoke(int(parallel_arg.split('=', 1)[1]))
    elif '--worker' in sys.argv:
        worker_smoke('--small' in sys.argv)
    else:
        main()
