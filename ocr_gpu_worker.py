"""Worker PaddleOCR GPU terisolasi dari proses PyTorch/YOLO."""
import os
import re
import sys
import time
from multiprocessing.connection import Client

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ["FLAGS_enable_pir_api"] = "0"
os.environ["FLAGS_use_mkldnn"] = "0"

import cv2
import numpy as np


def add_nvidia_dll_dirs():
    handles = []
    base = os.path.join(sys.prefix, "Lib", "site-packages")
    for pattern in (
        "nvidia/cublas/bin",
        "nvidia/cuda_runtime/bin",
        "nvidia/cuda_nvrtc/bin",
        "nvidia/cudnn/bin",
        "nvidia/cufft/bin",
        "nvidia/curand/bin",
        "nvidia/cusolver/bin",
        "nvidia/cusparse/bin",
        "nvidia/nvjitlink/bin",
    ):
        directory = os.path.join(base, *pattern.split("/"))
        if os.path.isdir(directory):
            handles.append(os.add_dll_directory(directory))
            os.environ["PATH"] = directory + os.pathsep + os.environ["PATH"]
    return handles


def preprocessing_gambar(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    gray = cv2.resize(
        gray, (width * 2, height * 2), interpolation=cv2.INTER_CUBIC)
    gray = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2)
    kernel = np.ones((2, 2), np.uint8)
    return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)


def ocr_text(result):
    if not result or result[0] is None:
        return ""
    return " ".join(line[1][0] for line in result[0])


def fuzzy_match(text, target, tolerance=1):
    clean_text = re.sub(r"[^A-Z0-9]", "", text.upper())
    clean_target = target.upper()
    if clean_target in clean_text:
        return True
    if len(clean_target) <= 4:
        return False
    for index in range(len(clean_text) - len(clean_target) + 1):
        candidate = clean_text[index:index + len(clean_target)]
        if sum(a != b for a, b in zip(candidate, clean_target)) <= tolerance:
            return True
    return False


def matches_target(result, target):
    return bool(target and fuzzy_match(ocr_text(result), target))


def merge_ocr_results(results):
    lines = []
    for result in results:
        if result and result[0] is not None:
            lines.extend(result[0])
    return [lines] if lines else [None]


def ocr_with_orientation(ocr, img, target):
    """Baca crop portrait dari dua arah; crop landscape tidak diubah."""
    shape = getattr(img, "shape", None)
    if shape is None or len(shape) < 2:
        return ocr.ocr(img, cls=False)
    height, width = img.shape[:2]
    if height <= width * 1.15:
        return ocr.ocr(img, cls=False)

    readings = []
    for rotation in (cv2.ROTATE_90_CLOCKWISE,
                     cv2.ROTATE_90_COUNTERCLOCKWISE):
        reading = ocr.ocr(cv2.rotate(img, rotation), cls=False)
        readings.append(reading)
        if matches_target(reading, target):
            return reading
    return merge_ocr_results(readings)


def read_crop(ocr, img, target, prefer_processed, single_pass):
    started = time.perf_counter()
    first = None
    second = None
    timings = dict(raw=0.0, preprocessing=0.0, fallback=0.0, total=0.0)
    order = ("processed", "raw") if prefer_processed else ("raw", "processed")
    for variant in order:
        stage_started = time.perf_counter()
        if variant == "raw":
            first = ocr_with_orientation(ocr, img, target)
            timings["raw"] = time.perf_counter() - stage_started
            result = first
        else:
            processed = preprocessing_gambar(img)
            prep_done = time.perf_counter()
            second = ocr_with_orientation(ocr, processed, target)
            timings["preprocessing"] = prep_done - stage_started
            timings["fallback"] = time.perf_counter() - prep_done
            result = second
        if matches_target(result, target) or single_pass:
            break
    timings["total"] = time.perf_counter() - started
    return first, second, timings


def main():
    host, port, authkey = sys.argv[1], int(sys.argv[2]), bytes.fromhex(sys.argv[3])
    connection = Client((host, port), authkey=authkey)
    dll_handles = add_nvidia_dll_dirs()
    try:
        import paddle
        from paddleocr import PaddleOCR

        if not paddle.is_compiled_with_cuda():
            raise RuntimeError("Paddle terpasang tanpa dukungan CUDA")
        ocr = PaddleOCR(
            lang="en", use_angle_cls=False, det_db_thresh=0.3,
            rec_algorithm="SVTR_LCNet", show_log=False, use_gpu=True,
            enable_mkldnn=False, cpu_threads=2)

        warmup = np.full((96, 256, 3), 255, dtype=np.uint8)
        cv2.putText(warmup, "K81", (20, 68), cv2.FONT_HERSHEY_SIMPLEX,
                    1.8, (0, 0, 0), 4, cv2.LINE_AA)
        ocr.ocr(warmup, cls=False)
        small = np.full((45, 90, 3), 255, dtype=np.uint8)
        cv2.putText(small, "K81", (2, 35), cv2.FONT_HERSHEY_SIMPLEX,
                    1.15, (0, 0, 0), 2, cv2.LINE_AA)
        ocr.ocr(preprocessing_gambar(small), cls=False)
        connection.send(("ready", "GPU"))

        preferred_variant = None
        previous_target = None
        while True:
            item = connection.recv()
            if item is None:
                break
            job_id, img, target, requested_processed = item
            try:
                if target != previous_target:
                    preferred_variant = None
                    previous_target = target
                small_crop = min(img.shape[:2]) < 96
                if preferred_variant is None:
                    use_processed = (small_crop if requested_processed is None
                                     else requested_processed)
                else:
                    use_processed = preferred_variant == "processed"
                first, second, timings = read_crop(
                    ocr, img, target, use_processed,
                    single_pass=requested_processed is not None)
                if matches_target(second, target):
                    preferred_variant = "processed"
                elif matches_target(first, target):
                    preferred_variant = "raw"
                print(f"[OCR GPU #{job_id}] " + " ".join(
                    f"{name}={seconds:.3f}s"
                    for name, seconds in timings.items()), flush=True)
                connection.send(("result", job_id, first, second, None))
            except Exception as exc:
                connection.send(("result", job_id, None, None, str(exc)))
    except Exception as exc:
        connection.send(("error", f"Inisialisasi OCR GPU gagal: {exc}"))
    finally:
        connection.close()
        # Menjaga handle DLL tetap hidup sampai seluruh predictor ditutup.
        del dll_handles


if __name__ == "__main__":
    main()
