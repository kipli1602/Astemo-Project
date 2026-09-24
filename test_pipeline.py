import queue
import threading
import time
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PySide6.QtCore import Qt

from aplikasi_inspeksi import (
    OcrManager, VideoCaptureThread, _ocr_with_orientation, read_ocr_crop)


class PipelineTests(unittest.TestCase):
    def test_portrait_crop_tries_opposite_rotation(self):
        ocr = Mock()
        wrong = [[[None, ('69g', 0.80)]]]
        correct = [[[None, ('K59', 0.99)]]]
        ocr.ocr.side_effect = [wrong, correct]
        result = _ocr_with_orientation(
            ocr, np.zeros((120, 50, 3), dtype=np.uint8), 'K59')
        self.assertEqual(result, correct)
        self.assertEqual(ocr.ocr.call_count, 2)

    def test_portrait_crop_stops_after_first_rotation_matches(self):
        ocr = Mock()
        correct = [[[None, ('52400-K59-A11', 0.99)]]]
        ocr.ocr.return_value = correct
        result = _ocr_with_orientation(
            ocr, np.zeros((120, 50, 3), dtype=np.uint8), 'K59')
        self.assertEqual(result, correct)
        self.assertEqual(ocr.ocr.call_count, 1)

    def test_fast_path_uses_existing_final_match_rule(self):
        # Keputusan lama sudah menerima hasil ini; OCR kedua tidak mengubah OK.
        for text, score in [('K81', 0.80), ('12345-K81-900', 0.94)]:
            with self.subTest(text=text):
                ocr = Mock()
                ocr.ocr.return_value = [[[None, (text, score)]]]
                _, second, _ = read_ocr_crop(ocr, np.zeros((8, 8, 3)), 'K81')
                self.assertIsNone(second)
                self.assertEqual(ocr.ocr.call_count, 1)

    def test_processed_first_match_skips_raw(self):
        ocr = Mock()
        ocr.ocr.return_value = [[[None, ('K81', 0.90)]]]
        with patch('aplikasi_inspeksi.preprocessing_gambar'):
            first, second, timings = read_ocr_crop(ocr, np.zeros((8, 8, 3)), 'K81', True)
        self.assertIsNone(first)
        self.assertIsNotNone(second)
        self.assertEqual(timings['raw'], 0)
        self.assertEqual(ocr.ocr.call_count, 1)

    def test_processed_first_miss_still_reads_raw(self):
        ocr = Mock()
        processed_read = [[[None, ('K80', 0.99)]]]
        raw_read = [[[None, ('K81', 0.99)]]]
        ocr.ocr.side_effect = [processed_read, raw_read]
        with patch('aplikasi_inspeksi.preprocessing_gambar'):
            first, second, _ = read_ocr_crop(ocr, np.zeros((8, 8, 3)), 'K81', True)
        self.assertEqual(first, raw_read)
        self.assertEqual(second, processed_read)

    def test_confident_exact_target_skips_second_pass(self):
        reading = [[[None, ('PART K81', 0.99)]]]
        ocr = Mock()
        ocr.ocr.return_value = reading
        with patch('aplikasi_inspeksi.preprocessing_gambar') as preprocess:
            first, second, _ = read_ocr_crop(ocr, np.zeros((8, 8, 3)), 'K81')
        self.assertEqual(first, reading)
        self.assertIsNone(second)
        self.assertEqual(ocr.ocr.call_count, 1)
        preprocess.assert_not_called()

    def test_uncertain_or_wrong_code_keeps_second_pass(self):
        cases = [('K80', 0.99), ('K8I', 0.99)]
        for code, confidence in cases:
            with self.subTest(code=code, confidence=confidence):
                ocr = Mock()
                ocr.ocr.side_effect = [[[[None, (code, confidence)]]], [None]]
                with patch('aplikasi_inspeksi.preprocessing_gambar') as preprocess:
                    _, second, _ = read_ocr_crop(ocr, np.zeros((8, 8, 3)), 'K81')
                self.assertEqual(second, [None])
                self.assertEqual(ocr.ocr.call_count, 2)
                preprocess.assert_called_once()

    def test_empty_first_read_keeps_second_pass(self):
        ocr = Mock()
        ocr.ocr.side_effect = [[None], [[[None, ('K81', 0.99)]]]]
        with patch('aplikasi_inspeksi.preprocessing_gambar'):
            _, second, _ = read_ocr_crop(ocr, np.zeros((8, 8, 3)), 'K81')
        self.assertIsNotNone(second[0])

    def test_single_pass_defers_fallback_to_next_frame(self):
        ocr = Mock()
        ocr.ocr.return_value = [[[None, ('K80', 0.99)]]]
        with patch('aplikasi_inspeksi.preprocessing_gambar'):
            first, second, _ = read_ocr_crop(
                ocr, np.zeros((8, 8, 3)), 'K81',
                prefer_processed=True, single_pass=True)
        self.assertIsNone(first)
        self.assertIsNotNone(second)
        self.assertEqual(ocr.ocr.call_count, 1)

    def make_worker(self):
        worker = VideoCaptureThread()
        worker._active_target = 'K81'
        worker.ocr_manager = Mock()
        worker.ocr_manager.is_alive.return_value = True
        return worker

    def result(self, worker, revision=0, current=True):
        state = dict(status='Scanning...', teks='', confidence=0, gagal_hitung=0)
        worker.memori_kardus[1] = state if current else state.copy()
        worker._pending_ocr = {
            1: (1, revision, 1, state, time.monotonic())}
        reading = [[[None, ('K81', 0.99)]]]
        worker.ocr_manager.poll.side_effect = [('result', 1, reading, None, None), None]
        worker._poll_ocr()
        return worker.memori_kardus[1]

    def test_matching_result(self):
        self.assertEqual(self.result(self.make_worker())['status'], 'Sesuai')

    def test_old_target_result_discarded(self):
        self.assertEqual(self.result(self.make_worker(), revision=-1)['status'], 'Scanning...')

    def test_removed_box_result_discarded(self):
        self.assertEqual(self.result(self.make_worker(), current=False)['status'], 'Scanning...')

    def test_target_changed_before_next_ai_frame(self):
        worker = self.make_worker()
        worker.set_target('K80')
        self.assertEqual(self.result(worker)['status'], 'Scanning...')

    def test_ocr_decision_timeout_marks_unreadable_as_ng(self):
        worker = self.make_worker()
        state = dict(status='Scanning...', teks='', confidence=0,
                     gagal_hitung=0, scan_started=time.monotonic() - 4)
        worker.memori_kardus[1] = state
        worker._pending_ocr = {
            1: (1, 0, 1, state, time.monotonic())}
        worker.ocr_manager.poll.side_effect = [
            ('result', 1, None, None, None), None]
        worker._poll_ocr()
        self.assertEqual(state['status'], 'Tidak Sesuai')
        self.assertEqual(state['teks'], 'TIDAK TERBACA')

    def test_match_has_priority_over_decision_timeout(self):
        worker = self.make_worker()
        state = dict(status='Scanning...', teks='', confidence=0,
                     gagal_hitung=0, scan_started=time.monotonic() - 4)
        worker.memori_kardus[1] = state
        worker._pending_ocr = {
            1: (1, 0, 1, state, time.monotonic())}
        reading = [[[None, ('K81', 0.99)]]]
        worker.ocr_manager.poll.side_effect = [
            ('result', 1, reading, None, None), None]
        worker._poll_ocr()
        self.assertEqual(state['status'], 'Sesuai')

    def test_ocr_timeout_does_not_block(self):
        worker = self.make_worker()
        worker._pending_ocr = {
            1: (1, 0, 1, {}, time.monotonic() - 31)}
        worker.ocr_manager.poll.return_value = None
        worker._poll_ocr()
        self.assertTrue(worker._ocr_failed)

    def test_mismatch_limit_marks_box_as_ng(self):
        worker = self.make_worker()
        state = dict(status='Scanning...', teks='', confidence=0,
                     gagal_hitung=99, scan_started=time.monotonic() - 60)
        worker.memori_kardus[1] = state
        worker._pending_ocr = {
            1: (1, 0, 1, state, time.monotonic())}
        reading = [[[None, ('K80', 0.99)]]]
        worker.ocr_manager.poll.side_effect = [
            ('result', 1, reading, None, None), None]
        worker._poll_ocr()
        self.assertEqual(state['status'], 'Tidak Sesuai')
        self.assertEqual(state['gagal_hitung'], 100)

    def test_dead_ocr_does_not_block(self):
        worker = self.make_worker()
        worker.ocr_manager.poll.return_value = None
        worker.ocr_manager.is_alive.return_value = False
        worker._poll_ocr()
        self.assertTrue(worker._ocr_failed)

    def test_full_ocr_queue_drops_submission(self):
        manager = OcrManager.__new__(OcrManager)
        manager.image_queue = queue.Queue(maxsize=1)
        crop = np.zeros((8, 8, 3), dtype=np.uint8)
        self.assertTrue(manager.submit(1, crop))
        self.assertFalse(manager.submit(2, crop))

    def test_manager_waits_until_all_workers_are_ready(self):
        manager = OcrManager.__new__(OcrManager)
        manager.worker_count = 2
        manager.ready_workers = 0
        manager.result_queue = queue.Queue()
        manager.result_queue.put(('ready', 'CPU'))
        self.assertIsNone(manager.poll())
        manager.result_queue.put(('ready', 'CPU'))
        self.assertEqual(manager.poll(), ('ready', 'CPU x2'))

    def test_parallel_results_are_matched_by_job_id(self):
        worker = self.make_worker()
        state1 = dict(status='Scanning...', teks='', confidence=0,
                      gagal_hitung=0, scan_started=time.monotonic())
        state2 = dict(status='Scanning...', teks='', confidence=0,
                      gagal_hitung=0, scan_started=time.monotonic())
        worker.memori_kardus = {1: state1, 2: state2}
        now = time.monotonic()
        worker._pending_ocr = {
            1: (1, 0, 1, state1, now),
            2: (2, 0, 2, state2, now),
        }
        reading = [[[None, ('K81', 0.99)]]]
        worker.ocr_manager.poll.side_effect = [
            ('result', 2, reading, None, None),
            ('result', 1, reading, None, None),
            None,
        ]
        worker._poll_ocr()
        self.assertEqual(state1['status'], 'Sesuai')
        self.assertEqual(state2['status'], 'Sesuai')
        self.assertEqual(worker._pending_ocr, {})

    def test_camera_continues_while_ai_waits(self):
        worker = VideoCaptureThread()
        frames = []
        released = threading.Event()

        class Camera:
            count = 0

            def isOpened(self):
                return True

            def set(self, *args):
                pass

            def read(self):
                time.sleep(0.005)
                self.count += 1
                if self.count >= 20:
                    worker._stop_event.set()
                return True, np.full((8, 8, 3), self.count, dtype=np.uint8)

            def release(self):
                released.set()

        def slow_ai():
            worker._stop_event.wait(2)

        def receive(frame):
            frames.append(int(frame[0, 0, 0]))
            worker._preview_pending = False

        worker.frame_ready.connect(receive, Qt.DirectConnection)
        with patch('aplikasi_inspeksi.cv2.VideoCapture', return_value=Camera()), \
                patch.object(worker, '_run_ai', side_effect=slow_ai):
            worker.run()
        self.assertEqual(len(frames), 20)
        self.assertEqual(worker._frames.qsize(), 1)
        self.assertEqual(int(worker._frames.get_nowait()[1][0, 0, 0]), 20)
        self.assertTrue(released.is_set())


if __name__ == '__main__':
    unittest.main()
