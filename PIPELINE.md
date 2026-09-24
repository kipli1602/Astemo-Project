# Alur kamera dan inspeksi

Jalankan dengan environment proyek:

```powershell
.\.venv\Scripts\python.exe aplikasi_inspeksi.py
```

- Thread kamera membaca frame dan memperbarui preview tanpa menunggu AI.
- Worker YOLO mengambil frame terbaru dari antrean berkapasitas satu frame.
- Pool OCR memakai empat proses (masing-masing dua thread CPU) dan menerima
  maksimal empat tugas aktif, sehingga empat kardus dapat dibaca bersamaan. Hasil yang
  selesai tidak berurutan tetap dipasangkan ke kardus asal melalui ID tugas.
  Jika satu bacaan sudah
  memenuhi aturan pencocokan akhir (`fuzzy_match`), pembacaan kedua dilewati.
  Aturan akhir yang sudah ada tidak menggunakan confidence OCR sebagai batas OK;
  jalur cepat sekarang mengikuti aturan yang sama. Jika hasil preprocessing lebih
  berhasil, worker mencoba preprocessing dahulu pada tugas berikutnya. Saat belum
  cocok, kedua varian tetap dibaca. Preferensi direset ketika kode target berubah.
  Crop dengan sisi terpendek di bawah 96 piksel langsung mencoba hasil pembesaran
  terlebih dahulu karena jalur raw pada ukuran tersebut biasanya tidak terbaca.
  Crop portrait dicoba dengan rotasi 90 derajat ke dua arah. Percobaan arah kedua
  dilewati jika arah pertama sudah menemukan target, sehingga label vertikal tetap
  terbaca tanpa menambah latensi pada label horizontal.
  Detector dan recognizer OCR juga dipanaskan sebelum status `OCR siap` dikirim,
  sehingga cold-start tidak dibebankan pada kardus pertama.
  Setiap giliran hanya menjalankan satu varian OCR. Jika belum cocok, varian raw
  dan preprocessing dicoba bergantian pada frame berikutnya. Kedua jalur tetap
  tersedia, tetapi satu frame tidak lagi menunggu dua inferensi berturut-turut.
  Aturan pencocokan akhir serta lima percobaan tidak cocok untuk NG tetap sama.
  Kardus mendapat giliran berdasarkan waktu pengajuan terakhir, tanpa jeda lima frame.
- Bounding box memakai hasil AI terbaru, sehingga dapat sedikit tertinggal dari
  gerakan kamera. Overlay yang lebih tua dari 0,5 detik disembunyikan.
- Hasil OCR dari target sebelumnya atau objek yang sudah dihapus diabaikan.
- Batas keputusan OCR adalah 3 detik sejak tugas pertama untuk kardus tersebut.
  Hasil MATCH tetap diprioritaskan; kardus menjadi NG setelah lima mismatch atau
  ketika batas waktu tercapai. Timeout keselamatan job 30 detik tetap aktif.
  Terminal mencetak `raw_text`, `processed_text`, dan gabungan hasil agar setiap
  percobaan dapat diperiksa. Error OCR dilaporkan melalui status aplikasi.

YOLO memilih CUDA bila tersedia. OCR memeriksa dukungan CUDA Paddle secara terpisah
dan menampilkan perangkat aktual melalui status `OCR siap: CPU/GPU`. Environment
yang diperiksa memakai PaddleOCR 2.7.3 dan PaddlePaddle GPU 2.6.2. API OCR yang
digunakan tetap API 2.x.
Saat worker Windows dimuat, backend OCR diimpor sebelum Qt untuk menghindari beban
hook import Qt. Waktu OCR asli, preprocessing, dan OCR kedua dicetak di terminal
sebagai `[OCR #...]`; ini membantu membedakan pembacaan lambat dari startup model.
Log juga mencatat teks tiap varian, ID kardus, revisi target, dan jumlah ketidakcocokan
sebelum hasil diterapkan. `fallback` adalah waktu OCR gambar preprocessing, termasuk
ketika gambar preprocessing dicoba dahulu.

Inferensi YOLO memakai ukuran 512. Pada 134 gambar kalibrasi, konfigurasi ini
mempertahankan seluruh 268 deteksi dari ukuran 640 dengan IoU rata-rata 0,964,
sekaligus menurunkan waktu inferensi. Buffer kamera dibatasi satu frame agar frame
lama dari driver tidak menambah latensi.

Pada Windows, Paddle GPU dijalankan oleh `ocr_gpu_worker.py` sebagai proses mandiri.
Pemisahan ini mencegah bentrok DLL/registrasi CUDA antara PyTorch YOLO dan Paddle.
Empat crop dapat mengantre pada frame yang sama meskipun satu predictor GPU yang
mengeksekusinya; timer keputusan semua box jadi dimulai bersamaan, bukan berantai.
Jika distribusi `paddlepaddle-gpu` tidak tersedia, aplikasi otomatis kembali ke
pool empat worker OCR CPU.

Tes tanpa membuka kamera fisik:

```powershell
.\.venv\Scripts\python.exe -m unittest test_pipeline -v
```

Benchmark empat crop kode sintetis (bukan pengukuran kardus asli):

```powershell
.\.venv\Scripts\python.exe benchmark_scanning.py
.\.venv\Scripts\python.exe benchmark_scanning.py --worker
.\.venv\Scripts\python.exe benchmark_scanning.py --worker --small
.\.venv\Scripts\python.exe benchmark_scanning.py --pool=2
.\.venv\Scripts\python.exe benchmark_scanning.py --pool=4
```

Untuk validasi lapangan, gerakkan kardus saat status scanning dan periksa kelancaran
preview, kecocokan ID/hasil, perubahan target, pergantian kamera, serta penutupan
aplikasi saat OCR masih berjalan. FPS kamera fisik belum diukur oleh tes simulasi.
