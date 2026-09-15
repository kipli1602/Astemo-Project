# Alur kamera dan inspeksi

Jalankan dengan environment proyek:

```powershell
.\.venv\Scripts\python.exe aplikasi_inspeksi.py
```

- Thread kamera membaca frame dan memperbarui preview tanpa menunggu AI.
- Worker YOLO mengambil frame terbaru dari antrean berkapasitas satu frame.
- Worker proses OCR menerima maksimal satu tugas aktif. Jika satu bacaan sudah
  memenuhi aturan pencocokan akhir (`fuzzy_match`), pembacaan kedua dilewati.
  Aturan akhir yang sudah ada tidak menggunakan confidence OCR sebagai batas OK;
  jalur cepat sekarang mengikuti aturan yang sama. Jika hasil preprocessing lebih
  berhasil, worker mencoba preprocessing dahulu pada tugas berikutnya. Saat belum
  cocok, kedua varian tetap dibaca. Preferensi direset ketika kode target berubah.
  Aturan pencocokan akhir serta lima percobaan tidak cocok untuk NG tetap sama.
  Kardus mendapat giliran berdasarkan waktu pengajuan terakhir, tanpa jeda lima frame.
- Bounding box memakai hasil AI terbaru, sehingga dapat sedikit tertinggal dari
  gerakan kamera. Overlay yang lebih tua dari 0,5 detik disembunyikan.
- Hasil OCR dari target sebelumnya atau objek yang sudah dihapus diabaikan.
- Error OCR dilaporkan melalui status aplikasi. Tugas OCR yang melampaui 30 detik
  menghentikan pengajuan OCR baru; tutup dan buka kembali aplikasi untuk mencoba lagi.

YOLO memilih CUDA bila tersedia. OCR memeriksa dukungan CUDA Paddle secara terpisah
dan menampilkan perangkat aktual melalui status `OCR siap: CPU/GPU`. Environment
yang diperiksa memakai PaddleOCR 2.7.3 dan PaddlePaddle 2.6.2 CPU. Perubahan pipeline
ini tidak mengganti paket atau memasang Paddle GPU; API OCR yang digunakan adalah 2.x.
Saat worker Windows dimuat, backend OCR diimpor sebelum Qt untuk menghindari beban
hook import Qt. Waktu OCR asli, preprocessing, dan OCR kedua dicetak di terminal
sebagai `[OCR #...]`; ini membantu membedakan pembacaan lambat dari startup model.
Log juga mencatat teks tiap varian, ID kardus, revisi target, dan jumlah ketidakcocokan
sebelum hasil diterapkan. `fallback` adalah waktu OCR gambar preprocessing, termasuk
ketika gambar preprocessing dicoba dahulu.

Tes tanpa membuka kamera fisik:

```powershell
.\.venv\Scripts\python.exe -m unittest test_pipeline -v
```

Benchmark empat crop kode sintetis (bukan pengukuran kardus asli):

```powershell
.\.venv\Scripts\python.exe benchmark_scanning.py
.\.venv\Scripts\python.exe benchmark_scanning.py --worker
```

Untuk validasi lapangan, gerakkan kardus saat status scanning dan periksa kelancaran
preview, kecocokan ID/hasil, perubahan target, pergantian kamera, serta penutupan
aplikasi saat OCR masih berjalan. FPS kamera fisik belum diukur oleh tes simulasi.
