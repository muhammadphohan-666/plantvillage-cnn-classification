# Image Classification — PlantVillage (Penyakit Tanaman)

Proyek klasifikasi gambar menggunakan CNN untuk mendeteksi penyakit tanaman dari dataset **PlantVillage** (color).

## Dataset

- **Sumber:** PlantVillage dataset (Kaggle: abdallahalidev/plantvillage-dataset)
- **Total gambar:** 54.305
- **Jumlah kelas:** 38 (berbagai jenis tanaman dan kondisi, mis. sehat, Apple_scab, Bacterial_spot, dll.)
- **Pembagian data:**
  - Train: 70% (38.013 gambar)
  - Validation: 20% (10.861 gambar)
  - Test: 10% (5.431 gambar)
  - Stratified split

## Model

- **Arsitektur:** Keras Sequential
- **Layers:** Input → Conv2D (32, 64, 128, 256) + BatchNormalization + MaxPooling2D → GlobalAveragePooling2D → Dense(256) → Dropout → Dense(38, softmax)
- **Input:** citra RGB 256×256, nilai piksel dinormalisasi ke [0, 1]
- **Augmentasi (hanya saat training):** RandomFlip, RandomRotation, RandomZoom, RandomContrast
- **Callback:** ModelCheckpoint (best val_accuracy), EarlyStopping
- **Optimizer:** Adam dengan CosineDecay

## Hasil (Performance)

- **Test accuracy:** 97,79% (5.431 sampel test)
- **F1 Macro:** 0,9729
- **F1 Weighted:** 0,9778
- **Precision (weighted avg):** 0,9788 · **Recall (weighted avg):** 0,9779


## Isi Submission

| Folder / File        | Keterangan                                                            |
| -------------------- | --------------------------------------------------------------------- |
| **saved_model/**     | Format TensorFlow (SavedModel) untuk deployment server/cloud          |
| **tflite/**          | `model.tflite` + `label.txt` (38 kelas) untuk mobile/embedded         |
| **tfjs_model/**      | `model.json` + shard weights untuk TensorFlow.js (browser/JavaScript) |
| **notebook.ipynb**   | Notebook lengkap beserta output (training, evaluasi, export)          |
| **requirements.txt** | Dependensi Python untuk menjalankan ulang                             |

## Cara Menggunakan

- **SavedModel:** muat dengan `tf.saved_model.load()` atau Keras `load_model()`.
- **TFLite:** gunakan interpreter TFLite; urutan kelas mengikuti baris di `tflite/label.txt`.
- **TF.js:** muat `model.json` dengan TensorFlow.js; input shape (1, 256, 256, 3), nilai [0, 1].
