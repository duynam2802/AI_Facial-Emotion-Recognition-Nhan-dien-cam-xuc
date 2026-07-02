"""
train.py
Huấn luyện CNN model cho Facial Emotion Recognition
Chạy: python src/train.py
"""

import os
import sys
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Tắt TF warnings
sys.path.insert(0, os.path.dirname(__file__))

from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from model import build_cnn, compile_model
from preprocess import load_fer2013, save_processed, load_processed, EMOTIONS

# ── Cấu hình ──────────────────────────────────────────────
EPOCHS     = 60
BATCH_SIZE = 64
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR   = os.path.join(PROJECT_ROOT, 'data')
MODEL_DIR  = os.path.join(PROJECT_ROOT, 'models')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'outputs')
os.makedirs(MODEL_DIR,  exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 1. Load data ──────────────────────────────────────────
if all(os.path.exists(os.path.join(DATA_DIR, f'{f}.npy')) for f in ['X_train','y_train','X_test','y_test']):
    print("Load numpy arrays đã lưu...")
    X_train, y_train, X_test, y_test = load_processed(DATA_DIR)
else:
    X_train, y_train, X_test, y_test = load_fer2013(DATA_DIR)
    save_processed(X_train, y_train, X_test, y_test, DATA_DIR)

print(f"\nTrain: {len(X_train):,} | Test: {len(X_test):,}")

# ── 2. Data Augmentation ─────────────────────────────────
# Tăng cường dữ liệu để tránh overfitting
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.1,
)
datagen.fit(X_train)

# ── 3. Build & compile model ─────────────────────────────
print("\nXây dựng mô hình CNN...")
model = build_cnn(input_shape=(48, 48, 1), num_classes=len(EMOTIONS))
model = compile_model(model, lr=1e-3)
model.summary()

# ── 4. Callbacks ─────────────────────────────────────────
callbacks = [
    # Lưu model tốt nhất (theo val_accuracy)
    keras.callbacks.ModelCheckpoint(
        filepath=f'{MODEL_DIR}/emotion_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
    # Dừng sớm nếu không cải thiện sau 10 epoch
    keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=10,
        restore_best_weights=True,
        verbose=1
    ),
    # Giảm learning rate khi plateau
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    ),
]

# ── 5. Train ──────────────────────────────────────────────
print(f"\nBắt đầu train {EPOCHS} epochs (batch_size={BATCH_SIZE})...")
print("⏱️  CPU: ~20-30 phút | GPU: ~5 phút\n")

history = model.fit(
    datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
    steps_per_epoch=len(X_train) // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(X_test, y_test),
    callbacks=callbacks,
    verbose=1
)

# ── 6. Đánh giá cuối ─────────────────────────────────────
print("\nĐánh giá trên tập test:")
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"  Test Loss:     {loss:.4f}")
print(f"  Test Accuracy: {acc:.4f} ({acc*100:.2f}%)")

# ── 7. Lưu model final ───────────────────────────────────
model.save(f'{MODEL_DIR}/emotion_model.h5')
print(f"\nĐã lưu model vào {MODEL_DIR}/emotion_model.h5")

# ── 8. Vẽ biểu đồ training history ──────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Accuracy
axes[0].plot(history.history['accuracy'],     label='Train Accuracy', color='steelblue')
axes[0].plot(history.history['val_accuracy'], label='Val Accuracy',   color='coral')
axes[0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Loss
axes[1].plot(history.history['loss'],     label='Train Loss', color='steelblue')
axes[1].plot(history.history['val_loss'], label='Val Loss',   color='coral')
axes[1].set_title('Model Loss', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.suptitle(f'Training History — Final Accuracy: {acc*100:.2f}%', fontsize=15)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/training_history.png', dpi=150)
plt.close()
print(f"Đã lưu {OUTPUT_DIR}/training_history.png")
print("\n✅ Train hoàn tất!")
