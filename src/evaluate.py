"""
evaluate.py
Đánh giá mô hình và vẽ biểu đồ kết quả
Chạy: python src/evaluate.py
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
sys.path.insert(0, os.path.dirname(__file__))

from tensorflow import keras
from preprocess import load_processed, EMOTIONS

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'emotion_model.h5')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Load model & data ────────────────────────────────────
print("Đang load mô hình...")
model = keras.models.load_model(MODEL_PATH)

print("Đang load test data...")
_, _, X_test, y_test = load_processed(DATA_DIR)

# ── Predict ───────────────────────────────────────────────
y_pred_prob = model.predict(X_test, verbose=0)
y_pred      = np.argmax(y_pred_prob, axis=1)

# ── 1. Classification Report ─────────────────────────────
print("\nClassification Report:")
report = classification_report(y_test, y_pred, target_names=EMOTIONS)
print(report)
with open(f'{OUTPUT_DIR}/classification_report.txt', 'w') as f:
    f.write(report)

# ── 2. Confusion Matrix ───────────────────────────────────
cm = confusion_matrix(y_test, y_pred)
cm_pct = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(
    cm_pct, annot=True, fmt='.1f', cmap='Blues',
    xticklabels=EMOTIONS, yticklabels=EMOTIONS,
    linewidths=0.5, ax=ax
)
ax.set_title('Confusion Matrix (%)\nFacial Emotion Recognition', fontsize=14, fontweight='bold')
ax.set_xlabel('Predicted', fontsize=12)
ax.set_ylabel('Actual', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/confusion_matrix.png', dpi=150)
plt.close()
print(f"Đã lưu {OUTPUT_DIR}/confusion_matrix.png")

# ── 3. Per-class Accuracy Bar Chart ──────────────────────
per_class_acc = cm.diagonal() / cm.sum(axis=1) * 100
colors = ['#e74c3c','#8e44ad','#2980b9','#27ae60','#95a5a6','#f39c12','#1abc9c']
emoji  = ['😡','🤢','😨','😊','😐','😢','😲']

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(
    [f'{e} {em}' for e, em in zip(EMOTIONS, emoji)],
    per_class_acc, color=colors, edgecolor='white', linewidth=1.5
)
for bar, val in zip(bars, per_class_acc):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f'{val:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

ax.set_title('Accuracy theo từng cảm xúc', fontsize=14, fontweight='bold')
ax.set_ylabel('Accuracy (%)')
ax.set_ylim(0, 110)
ax.axhline(y=np.mean(per_class_acc), color='red', linestyle='--', alpha=0.7,
           label=f'Trung bình: {np.mean(per_class_acc):.1f}%')
ax.legend()
ax.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/per_class_accuracy.png', dpi=150)
plt.close()
print(f"Đã lưu {OUTPUT_DIR}/per_class_accuracy.png")

# ── 4. Sample Predictions ─────────────────────────────────
fig, axes = plt.subplots(3, 6, figsize=(16, 8))
axes = axes.flatten()
for i in range(18):
    idx    = np.random.randint(len(X_test))
    img    = X_test[idx].squeeze()
    true_l = EMOTIONS[y_test[idx]]
    pred_l = EMOTIONS[y_pred[idx]]
    conf   = y_pred_prob[idx][y_pred[idx]] * 100
    correct = (y_test[idx] == y_pred[idx])

    axes[i].imshow(img, cmap='gray')
    axes[i].set_title(
        f'True: {true_l}\nPred: {pred_l} ({conf:.0f}%)',
        fontsize=8,
        color='green' if correct else 'red'
    )
    axes[i].axis('off')
    for spine in axes[i].spines.values():
        spine.set_edgecolor('green' if correct else 'red')
        spine.set_linewidth(2)

plt.suptitle('Sample Predictions (xanh=đúng, đỏ=sai)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/sample_predictions.png', dpi=150)
plt.close()
print(f"Đã lưu {OUTPUT_DIR}/sample_predictions.png")
print("\n✅ Đánh giá hoàn tất! Xem kết quả trong outputs/")
