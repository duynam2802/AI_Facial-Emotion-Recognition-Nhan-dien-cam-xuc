"""
preprocess.py - Chuẩn bị dữ liệu từ thư mục ảnh (train/test)
Chạy: python src/preprocess.py
"""

import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Nhãn 7 cảm xúc
EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
IMG_SIZE = 48
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DEFAULT_DATA_DIR = os.path.join(PROJECT_ROOT, 'data')

def load_fer2013(data_dir=DEFAULT_DATA_DIR):
    """
    Load dữ liệu từ thư mục ảnh (train/test) thay vì CSV.
    Yêu cầu cấu trúc: data/train/{class}/ và data/test/{class}/
    """
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')
    
    # Nếu không có thư mục test, thử dùng validation
    if not os.path.exists(test_dir):
        test_dir = os.path.join(data_dir, 'validation')
        if not os.path.exists(test_dir):
            raise FileNotFoundError(f"Không tìm thấy thư mục test hoặc validation trong {data_dir}")

    def load_images_from_folder(folder):
        images = []
        labels = []
        for label_idx, emotion in enumerate(EMOTIONS):
            class_path = os.path.join(folder, emotion)
            if not os.path.exists(class_path):
                print(f"Cảnh báo: Không tìm thấy thư mục {class_path}")
                continue
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                try:
                    # Đọc ảnh grayscale, resize về 48x48
                    img = load_img(img_path, color_mode='grayscale', target_size=(IMG_SIZE, IMG_SIZE))
                    img_array = img_to_array(img) / 255.0  # Normalize [0,1]
                    images.append(img_array)
                    labels.append(label_idx)
                except Exception as e:
                    print(f"Lỗi đọc ảnh {img_path}: {e}")
        return np.array(images), np.array(labels)

    print("Đang load ảnh train...")
    X_train, y_train = load_images_from_folder(train_dir)
    print("Đang load ảnh test...")
    X_test, y_test = load_images_from_folder(test_dir)

    print(f"\nX_train: {X_train.shape} | y_train: {y_train.shape}")
    print(f"X_test:  {X_test.shape}  | y_test:  {y_test.shape}")
    
    return X_train, y_train, X_test, y_test


def save_processed(X_train, y_train, X_test, y_test, out_dir=DEFAULT_DATA_DIR):
    """Lưu numpy arrays để tái sử dụng (tránh parse lại mỗi lần)."""
    os.makedirs(out_dir, exist_ok=True)
    np.save(f'{out_dir}/X_train.npy', X_train)
    np.save(f'{out_dir}/y_train.npy', y_train)
    np.save(f'{out_dir}/X_test.npy',  X_test)
    np.save(f'{out_dir}/y_test.npy',  y_test)
    print(f"\nĐã lưu numpy arrays vào {out_dir}/")


def load_processed(data_dir=DEFAULT_DATA_DIR):
    """Load numpy arrays đã lưu."""
    X_train = np.load(f'{data_dir}/X_train.npy')
    y_train = np.load(f'{data_dir}/y_train.npy')
    X_test  = np.load(f'{data_dir}/X_test.npy')
    y_test  = np.load(f'{data_dir}/y_test.npy')
    return X_train, y_train, X_test, y_test


if __name__ == '__main__':
    # Nếu đã có file .npy thì load luôn, không đọc lại ảnh
    if all(os.path.exists(os.path.join(DEFAULT_DATA_DIR, f'{f}.npy')) for f in ['X_train','y_train','X_test','y_test']):
        print("Đã có numpy arrays, load từ file...")
        X_train, y_train, X_test, y_test = load_processed()
    else:
        X_train, y_train, X_test, y_test = load_fer2013()
        save_processed(X_train, y_train, X_test, y_test)
    
    print("\n✅ Tiền xử lý hoàn tất!")

    # Thống kê class distribution
    print("\nClass distribution (train):")
    for i, emotion in enumerate(EMOTIONS):
        count = np.sum(y_train == i)
        pct   = count / len(y_train) * 100
        print(f"  {emotion:10s}: {count:5,} ({pct:.1f}%)")