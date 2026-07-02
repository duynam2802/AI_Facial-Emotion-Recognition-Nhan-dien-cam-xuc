# Huong Dan Train Tren Google Colab

Tai lieu nay gom 2 phuong an:

1. Bat buoc TensorFlow 2.13.0.
2. TensorFlow 2.19.1 de dung GPU Colab on dinh.

## 1) Chuan bi chung

1. Bat GPU: Runtime -> Change runtime type -> Hardware accelerator -> GPU.
2. Mount Drive:

```python
from google.colab import drive
drive.mount('/content/drive')
```

3. Lay source code (chay 1 trong 3 cach):

```bash
!unzip -q /content/drive/MyDrive/BTL.zip -d /content
%cd /content/BTL
```

```bash
!git clone <GITHUB_REPO_URL> /content/BTL
%cd /content/BTL
```

```bash
!cp -r /content/drive/MyDrive/BTL /content/BTL
%cd /content/BTL
```

4. Kiem tra thu muc du an:

```bash
!ls -la /content/BTL
```

5. Kiem tra va doi ten class neu dang chu thuong (Linux phan biet hoa thuong):

```python
import os

mapping = {
	'angry': 'Angry',
	'disgust': 'Disgust',
	'fear': 'Fear',
	'happy': 'Happy',
	'neutral': 'Neutral',
	'sad': 'Sad',
	'surprise': 'Surprise'
}

for split in ['train', 'test']:
	base = os.path.join('data', split)
	if not os.path.isdir(base):
		print(f'Khong tim thay: {base}')
		continue

	for name in os.listdir(base):
		src = os.path.join(base, name)
		if not os.path.isdir(src):
			continue

		key = name.strip().lower()
		if key in mapping:
			dst = os.path.join(base, mapping[key])
			if src != dst and not os.path.exists(dst):
				os.rename(src, dst)
				print(f'Renamed: {src} -> {dst}')

print('Done')
```

## 2) Phuong an A: Bat buoc TensorFlow 2.13.0

Phuong an nay co the khong dung duoc GPU tren Colab moi do mismatch CUDA.

1. Tao Python 3.10 venv:

```bash
!apt-get update -y
!apt-get install -y python3.10 python3.10-venv python3.10-distutils
!python3.10 -m venv /content/tf213
!/content/tf213/bin/python -m pip install --upgrade pip setuptools wheel
```

2. Cai dependencies TF 2.13:

```bash
!/content/tf213/bin/pip install -r requirements-colab-tf213.txt
```

3. Kiem tra version:

```bash
!/content/tf213/bin/python -c "import sys, tensorflow as tf; print(sys.version); print(tf.__version__); print(tf.config.list_physical_devices('GPU'))"
```

4. Train va evaluate:

```bash
!/content/tf213/bin/python src/preprocess.py
!/content/tf213/bin/python src/train.py
!/content/tf213/bin/python src/evaluate.py
```

## 3) Phuong an B: TensorFlow 2.19.1 de dung GPU Colab

Phuong an nay khuyen dung neu ban uu tien toc do train bang GPU.

1. Cai dependencies:

```bash
!pip install -U pip
!pip install -r requirements-colab-gpu.txt
```

2. Kiem tra GPU:

```bash
!nvidia-smi
```

```python
import tensorflow as tf
print(tf.__version__)
print(tf.config.list_physical_devices('GPU'))
```

3. Train va evaluate:

```bash
!python src/preprocess.py
!python src/train.py
!python src/evaluate.py
```

## 4) Luu model

```bash
!cp models/emotion_model.h5 /content/drive/MyDrive/emotion_model_retrained.h5
```

Tai ve may (tuy chon):

```python
from google.colab import files
files.download('models/emotion_model.h5')
```

## 5) Loi thuong gap

1. Loi `No matching distribution found for tensorflow==2.13.0`.
Nguyen nhan: Python mac dinh Colab qua moi.
Xu ly: dung phuong an A va chay bang `/content/tf213/bin/python`.

2. Da cai TF 2.13.0 nhung van ra version khac.
Nguyen nhan: chay nham `python` mac dinh Colab.
Xu ly: dung dung executable `/content/tf213/bin/python`.

3. GPU khong duoc su dung voi TF 2.13.0.
Nguyen nhan: mismatch CUDA/cuDNN giua Colab moi va TF 2.13.
Xu ly: dung phuong an B (TF 2.19.1) neu can GPU.

4. Khong tim thay `data/test` hoac `data/train`.
Xu ly: kiem tra lai thu muc sau khi giai nen:

```bash
!find data -maxdepth 2 -type d | sort
```

5. Out of memory.
Xu ly: giam `BATCH_SIZE` trong `src/train.py` (vi du 64 -> 32).
