"""
model.py
Định nghĩa kiến trúc CNN cho Facial Emotion Recognition
"""

from tensorflow import keras
from tensorflow.keras import layers


def build_cnn(input_shape=(48, 48, 1), num_classes=7) -> keras.Model:
    """
    Xây dựng CNN model 4 khối Conv với BatchNorm + Dropout.
    Phù hợp cho FER2013 48x48 grayscale images.
    
    Architecture:
        4x [Conv2D → BatchNorm → ReLU → MaxPool → Dropout]
        → Flatten → Dense(256) → Dense(7, softmax)
    """
    model = keras.Sequential(name='EmotionCNN')

    # ── Khối 1: 32 filters ────────────────────────────────
    model.add(layers.Conv2D(32, (3,3), padding='same', input_shape=input_shape))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Conv2D(32, (3,3), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D(2, 2))
    model.add(layers.Dropout(0.25))

    # ── Khối 2: 64 filters ────────────────────────────────
    model.add(layers.Conv2D(64, (3,3), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Conv2D(64, (3,3), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D(2, 2))
    model.add(layers.Dropout(0.25))

    # ── Khối 3: 128 filters ───────────────────────────────
    model.add(layers.Conv2D(128, (3,3), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Conv2D(128, (3,3), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D(2, 2))
    model.add(layers.Dropout(0.25))

    # ── Khối 4: 256 filters ───────────────────────────────
    model.add(layers.Conv2D(256, (3,3), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D(2, 2))
    model.add(layers.Dropout(0.25))

    # ── Fully Connected ───────────────────────────────────
    model.add(layers.Flatten())
    model.add(layers.Dense(256))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(0.5))

    model.add(layers.Dense(num_classes, activation='softmax'))

    return model


def compile_model(model: keras.Model, lr: float = 1e-3) -> keras.Model:
    """Compile model với Adam optimizer."""
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


if __name__ == '__main__':
    model = build_cnn()
    compile_model(model)
    model.summary()
    print(f"\nTổng tham số: {model.count_params():,}")
