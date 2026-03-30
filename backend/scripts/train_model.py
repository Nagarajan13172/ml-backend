#!/usr/bin/env python3
"""
Train the monkeypox classifier using EfficientNetB0 + FuzzyTriangularMembership.

── Dataset layout expected ──────────────────────────────────────────────────
backend/
└── data/
    ├── train/
    │   ├── Monkeypox/      ← lesion images
    │   ├── Chickenpox/
    │   ├── Measles/
    │   └── Normal/
    └── val/                ← optional; auto-split from train if absent
        ├── Monkeypox/
        ├── Chickenpox/
        ├── Measles/
        └── Normal/

── Usage ─────────────────────────────────────────────────────────────────────
    cd backend
    python scripts/train_model.py

    # Override defaults:
    python scripts/train_model.py --epochs 30 --batch 16 --lr 1e-4
──────────────────────────────────────────────────────────────────────────────
"""
import argparse
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
DATA_DIR    = BACKEND_DIR / "data" / "train"
VAL_DIR     = BACKEND_DIR / "data" / "val"
MODEL_PATH  = BACKEND_DIR / "models" / "monkeypox_classifier.keras"

CLASS_NAMES = ["Monkeypox", "Chickenpox", "Measles", "Normal"]
IMG_SIZE    = 224


# ── Custom fuzzy layer (must match classification_service.py exactly) ─────────
def build_fuzzy_layer(tf):
    class FuzzyTriangularMembership(tf.keras.layers.Layer):
        def __init__(self, a: float = 0.2, b: float = 0.6, c: float = 1.0, **kwargs):
            super().__init__(**kwargs)
            self.a, self.b, self.c = float(a), float(b), float(c)

        def call(self, inputs):
            left  = (inputs - self.a) / max(self.b - self.a, 1e-6)
            right = (self.c - inputs) / max(self.c - self.b, 1e-6)
            return tf.clip_by_value(tf.minimum(left, right), 0.0, 1.0)

        def get_config(self):
            cfg = super().get_config()
            cfg.update({"a": self.a, "b": self.b, "c": self.c})
            return cfg

    return FuzzyTriangularMembership


# ── Build model ───────────────────────────────────────────────────────────────
def build_model(tf, num_classes: int):
    FuzzyLayer = build_fuzzy_layer(tf)

    # Base: EfficientNetB0 pre-trained on ImageNet, no top
    base = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )
    base.trainable = False  # freeze base initially

    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="input_image")

    # Preprocessing (EfficientNet expects [0,255] but we normalise first)
    x = tf.keras.layers.Rescaling(1.0 / 255.0)(inputs)
    x = FuzzyTriangularMembership(a=0.0, b=0.5, c=1.0, name="fuzzy_membership")(x)

    # Rescale back to [0,255] for EfficientNet preprocessing
    x = tf.keras.layers.Rescaling(255.0)(x)
    x = tf.keras.applications.efficientnet.preprocess_input(x)

    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    return tf.keras.Model(inputs, outputs, name="monkeypox_classifier"), base


# ── Data pipeline ─────────────────────────────────────────────────────────────
def build_datasets(tf, batch_size: int):
    if not DATA_DIR.exists():
        print(f"\nERROR: Training data not found at: {DATA_DIR}")
        print("Create the directory structure:")
        print("  backend/data/train/Monkeypox/")
        print("  backend/data/train/Chickenpox/")
        print("  backend/data/train/Measles/")
        print("  backend/data/train/Normal/")
        print("Then add images to each class folder and re-run.\n")
        sys.exit(1)

    augment = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.2),
        tf.keras.layers.RandomZoom(0.15),
        tf.keras.layers.RandomBrightness(0.15),
        tf.keras.layers.RandomContrast(0.15),
    ], name="augmentation")

    common_kwargs = dict(
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=batch_size,
        class_names=CLASS_NAMES,
        label_mode="categorical",
        interpolation="bilinear",
        shuffle=True,
        seed=42,
    )

    if VAL_DIR.exists():
        train_ds = tf.keras.utils.image_dataset_from_directory(str(DATA_DIR), **common_kwargs)
        val_ds   = tf.keras.utils.image_dataset_from_directory(str(VAL_DIR),  **{**common_kwargs, "shuffle": False})
    else:
        print("No val/ directory found — using 80/20 train/val split from train/")
        train_ds = tf.keras.utils.image_dataset_from_directory(
            str(DATA_DIR), validation_split=0.2, subset="training",  **common_kwargs
        )
        val_ds   = tf.keras.utils.image_dataset_from_directory(
            str(DATA_DIR), validation_split=0.2, subset="validation", **{**common_kwargs, "shuffle": False}
        )

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.map(
        lambda x, y: (augment(x, training=True), y), num_parallel_calls=AUTOTUNE
    ).prefetch(AUTOTUNE)
    val_ds = val_ds.prefetch(AUTOTUNE)

    return train_ds, val_ds


# ── Training ──────────────────────────────────────────────────────────────────
def train(epochs: int = 20, batch_size: int = 32, lr: float = 1e-3, fine_tune_lr: float = 1e-5):
    try:
        import tensorflow as tf
    except ImportError:
        print("ERROR: TensorFlow not installed.")
        sys.exit(1)

    print(f"TensorFlow {tf.__version__}")

    train_ds, val_ds = build_datasets(tf, batch_size)
    num_classes = len(CLASS_NAMES)

    model, base = build_model(tf, num_classes)
    model.summary(show_trainable=True)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            str(MODEL_PATH), monitor="val_accuracy", save_best_only=True, verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=5, restore_best_weights=True, verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, verbose=1,
        ),
    ]

    # ── Phase 1: train head only (frozen base) ──
    print(f"\n{'='*60}")
    print("Phase 1 — training classifier head (base frozen)")
    print(f"{'='*60}\n")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks)

    # ── Phase 2: fine-tune top layers of EfficientNet ──
    print(f"\n{'='*60}")
    print("Phase 2 — fine-tuning top 30 layers of EfficientNetB0")
    print(f"{'='*60}\n")
    base.trainable = True
    # Freeze all but the last 30 layers
    for layer in base.layers[:-30]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(fine_tune_lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(train_ds, validation_data=val_ds, epochs=epochs // 2, callbacks=callbacks)

    print(f"\nBest model saved → {MODEL_PATH}")
    print("Restart the backend server to load the new model.")


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train monkeypox classifier")
    parser.add_argument("--epochs",   type=int,   default=20,   help="Phase-1 epochs (default 20)")
    parser.add_argument("--batch",    type=int,   default=32,   help="Batch size (default 32)")
    parser.add_argument("--lr",       type=float, default=1e-3, help="Head learning rate (default 1e-3)")
    parser.add_argument("--fine-lr",  type=float, default=1e-5, help="Fine-tune LR (default 1e-5)")
    args = parser.parse_args()

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    train(epochs=args.epochs, batch_size=args.batch, lr=args.lr, fine_tune_lr=args.fine_lr)
