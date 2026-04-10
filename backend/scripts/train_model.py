#!/usr/bin/env python3
"""
Train the Monkeypox skin lesion classifier.
Architecture: EfficientNetB0 + FuzzyTriangularMembership
Classes auto-detected from backend/data/train/ subdirectories.

Usage:
    cd backend
    python scripts/train_model.py
    python scripts/train_model.py --epochs 30 --batch 16
"""
import argparse
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
TRAIN_DIR   = BACKEND_DIR / "data" / "train"
VAL_DIR     = BACKEND_DIR / "data" / "val"
MODEL_PATH  = BACKEND_DIR / "models" / "monkeypox_classifier.keras"
ENV_PATH    = BACKEND_DIR / ".env"
IMG_SIZE    = 224


def build_fuzzy_layer(tf):
    class FuzzyTriangularMembership(tf.keras.layers.Layer):
        def __init__(self, a=0.2, b=0.6, c=1.0, **kwargs):
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


def get_class_names():
    if not TRAIN_DIR.exists():
        print(f"ERROR: {TRAIN_DIR} not found. Run scripts/prepare_data.py first.")
        sys.exit(1)
    names = sorted(p.name for p in TRAIN_DIR.iterdir() if p.is_dir())
    if not names:
        print(f"ERROR: No class subdirectories in {TRAIN_DIR}")
        sys.exit(1)
    return names


def compute_class_weights(class_names):
    """Compute balanced class weights to handle dataset imbalance."""
    import numpy as np

    counts = []
    for cls in class_names:
        n = len(list((TRAIN_DIR / cls).glob("*")))
        counts.append(n)
        print(f"  {cls}: {n} train images")

    total = sum(counts)
    n_classes = len(class_names)
    # sklearn-style balanced weights: n_total / (n_classes * n_i)
    weights = {i: total / (n_classes * c) for i, c in enumerate(counts)}
    print(f"\nClass weights: { {class_names[i]: f'{w:.3f}' for i, w in weights.items()} }")
    return weights


def build_model(tf, num_classes):
    FuzzyLayer = build_fuzzy_layer(tf)

    base = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )
    base.trainable = False

    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="input_image")
    x = tf.keras.layers.Rescaling(1.0 / 255.0)(inputs)
    x = FuzzyLayer(a=0.0, b=0.5, c=1.0, name="fuzzy_membership")(x)
    x = tf.keras.layers.Rescaling(255.0)(x)
    x = tf.keras.applications.efficientnet.preprocess_input(x)
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    return tf.keras.Model(inputs, outputs, name="monkeypox_classifier"), base


def build_datasets(tf, class_names, batch_size):
    AUTOTUNE = tf.data.AUTOTUNE

    # Strong augmentation for medical imaging — helps generalise across
    # lighting conditions, skin tones, and camera angles.
    augment = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.3),
        tf.keras.layers.RandomZoom(0.2),
        tf.keras.layers.RandomBrightness(0.25),
        tf.keras.layers.RandomContrast(0.25),
        tf.keras.layers.RandomTranslation(0.1, 0.1),
    ])

    common = dict(
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=batch_size,
        class_names=class_names,
        label_mode="categorical",
        interpolation="bilinear",
        seed=42,
    )

    if VAL_DIR.exists():
        train_ds = tf.keras.utils.image_dataset_from_directory(
            str(TRAIN_DIR), shuffle=True, **common
        )
        val_ds = tf.keras.utils.image_dataset_from_directory(
            str(VAL_DIR), shuffle=False, **common
        )
    else:
        train_ds = tf.keras.utils.image_dataset_from_directory(
            str(TRAIN_DIR), validation_split=0.2, subset="training", shuffle=True, **common
        )
        val_ds = tf.keras.utils.image_dataset_from_directory(
            str(TRAIN_DIR), validation_split=0.2, subset="validation", shuffle=False, **common
        )

    train_ds = (
        train_ds
        .map(lambda x, y: (augment(x, training=True), y), num_parallel_calls=AUTOTUNE)
        .prefetch(AUTOTUNE)
    )
    val_ds = val_ds.prefetch(AUTOTUNE)
    return train_ds, val_ds


def update_env(class_names):
    """Write the learned class names back into .env so the API uses them."""
    text = ENV_PATH.read_text()
    new_val = ",".join(class_names)
    lines = []
    found = False
    for line in text.splitlines():
        if line.startswith("CLASSIFICATION_CLASS_NAMES="):
            lines.append(f"CLASSIFICATION_CLASS_NAMES={new_val}")
            found = True
        else:
            lines.append(line)
    if not found:
        lines.append(f"CLASSIFICATION_CLASS_NAMES={new_val}")
    ENV_PATH.write_text("\n".join(lines) + "\n")
    print(f"Updated .env → CLASSIFICATION_CLASS_NAMES={new_val}")


def train(epochs=30, batch_size=16, head_lr=1e-3, fine_lr=1e-5):
    try:
        import tensorflow as tf
    except ImportError:
        print("ERROR: TensorFlow not installed.")
        sys.exit(1)

    print(f"TensorFlow {tf.__version__}")
    class_names = get_class_names()
    num_classes = len(class_names)
    print(f"\nClasses ({num_classes}): {class_names}")
    print("\nImage counts:")
    class_weights = compute_class_weights(class_names)

    train_ds, val_ds = build_datasets(tf, class_names, batch_size)
    model, base      = build_model(tf, num_classes)
    model.summary(show_trainable=True)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            str(MODEL_PATH), monitor="val_accuracy",
            save_best_only=True, verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=8,
            restore_best_weights=True, verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=4, min_lr=1e-7, verbose=1,
        ),
    ]

    # ── Phase 1: train head only ──────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Phase 1 — head training ({epochs} epochs, lr={head_lr})")
    print(f"{'='*60}")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(head_lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        class_weight=class_weights,
    )
    best_acc = max(history.history.get("val_accuracy", [0]))
    print(f"\nBest Phase-1 val_accuracy: {best_acc:.4f}")

    # ── Phase 2: fine-tune top layers (only when Phase 1 converged well) ──────
    if best_acc >= 0.70:
        print(f"\n{'='*60}")
        print(f"Phase 2 — fine-tuning top 30 EfficientNet layers (lr={fine_lr})")
        print(f"{'='*60}")
        base.trainable = True
        for layer in base.layers[:-30]:
            layer.trainable = False

        ft_checkpoint = str(MODEL_PATH).replace(".keras", "_ft.keras")
        ft_callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                ft_checkpoint, monitor="val_accuracy",
                save_best_only=True, verbose=1,
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor="val_accuracy", patience=6,
                restore_best_weights=True, verbose=1,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=3, min_lr=1e-8, verbose=1,
            ),
        ]
        model.compile(
            optimizer=tf.keras.optimizers.Adam(fine_lr),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        ft_history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=max(epochs // 2, 10),
            callbacks=ft_callbacks,
            class_weight=class_weights,
        )
        ft_best = max(ft_history.history.get("val_accuracy", [0]))

        if ft_best > best_acc:
            import shutil
            shutil.copy2(ft_checkpoint, str(MODEL_PATH))
            print(f"Fine-tuned model adopted ({ft_best:.4f} > {best_acc:.4f})")
        else:
            print(f"Keeping Phase-1 model ({best_acc:.4f} >= fine-tuned {ft_best:.4f})")
    else:
        print(
            f"Phase-1 val_accuracy {best_acc:.4f} < 70% — skipping fine-tune. "
            "Consider adding more data or increasing epochs."
        )

    update_env(class_names)
    print(f"\nModel saved → {MODEL_PATH}")
    print("Restart the backend server to load the new model.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Monkeypox lesion classifier")
    parser.add_argument("--epochs",  type=int,   default=30,  help="Phase-1 epochs (default 30)")
    parser.add_argument("--batch",   type=int,   default=16,  help="Batch size (default 16)")
    parser.add_argument("--lr",      type=float, default=1e-3, help="Head learning rate")
    parser.add_argument("--fine-lr", type=float, default=1e-5, help="Fine-tune learning rate")
    args = parser.parse_args()
    train(args.epochs, args.batch, args.lr, args.fine_lr)
