#!/usr/bin/env python3
"""
Train the skin lesion classifier using EfficientNetB0 + FuzzyTriangularMembership.
Classes are auto-detected from backend/data/train/ subdirectories.

Usage:
    cd backend
    python scripts/train_model.py
    python scripts/train_model.py --epochs 20 --batch 16
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
        print(f"ERROR: {TRAIN_DIR} not found.")
        sys.exit(1)
    names = sorted(p.name for p in TRAIN_DIR.iterdir() if p.is_dir())
    if not names:
        print(f"ERROR: No class subdirectories in {TRAIN_DIR}")
        sys.exit(1)
    return names


def build_model(tf, num_classes):
    FuzzyLayer = build_fuzzy_layer(tf)

    base = tf.keras.applications.EfficientNetB0(
        include_top=False, weights="imagenet",
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

    return tf.keras.Model(inputs, outputs, name="skin_lesion_classifier"), base


def build_datasets(tf, class_names, batch_size):
    AUTOTUNE = tf.data.AUTOTUNE

    augment = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.3),
        tf.keras.layers.RandomZoom(0.2),
        tf.keras.layers.RandomBrightness(0.2),
        tf.keras.layers.RandomContrast(0.2),
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


def train(epochs=20, batch_size=16, head_lr=1e-3, fine_lr=1e-5):
    try:
        import tensorflow as tf
    except ImportError:
        print("ERROR: TensorFlow not installed.")
        sys.exit(1)

    print(f"TensorFlow {tf.__version__}")
    class_names = get_class_names()
    num_classes = len(class_names)
    print(f"Classes ({num_classes}): {class_names}")

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
            monitor="val_accuracy", patience=6,
            restore_best_weights=True, verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, verbose=1,
        ),
    ]

    # ── Phase 1: head only ───────────────────────────────────────────────
    print(f"\n{'='*55}\nPhase 1 — head training ({epochs} epochs)\n{'='*55}")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(head_lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks)
    best_acc = max(history.history.get("val_accuracy", [0]))
    print(f"\nBest Phase-1 val_accuracy: {best_acc:.4f}")

    # ── Phase 2: gentle fine-tune top 20 layers (only if Phase 1 converged) ──
    if best_acc >= 0.70:
        print(f"\n{'='*55}\nPhase 2 — fine-tuning top 20 layers\n{'='*55}")
        base.trainable = True
        for layer in base.layers[:-20]:
            layer.trainable = False

        # Use separate checkpoint path so Phase 1 best model is never overwritten
        ft_checkpoint = str(MODEL_PATH).replace(".keras", "_ft.keras")
        ft_callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                ft_checkpoint, monitor="val_accuracy",
                save_best_only=True, verbose=1,
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor="val_accuracy", patience=5,
                restore_best_weights=True, verbose=1,
            ),
        ]
        model.compile(
            optimizer=tf.keras.optimizers.Adam(fine_lr),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        ft_history = model.fit(
            train_ds, validation_data=val_ds,
            epochs=max(epochs // 2, 5), callbacks=ft_callbacks,
        )
        ft_best = max(ft_history.history.get("val_accuracy", [0]))
        # Only adopt fine-tuned model if it's strictly better
        if ft_best > best_acc:
            import shutil
            shutil.copy2(ft_checkpoint, str(MODEL_PATH))
            print(f"Fine-tuned model adopted ({ft_best:.4f} > {best_acc:.4f})")
        else:
            print(f"Keeping Phase-1 model ({best_acc:.4f} >= fine-tuned {ft_best:.4f})")
    else:
        print("Phase-1 accuracy < 70% — skipping fine-tune. Add more data or epochs.")

    # Save class names to .env so the API picks them up
    update_env(class_names)

    print(f"\nModel saved → {MODEL_PATH}")
    print("Restart the backend server to load the trained model.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",   type=int,   default=20)
    parser.add_argument("--batch",    type=int,   default=16)
    parser.add_argument("--lr",       type=float, default=1e-3)
    parser.add_argument("--fine-lr",  type=float, default=1e-5)
    args = parser.parse_args()
    train(args.epochs, args.batch, args.lr, args.fine_lr)
