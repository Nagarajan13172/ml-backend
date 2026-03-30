#!/usr/bin/env python3
"""
Generate a minimal placeholder Keras model for API pipeline testing.

The model accepts 224x224 RGB images and outputs 4 class probabilities:
  [Monkeypox, Chickenpox, Measles, Normal]

This is an UNTRAINED demo model — predictions are not medically meaningful.
Replace it with your actual trained model when available.

Usage:
    cd backend
    python scripts/create_demo_model.py
"""
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BACKEND_DIR / "models" / "monkeypox_classifier.keras"

sys.path.insert(0, str(BACKEND_DIR))


def create_demo_model() -> None:
    try:
        import tensorflow as tf
    except ImportError:
        print("ERROR: TensorFlow is not installed.")
        print("  Mac (Apple Silicon): pip install tensorflow-macos tensorflow-metal")
        print("  Other:               pip install tensorflow")
        sys.exit(1)

    print(f"TensorFlow {tf.__version__} loaded.")

    # ── Custom fuzzy layer (must match classification_service.py exactly) ──
    class FuzzyTriangularMembership(tf.keras.layers.Layer):
        def __init__(self, a: float = 0.2, b: float = 0.6, c: float = 1.0, **kwargs):
            super().__init__(**kwargs)
            self.a = float(a)
            self.b = float(b)
            self.c = float(c)

        def call(self, inputs):
            left  = (inputs - self.a) / max(self.b - self.a, 1e-6)
            right = (self.c - inputs) / max(self.c - self.b, 1e-6)
            return tf.clip_by_value(tf.minimum(left, right), 0.0, 1.0)

        def get_config(self):
            cfg = super().get_config()
            cfg.update({"a": self.a, "b": self.b, "c": self.c})
            return cfg

    # ── Build a small CNN ──────────────────────────────────────────────────
    inputs = tf.keras.Input(shape=(224, 224, 3), name="input_image")

    # Normalise [0,255] → [0,1] then apply fuzzy membership
    x = tf.keras.layers.Rescaling(1.0 / 255.0)(inputs)
    x = FuzzyTriangularMembership(a=0.0, b=0.5, c=1.0, name="fuzzy_membership")(x)

    x = tf.keras.layers.Conv2D(16, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(4, activation="softmax", name="predictions")(x)

    model = tf.keras.Model(inputs, outputs, name="monkeypox_classifier_demo")
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    model.summary()

    # ── Save ──────────────────────────────────────────────────────────────
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)

    print()
    print(f"Demo model saved → {MODEL_PATH}")
    print()
    print("IMPORTANT: This is an untrained placeholder.")
    print("           Replace with your trained model for real predictions.")


if __name__ == "__main__":
    create_demo_model()
