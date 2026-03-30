# Classifier Model Folder

Save the trained TensorFlow classifier model in this folder as:

`monkeypox_classifier.keras`

If you use a different filename or location, update `CLASSIFICATION_MODEL_PATH` in `backend/.env`.

Example save call from training code:

```python
model.save("/Users/nagarajan/ML_DEMO/backend/models/monkeypox_classifier.keras")
```
