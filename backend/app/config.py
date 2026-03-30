from pathlib import Path
from typing import List

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


BACKEND_DIR = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(BACKEND_DIR / ".env"),
        env_file_encoding="utf-8",
    )

    APP_NAME: str = "Monkeypox AI API"
    APP_VERSION: str = "1.1.0"
    DEBUG: bool = True
    ALLOWED_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:5173", "http://localhost:8080"]
    CLASSIFICATION_MODEL_PATH: str = "models/monkeypox_classifier.keras"
    CLASSIFICATION_IMAGE_SIZE: int = 224
    CLASSIFICATION_CLASS_NAMES: str = ""

    @field_validator("DEBUG", mode="before")
    @classmethod
    def normalize_debug_flag(cls, value):
        if isinstance(value, bool):
            return value

        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"1", "true", "yes", "on", "debug", "development"}:
                return True
            if normalized in {"0", "false", "no", "off", "release", "prod", "production"}:
                return False

        return value

    @property
    def classification_model_path(self) -> Path:
        model_path = Path(self.CLASSIFICATION_MODEL_PATH)
        if model_path.is_absolute():
            return model_path
        return (BACKEND_DIR / model_path).resolve()

    @property
    def classification_class_names(self) -> List[str]:
        return [
            name.strip()
            for name in self.CLASSIFICATION_CLASS_NAMES.split(",")
            if name.strip()
        ]


settings = Settings()
