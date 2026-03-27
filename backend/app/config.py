from pydantic_settings import BaseSettings
from typing import List
import json
import os


class Settings(BaseSettings):
    APP_NAME: str = "Fuzzy Segmentation API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True
    ALLOWED_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:5173", "http://localhost:8080"]

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
