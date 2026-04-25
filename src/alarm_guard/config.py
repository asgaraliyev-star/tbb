from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import yaml


@dataclass
class PomodoroConfig:
    focus_minutes: int
    break_minutes: int


@dataclass
class DrowsinessConfig:
    eyes_closed_seconds_threshold: float
    frame_skip: int
    eye_model_path: str


@dataclass
class AlarmConfig:
    default_video: str
    max_volume: int


@dataclass
class FunGameConfig:
    expression_model_path: str


@dataclass
class AppConfig:
    pomodoro: PomodoroConfig
    drowsiness: DrowsinessConfig
    alarm: AlarmConfig
    fun_game: FunGameConfig



def load_config(path: str | Path) -> AppConfig:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config tapılmadı: {config_path}")

    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    return AppConfig(
        pomodoro=PomodoroConfig(**data["pomodoro"]),
        drowsiness=DrowsinessConfig(**data["drowsiness"]),
        alarm=AlarmConfig(**data["alarm"]),
        fun_game=FunGameConfig(**data["fun_game"]),
    )
