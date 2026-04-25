from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import time


class Mode(str, Enum):
    FOCUS = "FOCUS"
    BREAK = "BREAK"


@dataclass
class PomodoroState:
    mode: Mode
    seconds_left: int
    cycle_count: int


class PomodoroEngine:
    def __init__(self, focus_minutes: int, break_minutes: int) -> None:
        self.focus_seconds = focus_minutes * 60
        self.break_seconds = break_minutes * 60
        self.state = PomodoroState(mode=Mode.FOCUS, seconds_left=self.focus_seconds, cycle_count=1)

    def tick(self) -> PomodoroState:
        time.sleep(1)
        self.state.seconds_left -= 1
        if self.state.seconds_left <= 0:
            self._switch_mode()
        return self.state

    def _switch_mode(self) -> None:
        if self.state.mode == Mode.FOCUS:
            self.state.mode = Mode.BREAK
            self.state.seconds_left = self.break_seconds
        else:
            self.state.mode = Mode.FOCUS
            self.state.seconds_left = self.focus_seconds
            self.state.cycle_count += 1

    @staticmethod
    def format_remaining(seconds: int) -> str:
        mins, secs = divmod(max(seconds, 0), 60)
        return f"{mins:02d}:{secs:02d}"
