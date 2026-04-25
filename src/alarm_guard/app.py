from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from alarm_guard.alarm import AlarmPlayer
from alarm_guard.config import load_config
from alarm_guard.fun_game import MonkeyEmojiGame
from alarm_guard.pomodoro import PomodoroEngine
from alarm_guard.vision import DrowsinessDetector


def run(config_path: str, custom_alarm: str | None = None) -> None:
    cfg = load_config(config_path)
    alarm_source = custom_alarm or cfg.alarm.default_video

    detector = DrowsinessDetector(
        eye_model_path=cfg.drowsiness.eye_model_path,
        threshold_seconds=cfg.drowsiness.eyes_closed_seconds_threshold,
        frame_skip=cfg.drowsiness.frame_skip,
    )
    alarm = AlarmPlayer(alarm_source, cfg.alarm.max_volume)
    pomodoro = PomodoroEngine(cfg.pomodoro.focus_minutes, cfg.pomodoro.break_minutes)
    game = MonkeyEmojiGame(cfg.fun_game.expression_model_path)

    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        raise RuntimeError("Kamera açıla bilmədi")

    print("App başladı. Q çıxış, G monkey game.")
    alarm_running = False

    try:
        while True:
            ok, frame = cam.read()
            if not ok:
                break

            timer_state = pomodoro.tick()
            signal = detector.process(frame)

            cv2.putText(
                frame,
                f"{timer_state.mode} {pomodoro.format_remaining(timer_state.seconds_left)}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.85,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                frame,
                f"Closed: {signal.eyes_closed_for:.2f}s",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 255, 255),
                2,
            )

            if signal.triggered and not alarm_running:
                alarm.play()
                alarm_running = True

            key = cv2.waitKey(1) & 0xFF
            if key == ord("g"):
                label, emoji, conf = game.classify(frame)
                print(f"Monkey game => {label} {emoji} ({conf:.2%})")
            elif key == ord("s"):
                alarm.stop()
                alarm_running = False
            elif key == ord("q"):
                break

            cv2.imshow("AlarmGuard", frame)

    finally:
        alarm.stop()
        cam.release()
        cv2.destroyAllWindows()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AlarmGuard - Pomodoro + drowsiness alarm")
    parser.add_argument("--config", default="config.yaml", help="Config faylı")
    parser.add_argument("--alarm", default=None, help="Alarm video URL və ya lokal fayl")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    config_path = str(Path(args.config).resolve())
    run(config_path=config_path, custom_alarm=args.alarm)
