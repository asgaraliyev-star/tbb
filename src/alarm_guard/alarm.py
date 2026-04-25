from __future__ import annotations

import vlc


class AlarmPlayer:
    def __init__(self, source: str, volume: int = 200) -> None:
        self.instance = vlc.Instance()
        self.player = self.instance.media_player_new()
        self.source = source
        self.volume = max(0, min(volume, 200))

    def play(self) -> None:
        media = self.instance.media_new(self.source)
        self.player.set_media(media)
        self.player.audio_set_volume(self.volume)
        self.player.set_fullscreen(True)
        self.player.play()

    def stop(self) -> None:
        self.player.stop()
