"""Стриминговый синтез речи и воспроизведение аудио."""

from src.modules.tts.engine import KokoroEngine
from src.modules.tts.player import SoundPlayer
from src.modules.tts.streamer import MockTTSModule, TTSModule

__all__ = ["KokoroEngine", "MockTTSModule", "SoundPlayer", "TTSModule"]
