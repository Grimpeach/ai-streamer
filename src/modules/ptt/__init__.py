"""Push-to-Talk хоста: перехват клавиши, запись микрофона и STT (Priority 1)."""

from src.modules.ptt.hotkey import MockPushToTalkModule, PushToTalkModule
from src.modules.ptt.recorder import AudioRecorder, RecordingResult
from src.modules.ptt.stt import Transcript, WhisperTranscriber

__all__ = [
    "AudioRecorder",
    "MockPushToTalkModule",
    "PushToTalkModule",
    "RecordingResult",
    "Transcript",
    "WhisperTranscriber",
]
