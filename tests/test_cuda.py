"""Поиск CUDA DLL для CTranslate2 на Windows."""

from __future__ import annotations

from src.core.cuda import cuda_dll_directories, is_missing_cuda_runtime


def test_torch_lib_is_discovered_when_present() -> None:
    """В этом проекте torch кладёт cublas64_12.dll в site-packages/torch/lib."""
    directories = cuda_dll_directories()
    names = {path.name for path in directories}
    assert "lib" in names or any((path / "cublas64_12.dll").exists() for path in directories)


def test_cublas_error_is_recognized() -> None:
    """Сообщение CTranslate2 про отсутствующий DLL должно давать fallback."""
    error = RuntimeError("Library cublas64_12.dll is not found or cannot be loaded")
    assert is_missing_cuda_runtime(error) is True
    assert is_missing_cuda_runtime(ValueError("bad audio shape")) is False


def test_cudnn_symbol_error_is_recognized() -> None:
    """CTranslate2 подсовывает старый cudnn64_9.dll — PyTorch LSTM падает так."""
    error = OSError("Could not load symbol cudnnGetLibConfig. Error code 127")
    assert is_missing_cuda_runtime(error) is True


def test_discovered_dir_contains_cublas() -> None:
    """Регистрация бесполезна, если в каталоге нет самого DLL."""
    directories = cuda_dll_directories()
    assert any((path / "cublas64_12.dll").is_file() for path in directories)


def test_stt_falls_back_to_cpu_when_cublas_missing() -> None:
    """PTT не должен умирать, если CUDA-библиотека не грузится."""
    from src.core.config import STTSettings
    from src.modules.ptt.stt import WhisperTranscriber

    transcriber = WhisperTranscriber(STTSettings(warmup=False, device="cuda", fallback_to_cpu=True))
    calls: list[tuple[str, str]] = []

    def fake_create(device: str, compute_type: str) -> object:
        calls.append((device, compute_type))
        if device == "cuda":
            raise RuntimeError("Library cublas64_12.dll is not found or cannot be loaded")
        return object()

    transcriber._create_model = fake_create  # type: ignore[method-assign]
    transcriber._probe_encoder = lambda: None
    transcriber._load_sync()

    assert transcriber.device == "cpu"
    assert transcriber.compute_type == "int8"
    assert calls[0][0] == "cuda"
    assert calls[-1] == ("cpu", "int8")
