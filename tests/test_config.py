import sys
from unittest.mock import MagicMock

from swsearch.config import ModelSettings, _default_device


def test_default_device_uses_cuda_when_available(monkeypatch):
    fake_torch = MagicMock()
    fake_torch.cuda.is_available.return_value = True
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    assert _default_device() == "cuda"


def test_default_device_falls_back_to_cpu_when_unavailable(monkeypatch):
    fake_torch = MagicMock()
    fake_torch.cuda.is_available.return_value = False
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    assert _default_device() == "cpu"


def test_default_device_falls_back_to_cpu_when_torch_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)  # forces `import torch` to raise ImportError

    assert _default_device() == "cpu"


def test_model_settings_device_defaults_from_factory(monkeypatch):
    fake_torch = MagicMock()
    fake_torch.cuda.is_available.return_value = True
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    assert ModelSettings().device == "cuda"


def test_model_settings_device_override_bypasses_autodetect(monkeypatch):
    fake_torch = MagicMock()
    fake_torch.cuda.is_available.return_value = True
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    assert ModelSettings(device="cpu").device == "cpu"
