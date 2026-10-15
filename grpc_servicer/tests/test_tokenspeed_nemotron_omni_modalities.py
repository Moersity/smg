"""The TokenSpeed servicer advertises Nemotron-H Omni's encoders from the engine's
generic flags: multimodal (image), a video token (video) and an audio model with an
audio token (audio).

Run with: pytest grpc_servicer/tests/test_tokenspeed_nemotron_omni_modalities.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

pytest.importorskip("msgspec")
pytest.importorskip("zmq")
pytest.importorskip("numpy")
pytest.importorskip("smg_grpc_proto")
from smg_grpc_proto.generated import common_pb2  # noqa: E402

_ROOT = Path(__file__).parents[1] / "smg_grpc_servicer" / "tokenspeed"
MODEL_TYPE = "NemotronH_Nano_Omni_Reasoning_V3"


@pytest.fixture
def servicer_mod(monkeypatch):
    # Importing the engine (and torch) loads the runtime's dependencies; these
    # stand-ins cover the servicer module's imports and nothing of the
    # modality advertisement.
    symbols = {
        "torch": "",
        "tokenspeed.runtime.multimodal.inputs": "Modality MultimodalDataItem MultimodalInputs",
        "tokenspeed.runtime.multimodal.shm_transport": "ShmTensorHandle",
        "tokenspeed.runtime.pd.kv_events": "KVEventBatch",
    }
    for name, attrs in symbols.items():
        parts = name.split(".")
        for end in range(1, len(parts)):
            parent = ".".join(parts[:end])
            if parent not in sys.modules:
                module = ModuleType(parent)
                module.__path__ = []
                monkeypatch.setitem(sys.modules, parent, module)
        module = ModuleType(name)
        for attr in attrs.split():
            setattr(module, attr, object)
        monkeypatch.setitem(sys.modules, name, module)
    name = "smg_grpc_servicer.tokenspeed.servicer"
    spec = importlib.util.spec_from_file_location(name, _ROOT / "servicer.py")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


def _modalities(servicer_mod, hf_config):
    # The engine's ModelConfig flags for this architecture.
    model_config = SimpleNamespace(
        is_multimodal_active=True, is_multimodal=True, is_audio_model=True
    )
    return servicer_mod.TokenSpeedSchedulerServicer._static_supported_modalities(
        model_config, hf_config
    )


def test_vision_and_sound_towers_advertise_image_video_and_audio(servicer_mod):
    config = {"model_type": MODEL_TYPE, "video_token_id": 18, "audio_token_id": 27}
    assert _modalities(servicer_mod, config) == [
        common_pb2.IMAGE,
        common_pb2.VIDEO,
        common_pb2.AUDIO,
    ]


def test_without_an_audio_token_audio_is_not_advertised(servicer_mod):
    config = {"model_type": MODEL_TYPE, "video_token_id": 18, "audio_token_id": None}
    assert _modalities(servicer_mod, config) == [common_pb2.IMAGE, common_pb2.VIDEO]
