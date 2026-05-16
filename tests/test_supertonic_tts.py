import importlib
import sys
import types
import numpy as np
import pytest
from pipecat.frames.frames import AggregatedTextFrame, ErrorFrame, TTSAudioRawFrame, TTSSpeakFrame, TTSStartedFrame, TTSStoppedFrame, TTSTextFrame
from pipecat.tests.utils import run_test
from pipecat.transcriptions.language import Language


class _FakeSupertonicTTS:
    instances: list["_FakeSupertonicTTS"] = []

    def __init__(
        self,
        model: str = "supertonic-3",
        auto_download: bool = True,
        intra_op_num_threads: int | None = None,
        inter_op_num_threads: int | None = None,
    ):
        self.model = model
        self.auto_download = auto_download
        self.intra_op_num_threads = intra_op_num_threads
        self.inter_op_num_threads = inter_op_num_threads
        self.sample_rate = 24000
        self.is_multilingual = model != "supertonic"
        self.voice_style_names = ["F1", "M1"]
        self.last_synthesize_call: dict[str, object] | None = None
        self.__class__.instances.append(self)

    def get_voice_style(self, voice_name: str):
        return {"voice_name": voice_name}

    def synthesize(
        self,
        text: str,
        voice_style,
        *,
        total_steps: int = 5,
        speed: float = 1.05,
        max_chunk_length: int | None = None,
        silence_duration: float = 0.3,
        lang: str = "en",
    ):
        self.last_synthesize_call = {
            "text": text,
            "voice_style": voice_style,
            "total_steps": total_steps,
            "speed": speed,
            "max_chunk_length": max_chunk_length,
            "silence_duration": silence_duration,
            "lang": lang,
        }
        waveform = np.array([[0.0, 0.25, -0.25, 0.5]], dtype=np.float32)
        duration = np.array([0.1], dtype=np.float32)
        return waveform, duration


@pytest.fixture
def supertonic_module(monkeypatch):
    fake_module = types.ModuleType("supertonic")
    fake_module.TTS = _FakeSupertonicTTS
    monkeypatch.setitem(sys.modules, "supertonic", fake_module)

    sys.modules.pop("pipecat_supertonic.tts", None)

    module = importlib.import_module("pipecat_supertonic.tts")
    _FakeSupertonicTTS.instances.clear()
    yield module
    sys.modules.pop("pipecat_supertonic.tts", None)


@pytest.mark.asyncio
async def test_run_supertonic_tts_success(supertonic_module):
    tts_service = supertonic_module.SupertonicTTSService(sample_rate=24000)
    await tts_service.warmup()

    down_frames, _ = await run_test(
        tts_service,
        frames_to_send=[TTSSpeakFrame(text="Hello world.")],
    )

    frame_types = [type(frame) for frame in down_frames]

    assert AggregatedTextFrame in frame_types
    assert TTSStartedFrame in frame_types
    assert TTSStoppedFrame in frame_types
    assert TTSTextFrame in frame_types

    audio_frames = [frame for frame in down_frames if isinstance(frame, TTSAudioRawFrame)]
    assert len(audio_frames) == 1
    assert audio_frames[0].sample_rate == 24000

    sdk = _FakeSupertonicTTS.instances[0]
    assert sdk.last_synthesize_call is not None
    assert sdk.last_synthesize_call["lang"] == "en"


@pytest.mark.asyncio
async def test_run_supertonic_tts_invalid_voice(supertonic_module):
    tts_service = supertonic_module.SupertonicTTSService(
        sample_rate=24000,
        stop_frame_timeout_s=0.01,
        settings=supertonic_module.SupertonicTTSService.Settings(voice="BAD_VOICE"),
    )
    await tts_service.warmup()

    _, up_frames = await run_test(
        tts_service,
        frames_to_send=[TTSSpeakFrame(text="Hello world.", append_to_context=False)],
        expected_down_frames=[AggregatedTextFrame, TTSStartedFrame, TTSTextFrame, TTSStoppedFrame],
        expected_up_frames=[ErrorFrame],
    )

    error = up_frames[0]
    assert isinstance(error, ErrorFrame)
    assert "BAD_VOICE" in error.error


def test_supertonic_language_fallback_on_init(supertonic_module):
    tts_service = supertonic_module.SupertonicTTSService(
        settings=supertonic_module.SupertonicTTSService.Settings(language=Language.ZH_CN)
    )

    assert tts_service._settings.language == "na"


@pytest.mark.asyncio
async def test_run_supertonic_tts_requires_warmup(supertonic_module):
    tts_service = supertonic_module.SupertonicTTSService(
        sample_rate=24000,
        stop_frame_timeout_s=0.01,
    )

    _, up_frames = await run_test(
        tts_service,
        frames_to_send=[TTSSpeakFrame(text="Hello world.", append_to_context=False)],
        expected_up_frames=[ErrorFrame],
    )

    error = up_frames[0]
    assert isinstance(error, ErrorFrame)
    assert "warmup" in error.error
