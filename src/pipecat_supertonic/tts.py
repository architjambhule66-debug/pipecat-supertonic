import asyncio
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import Any
import numpy as np
from loguru import logger
from pipecat.audio.utils import create_stream_resampler
from pipecat.frames.frames import ErrorFrame, Frame, TTSAudioRawFrame
from pipecat.services.settings import NOT_GIVEN, TTSSettings, _NotGiven, assert_given
from pipecat.services.tts_service import TTSService
from pipecat.transcriptions.language import Language
from pipecat.utils.tracing.service_decorators import traced_tts
try:
    from supertonic import TTS as SupertonicSDK
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use Supertonic, you need to `pip install supertonic`.")
    raise Exception(f"Missing module: {e}")


SUPPORTED_LANGUAGES = frozenset(
    {
        "ar",
        "bg",
        "cs",
        "da",
        "de",
        "el",
        "en",
        "es",
        "et",
        "fi",
        "fr",
        "hi",
        "hr",
        "hu",
        "id",
        "it",
        "ja",
        "ko",
        "lt",
        "lv",
        "nl",
        "pl",
        "pt",
        "ro",
        "ru",
        "sk",
        "sl",
        "sv",
        "tr",
        "uk",
        "vi",
    }
)
UNKNOWN_LANGUAGE = "na"


def language_to_supertonic_language(language: Language) -> str:
    base_code = str(language).split("-")[0].lower()
    if base_code in SUPPORTED_LANGUAGES:
        return base_code

    logger.warning(
        f"Language {language} is not supported by Supertonic. Using fallback "
        f"language '{UNKNOWN_LANGUAGE}'."
    )
    return UNKNOWN_LANGUAGE


@dataclass
class SupertonicTTSSettings(TTSSettings):
    """Settings for SupertonicTTSService.

    Parameters:
        speed: Speech speed multiplier.
        total_steps: Number of synthesis steps.
        max_chunk_length: Maximum characters per synthesized chunk.
        silence_duration: Silence inserted between synthesized chunks.
    """

    speed: float | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    total_steps: int | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    max_chunk_length: int | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    silence_duration: float | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)


class SupertonicTTSService(TTSService):
    Settings = SupertonicTTSSettings
    _settings: Settings

    def __init__(
        self,
        *,
        model: str | None = None,
        voice: str | None = None,
        language: Language | str | None = None,
        speed: float | None = None,
        total_steps: int | None = None,
        max_chunk_length: int | None = None,
        silence_duration: float | None = None,
        auto_download: bool = True,
        intra_op_num_threads: int | None = None,
        inter_op_num_threads: int | None = None,
        sample_rate: int | None = None,
        settings: Settings | None = None,
        **kwargs,
    ):
        """Initialize the Supertonic TTS service.

        Args:
            model: Supertonic model name.
            voice: Supertonic voice name.
            language: Language for synthesis.
            speed: Speech speed multiplier.
            total_steps: Number of synthesis steps.
            max_chunk_length: Maximum characters per synthesized chunk.
            silence_duration: Silence inserted between synthesized chunks.
            auto_download: Whether to download model assets automatically.
            intra_op_num_threads: ONNX intra-op thread count.
            inter_op_num_threads: ONNX inter-op thread count.
            sample_rate: Output sample rate for generated audio.
            settings: Runtime-updatable settings. When provided alongside direct
                parameters, ``settings`` values take precedence.
            **kwargs: Additional keyword arguments passed to ``TTSService``.
        """
        default_settings = self.Settings(
            model="supertonic-3",
            voice="M1",
            language=Language.EN,
            speed=1.05,
            total_steps=5,
            max_chunk_length=None,
            silence_duration=0.3,
        )

        if model is not None:
            default_settings.model = model
        if voice is not None:
            default_settings.voice = voice
        if language is not None:
            default_settings.language = language
        if speed is not None:
            default_settings.speed = speed
        if total_steps is not None:
            default_settings.total_steps = total_steps
        if max_chunk_length is not None:
            default_settings.max_chunk_length = max_chunk_length
        if silence_duration is not None:
            default_settings.silence_duration = silence_duration

        if settings is not None:
            default_settings.apply_update(settings)

        super().__init__(
            sample_rate=sample_rate,
            push_start_frame=True,
            push_stop_frames=True,
            settings=default_settings,
            **kwargs,
        )

        self._auto_download = auto_download
        self._intra_op_num_threads = intra_op_num_threads
        self._inter_op_num_threads = inter_op_num_threads

        self._resampler = create_stream_resampler()
        self._tts: Any | None = None
        self._voice_styles: dict[str, object] = {}
        self._available_voice_names: tuple[str, ...] = ()
        self._tts_lock = asyncio.Lock()

    async def warmup(self) -> None:
        """Download and initialize Supertonic assets for this service instance.

        Call this during application startup before the service is used in a
        live Pipecat pipeline. This avoids first-request cold-start delays and
        keeps TTS frame ordering stable during active calls.
        """
        await self._ensure_tts()

    def can_generate_metrics(self) -> bool:
        """Indicate that this service supports TTFB and usage metrics."""
        return True

    def language_to_service_language(self, language: Language) -> str:
        """Convert a Pipecat language enum to Supertonic's language format."""
        return language_to_supertonic_language(language)

    async def _update_settings(self, delta: Settings) -> dict[str, object]:
        """Apply a settings delta.

        Model updates clear the cached SDK instance so the next synthesis call
        reinitializes with the updated model.
        """
        changed = await super()._update_settings(delta)
        if "model" in changed:
            async with self._tts_lock:
                self._tts = None
                self._voice_styles.clear()
                self._available_voice_names = ()
        return changed

    async def _ensure_tts(self) -> Any:
        if self._tts is not None:
            return self._tts

        async with self._tts_lock:
            if self._tts is None:
                model = assert_given(self._settings.model)
                self._tts = await asyncio.to_thread(
                    SupertonicSDK,
                    model=model,
                    auto_download=self._auto_download,
                    intra_op_num_threads=self._intra_op_num_threads,
                    inter_op_num_threads=self._inter_op_num_threads,
                )
                self._available_voice_names = tuple(self._tts.voice_style_names)
        return self._tts

    def _require_warmup(self) -> Any:
        if self._tts is None:
            raise RuntimeError(
                "SupertonicTTSService is not warmed up. Call `await tts.warmup()` "
                "during application startup before using the service."
            )
        return self._tts

    async def _get_voice_style(self, voice_name: str) -> object:
        tts = self._require_warmup()

        if voice_name not in self._available_voice_names:
            valid_voices = ", ".join(sorted(self._available_voice_names)) or "none"
            raise ValueError(
                f"Supertonic TTS voice {voice_name!r} is not supported "
                f"(must be one of: {valid_voices})"
            )

        cached = self._voice_styles.get(voice_name)
        if cached is not None:
            return cached

        style = await asyncio.to_thread(tts.get_voice_style, voice_name)
        self._voice_styles[voice_name] = style
        return style

    def _waveform_to_pcm16(self, waveform: np.ndarray) -> bytes:
        """Convert a Supertonic waveform array to mono PCM16 bytes."""
        audio = np.asarray(waveform)

        if audio.ndim == 2:
            if audio.shape[0] == 1:
                audio = audio[0]
            elif audio.shape[1] == 1:
                audio = audio[:, 0]
            else:
                raise ValueError(f"Expected mono audio from Supertonic, got shape {audio.shape}")
        elif audio.ndim != 1:
            raise ValueError(f"Expected 1-D or mono 2-D audio from Supertonic, got {audio.shape}")

        if audio.size == 0:
            raise ValueError("Supertonic returned empty audio")

        if np.issubdtype(audio.dtype, np.floating):
            audio = np.clip(audio, -1.0, 1.0)
            audio = (audio * np.iinfo(np.int16).max).astype(np.int16)
        elif np.issubdtype(audio.dtype, np.integer):
            audio = np.clip(audio, np.iinfo(np.int16).min, np.iinfo(np.int16).max).astype(np.int16)
        else:
            raise TypeError(f"Unsupported Supertonic waveform dtype: {audio.dtype}")

        return audio.tobytes()

    @traced_tts
    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        """Generate speech from text using Supertonic."""
        logger.debug(f"{self}: Generating TTS [{text}]")

        try:
            await self.start_tts_usage_metrics(text)

            voice = assert_given(self._settings.voice)
            if voice is None:
                raise ValueError("Supertonic TTS voice must be specified")

            language = assert_given(self._settings.language)
            speed = assert_given(self._settings.speed)
            total_steps = assert_given(self._settings.total_steps)
            max_chunk_length = assert_given(self._settings.max_chunk_length)
            silence_duration = assert_given(self._settings.silence_duration)

            tts = self._require_warmup()
            voice_style = await self._get_voice_style(voice)

            synthesis_language = language or UNKNOWN_LANGUAGE
            if not tts.is_multilingual:
                synthesis_language = "en"

            waveform, _ = await asyncio.to_thread(
                tts.synthesize,
                text,
                voice_style,
                total_steps=total_steps,
                speed=speed,
                max_chunk_length=max_chunk_length,
                silence_duration=silence_duration,
                lang=synthesis_language,
            )

            await self.stop_ttfb_metrics()

            audio = self._waveform_to_pcm16(waveform)
            if tts.sample_rate != self.sample_rate:
                audio = await self._resampler.resample(audio, tts.sample_rate, self.sample_rate)

            yield TTSAudioRawFrame(
                audio=audio,
                sample_rate=self.sample_rate,
                num_channels=1,
                context_id=context_id,
            )
        except Exception as e:
            yield ErrorFrame(error=f"Unknown error occurred: {e}")
        finally:
            await self.stop_ttfb_metrics()
