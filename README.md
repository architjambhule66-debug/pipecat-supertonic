# pipecat-supertonic

`pipecat-supertonic` provides a Pipecat-compatible `TTSService` wrapper for the
official [Supertonic](https://github.com/supertone-inc/supertonic) Python SDK.

The package is designed to feel like a native Pipecat service:

- import with `from pipecat_supertonic import SupertonicTTSService`
- configure with `SupertonicTTSService.Settings(...)`
- drop directly into an existing Pipecat pipeline

This project is an independent community integration maintained by Archit
Jambhule. It is not affiliated with Supertone or the Supertonic team.

## Install

```bash
pip install pipecat-supertonic
```

Or with `uv`:

```bash
uv add pipecat-supertonic
```

## Supported Voices

`F1`, `F2`, `F3`, `F4`, `F5`, `M1`, `M2`, `M3`, `M4`, `M5`

## Basic Usage

```python
from pipecat_supertonic import SupertonicTTSService

tts = SupertonicTTSService(
    settings=SupertonicTTSService.Settings(
        voice="M1",
        language="en",
        total_steps=5,
        speed=1.05,
    )
)

await tts.warmup()
```

`warmup()` is required before the service is used in a live Pipecat pipeline.
Call it during application startup so Supertonic can download and cache the
model before the first user request arrives.

## Pipecat Pipeline Usage

Use `SupertonicTTSService` anywhere Pipecat expects a TTS processor:

```python
from pipecat.pipeline.pipeline import Pipeline
from pipecat_supertonic import SupertonicTTSService

tts = SupertonicTTSService(
    settings=SupertonicTTSService.Settings(
        voice="M1",
        language="en",
        total_steps=5,
        speed=1.05,
    )
)

await tts.warmup()

pipeline = Pipeline(
    [
        transport.input(),
        stt,
        llm,
        tts,
        transport.output(),
    ]
)
```

## Warmup Contract

This package intentionally does not lazy-load Supertonic during active TTS
requests. If the service is used before `warmup()`, it fails fast with a clear
error telling the caller to warm the service up first.

This avoids first-request cold-start delays and keeps Pipecat TTS frame ordering
stable.

## Foundational Example

`examples/voice-supertonic.py` synthesizes a short utterance and writes the
generated PCM audio to `examples/supertonic-demo.wav`.

Run it from the repository root:

```bash
uv sync --group dev
uv run python examples/voice-supertonic.py
```

## Compatibility

Tested with `pipecat-ai==1.2.0` and `supertonic==1.2.1`.

## Development

```bash
uv sync --group dev
uv run pytest
uv run ruff check .
```
