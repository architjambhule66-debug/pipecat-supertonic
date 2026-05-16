# pipecat-supertonic

`pipecat-supertonic` provides a Pipecat-compatible `TTSService` wrapper for the
official [Supertonic](https://github.com/supertone-inc/supertonic) Python SDK.

The package is designed to feel like a native Pipecat service:

- import with `from pipecat_supertonic import SupertonicTTSService`
- configure with `SupertonicTTSService.Settings(...)`
- drop directly into an existing Pipecat pipeline

SUPPORTED VOICES : ('F1', 'F2', 'F3', 'F4', 'F5', 'M1', 'M2', 'M3', 'M4', 'M5')

## Install

```bash
pip install pipecat-supertonic
```

Or with `uv`:

```bash
uv add pipecat-supertonic
```

## Usage

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

## Warmup Contract

This package intentionally does not lazy-load Supertonic during active TTS
requests. If the service is used before `warmup()`, it fails fast with a clear
error telling the caller to warm the service up first.

This avoids first-request cold-start delays and keeps Pipecat TTS frame ordering
stable.

## Example

See `examples/voice-supertonic.py` for a minimal package-level example.

## Development

```bash
uv sync --group dev
uv run pytest
uv run ruff check .
```
