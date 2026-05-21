import asyncio
from pathlib import Path
import wave

from pipecat.clocks.system_clock import SystemClock
from pipecat.frames.frames import EndFrame
from pipecat.frames.frames import StartFrame
from pipecat.frames.frames import TTSAudioRawFrame
from pipecat.processors.frame_processor import FrameProcessorSetup
from pipecat.utils.asyncio.task_manager import TaskManager
from pipecat.utils.asyncio.task_manager import TaskManagerParams

from pipecat_supertonic import SupertonicTTSService


OUTPUT_PATH = Path(__file__).with_name("supertonic-demo.wav")


async def main() -> None:
    task_manager = TaskManager()
    task_manager.setup(TaskManagerParams(loop=asyncio.get_running_loop()))

    clock = SystemClock()
    clock.start()

    tts = SupertonicTTSService(
        settings=SupertonicTTSService.Settings(
            voice="M1",
            language="en",
            total_steps=5,
            speed=1.05,
        )
    )
    await tts.setup(FrameProcessorSetup(clock=clock, task_manager=task_manager))
    await tts.warmup()
    await tts.start(StartFrame(audio_out_sample_rate=24000))

    audio_chunks: list[bytes] = []
    sample_rate: int | None = None

    try:
        async for frame in tts.run_tts(
            "Hello from pipecat supertonic. This example writes synthesized audio to a wave file.",
            context_id="readme-example",
        ):
            if isinstance(frame, TTSAudioRawFrame):
                audio_chunks.append(frame.audio)
                sample_rate = frame.sample_rate
    finally:
        await tts.stop(EndFrame())
        await tts.cleanup()

    if not audio_chunks or sample_rate is None:
        raise RuntimeError("Supertonic did not produce any audio frames")

    with wave.open(str(OUTPUT_PATH), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(b"".join(audio_chunks))

    print(f"Wrote synthesized audio to {OUTPUT_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
