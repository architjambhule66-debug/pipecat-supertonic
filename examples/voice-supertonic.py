import asyncio
from pipecat_supertonic import SupertonicTTSService


async def main() -> None:
    tts = SupertonicTTSService(
        settings=SupertonicTTSService.Settings(
            voice="M1",
            language="en",
            total_steps=5,
            speed=1.05,
        )
    )
    await tts.warmup()
    print(tts)


if __name__ == "__main__":
    asyncio.run(main())
