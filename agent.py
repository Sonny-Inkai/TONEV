from __future__ import annotations

import logging
from dotenv import load_dotenv
from livekit.agents import (
    AutoSubscribe,
    JobContext,
    JobProcess,
    WorkerOptions,
    cli,
)
from livekit.agents.pipeline import VoiceConversionPipeline, ConversionOptions
from livekit.plugins import openai, deepgram, silero, ttsapi


load_dotenv(dotenv_path=".env.local")
logger = logging.getLogger("voice-converter")


def prewarm(proc: JobProcess):
    """Load VAD model in advance"""
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    """Entry point for voice conversion service"""
    logger.info(f"connecting to room {ctx.room.name}")
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    # Wait for the first participant to connect
    participant = await ctx.wait_for_participant()
    logger.info(f"starting voice converter for participant {participant.identity}")

    # Initialize voice conversion pipeline with options
    converter = VoiceConversionPipeline(
        vad=ctx.proc.userdata["vad"],
        stt=deepgram.STT(
            model="nova-2-general"
        ),
        tts=ttsapi.TTS(
            base_url="https://0f1d-2001-ee0-4e4f-33a0-8c94-2dc9-740e-c129.ngrok-free.app",
        ),
        options=ConversionOptions(
            allow_interruptions=True,
            interrupt_speech_duration=0.5,
            min_endpointing_delay=0.3,
            transcription=True,
            plotting=False
        )
    )

    # Set up logging for speech events
    @converter.on("user_started_speaking")
    def on_user_started():
        logger.info("User started speaking")

    @converter.on("user_stopped_speaking")
    def on_user_stopped():
        logger.info("User stopped speaking")

    @converter.on("agent_started_speaking")
    def on_agent_started():
        logger.info("Converting and playing speech")

    @converter.on("agent_stopped_speaking")
    def on_agent_stopped():
        logger.info("Finished playing converted speech")

    @converter.on("speech_converted")
    def on_speech_converted(text: str):
        logger.info(f"Converted speech: {text}")

    # Start the pipeline
    await converter.start(ctx.room, participant)

    # Log ready state
    logger.info("Voice converter started and ready")

if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
        ),
    )