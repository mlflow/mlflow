import logging
import os

from livekit.agents import JobContext, JobProcess, WorkerOptions, cli
from livekit.agents.telemetry import set_tracer_provider
from livekit.agents.voice import Agent, AgentSession
from livekit.plugins import openai, silero
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("voice-agent")


def configure_mlflow_tracing():
    """Configure OpenTelemetry to send traces to MLflow."""
    if not os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"):
        logger.warning("OTEL_EXPORTER_OTLP_ENDPOINT not set, tracing disabled")
        return None

    service_name = os.getenv("OTEL_SERVICE_NAME", "livekit-voice-agent")
    resource = Resource.create({SERVICE_NAME: service_name})

    provider = TracerProvider(resource=resource)
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))

    trace.set_tracer_provider(provider)
    set_tracer_provider(provider)

    logger.info("MLflow tracing configured successfully!")
    return provider


async def entrypoint(ctx: JobContext) -> None:
    """Main entrypoint for the agent."""
    logger.info(f"Agent starting for room: {ctx.room.name}")

    # Connect to the room
    await ctx.connect()

    # Create the voice agent with all components
    agent = Agent(
        instructions="""You are a helpful voice assistant. Keep your responses
        concise and conversational since you're speaking out loud.
        Be friendly and helpful.""",
        vad=silero.VAD.load(),  # Voice Activity Detection
        stt=openai.STT(),  # Speech-to-Text (Whisper)
        llm=openai.LLM(model="gpt-4o-mini"),  # Language Model
        tts=openai.TTS(voice="alloy"),  # Text-to-Speech
    )

    # Create and start the agent session
    session = AgentSession()
    await session.start(agent, room=ctx.room)

    logger.info("Agent session started! Ready for conversation.")


def prewarm(proc: JobProcess) -> None:
    """Prewarm function to load models before handling requests."""
    # Configure tracing before anything else
    configure_mlflow_tracing()

    # Preload Silero VAD model for faster startup
    proc.userdata["vad"] = silero.VAD.load()
    logger.info("Prewarmed VAD model")


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
        )
    )
