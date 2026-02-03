# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# -------------------------------------------------------------------------
from __future__ import annotations

import argparse
import asyncio
import base64
import logging
import os
import queue
import signal
import sys
import threading
from datetime import datetime
from typing import Optional, TYPE_CHECKING, Union, cast

from dotenv import load_dotenv
import pyaudio

from azure.core.credentials import AzureKeyCredential
from azure.core.credentials_async import AsyncTokenCredential
from azure.identity.aio import AzureCliCredential, DefaultAzureCredential as AsyncDefaultAzureCredential
from azure.identity import DefaultAzureCredential as SyncDefaultAzureCredential
from azure.ai.projects import AIProjectClient

from azure.ai.voicelive.aio import connect
from azure.ai.voicelive.models import (
    AudioEchoCancellation,
    AudioInputTranscriptionOptions,
    AudioNoiseReduction,
    AzureStandardVoice,
    InputAudioFormat,
    Modality,
    OutputAudioFormat,
    RequestSession,
    ServerEventType,
    ServerVad,
)

if TYPE_CHECKING:
    from azure.ai.voicelive.aio import VoiceLiveConnection


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

load_dotenv("./.env", override=True)

LOG_DIR = "logs"
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
logfilename = f"{timestamp}_conversation.log"

logging.basicConfig(
    filename=os.path.join(LOG_DIR, f"{timestamp}_voicelive.log"),
    filemode="w",
    format="%(asctime)s:%(name)s:%(levelname)s:%(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


class AudioProcessor:
    """Handles real-time audio capture and playback for the voice assistant."""

    loop: asyncio.AbstractEventLoop

    class AudioPlaybackPacket:
        """Represents a packet that can be sent to the audio playback queue."""

        def __init__(self, seq_num: int, data: Optional[bytes]):
            self.seq_num = seq_num
            self.data = data

    def __init__(self, connection: "VoiceLiveConnection"):
        self.connection = connection
        self.audio = pyaudio.PyAudio()

        self.format = pyaudio.paInt16
        self.channels = 1
        self.rate = 24000
        self.chunk_size = 1200  # 50ms

        self.input_stream: Optional[pyaudio.Stream] = None
        self.playback_queue: queue.Queue[AudioProcessor.AudioPlaybackPacket] = queue.Queue()
        self.playback_base = 0
        self.next_seq_num = 0
        self.output_stream: Optional[pyaudio.Stream] = None
        self.muted = False
        # Event set by playback callback when speaker is truly idle
        # (queue empty AND no remaining bytes).
        self._playback_idle = threading.Event()
        self._playback_idle.set()  # idle at start

        logger.info("AudioProcessor initialized with 24kHz PCM16 mono audio")

    def start_capture(self) -> None:
        """Start capturing audio from microphone."""

        def _capture_callback(in_data, _frame_count, _time_info, _status_flags):
            if not self.muted:
                audio_base64 = base64.b64encode(in_data).decode("utf-8")
                asyncio.run_coroutine_threadsafe(
                    self.connection.input_audio_buffer.append(audio=audio_base64), self.loop
                )
            return (None, pyaudio.paContinue)

        if self.input_stream:
            return

        self.loop = asyncio.get_event_loop()

        try:
            self.input_stream = self.audio.open(
                format=self.format,
                channels=self.channels,
                rate=self.rate,
                input=True,
                frames_per_buffer=self.chunk_size,
                stream_callback=_capture_callback,
            )
            logger.info("Started audio capture")
        except Exception:
            logger.exception("Failed to start audio capture")
            raise

    def start_playback(self) -> None:
        """Initialize audio playback system."""
        if self.output_stream:
            return

        remaining = bytes()

        def _playback_callback(_in_data, frame_count, _time_info, _status_flags):
            nonlocal remaining
            frame_count *= pyaudio.get_sample_size(pyaudio.paInt16)

            out = remaining[:frame_count]
            remaining = remaining[frame_count:]

            got_real_audio = len(out) > 0

            while len(out) < frame_count:
                try:
                    packet = self.playback_queue.get_nowait()
                except queue.Empty:
                    out = out + bytes(frame_count - len(out))
                    continue
                except Exception:
                    logger.exception("Error in audio playback")
                    raise

                if not packet or not packet.data:
                    logger.info("End of playback queue.")
                    break

                if packet.seq_num < self.playback_base:
                    if len(remaining) > 0:
                        remaining = bytes()
                    continue

                got_real_audio = True
                num_to_take = frame_count - len(out)
                out = out + packet.data[:num_to_take]
                remaining = packet.data[num_to_take:]

            # Track whether speaker is truly idle
            if got_real_audio or len(remaining) > 0:
                self._playback_idle.clear()
            else:
                self._playback_idle.set()

            if len(out) >= frame_count:
                return (out, pyaudio.paContinue)
            return (out, pyaudio.paComplete)

        try:
            self.output_stream = self.audio.open(
                format=self.format,
                channels=self.channels,
                rate=self.rate,
                output=True,
                frames_per_buffer=self.chunk_size,
                stream_callback=_playback_callback,
            )
            logger.info("Audio playback system ready")
        except Exception:
            logger.exception("Failed to initialize audio playback")
            raise

    def _get_and_increase_seq_num(self) -> int:
        seq = self.next_seq_num
        self.next_seq_num += 1
        return seq

    def queue_audio(self, audio_data: Optional[bytes]) -> None:
        self.playback_queue.put(
            AudioProcessor.AudioPlaybackPacket(seq_num=self._get_and_increase_seq_num(), data=audio_data)
        )

    def skip_pending_audio(self) -> None:
        self.playback_base = self._get_and_increase_seq_num()

    async def wait_until_playback_idle(self, timeout: float = 15.0) -> None:
        """Block (async) until the speaker has finished playing all audio."""
        await asyncio.to_thread(self._playback_idle.wait, timeout)

    def shutdown(self) -> None:
        if self.input_stream:
            self.input_stream.stop_stream()
            self.input_stream.close()
            self.input_stream = None

        logger.info("Stopped audio capture")

        if self.output_stream:
            self.skip_pending_audio()
            self.queue_audio(None)
            self.output_stream.stop_stream()
            self.output_stream.close()
            self.output_stream = None

        logger.info("Stopped audio playback")

        if self.audio:
            self.audio.terminate()

        logger.info("Audio processor cleaned up")


class VoiceLiveOrderAssistant:
    """VoiceLive assistant connected to a Foundry agent for order status."""

    def __init__(
        self,
        endpoint: str,
        credential: Union[AzureKeyCredential, AsyncTokenCredential],
        agent_id: str,
        foundry_project_name: str,
        voice: str,
    ):
        self.endpoint = endpoint
        self.credential = credential
        self.agent_id = agent_id
        self.foundry_project_name = foundry_project_name
        self.voice = voice
        self.connection: Optional["VoiceLiveConnection"] = None
        self.audio_processor: Optional[AudioProcessor] = None
        self.session_ready = False
        self.conversation_started = False
        self._active_response = False
        self._response_api_done = False
        self._unmute_task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        try:
            logger.info(
                "Connecting to VoiceLive API with agent %s for project %s",
                self.agent_id,
                self.foundry_project_name,
            )

            async with AsyncDefaultAzureCredential() as token_credential:
                agent_access_token = (await token_credential.get_token("https://ai.azure.com/.default")).token

            async with connect(
                endpoint=self.endpoint,
                credential=self.credential,
                query={
                    "agent-id": self.agent_id,
                    "agent-project-name": self.foundry_project_name,
                    "agent-access-token": agent_access_token,
                },
            ) as connection:
                self.connection = connection

                ap = AudioProcessor(connection)
                self.audio_processor = ap

                await self._setup_session()

                ap.start_playback()

                print("\n" + "=" * 60)
                print("VOICE ASSISTANT READY")
                print("Ask about your order (customer + order number)")
                print("Press Ctrl+C to exit")
                print("=" * 60 + "\n")

                await self._process_events()
        finally:
            if self.audio_processor:
                self.audio_processor.shutdown()

    async def _setup_session(self) -> None:
        logger.info("Setting up voice conversation session...")

        voice_config: Union[AzureStandardVoice, str]
        if self.voice.startswith("en-") or "-" in self.voice:
            voice_config = AzureStandardVoice(name=self.voice)
        else:
            voice_config = self.voice

        turn_detection_config = ServerVad(threshold=0.5, prefix_padding_ms=300, silence_duration_ms=500)

        session_config = RequestSession(
            modalities=[Modality.TEXT, Modality.AUDIO],
            voice=voice_config,
            input_audio_format=InputAudioFormat.PCM16,
            output_audio_format=OutputAudioFormat.PCM16,
            turn_detection=turn_detection_config,
            input_audio_echo_cancellation=AudioEchoCancellation(),
            input_audio_noise_reduction=AudioNoiseReduction(type="azure_deep_noise_suppression"),
            input_audio_transcription=AudioInputTranscriptionOptions(model="azure-speech"),
        )

        assert self.connection is not None
        await self.connection.session.update(session=session_config)

        logger.info("Session configuration sent")

    async def _process_events(self) -> None:
        assert self.connection is not None
        async for event in self.connection:
            await self._handle_event(event)

    async def _handle_event(self, event) -> None:
        ap = self.audio_processor
        conn = self.connection
        assert ap is not None
        assert conn is not None

        if event.type == ServerEventType.SESSION_UPDATED:
            await write_conversation_log(f"SessionID: {event.session.id}")
            await write_conversation_log(f"Model: {event.session.model}")
            await write_conversation_log(f"Voice: {event.session.voice}")
            await write_conversation_log(f"Instructions: {event.session.instructions}")
            await write_conversation_log("")
            self.session_ready = True
            # Start mic capture first so the server-side echo cancellation
            # receives reference audio before the agent starts speaking.
            ap.start_capture()
            if not self.conversation_started:
                self.conversation_started = True
                # Small delay to let echo cancellation calibrate with
                # ambient noise before the agent speaks.
                await asyncio.sleep(0.5)
                try:
                    await conn.response.create()
                except Exception:
                    logger.exception("Failed to send proactive greeting request")

        elif event.type == ServerEventType.CONVERSATION_ITEM_INPUT_AUDIO_TRANSCRIPTION_COMPLETED:
            transcript = event.get("transcript", "")
            print(f"You said:\t{transcript}")
            await write_conversation_log(f"User Input:\t{transcript}")

        elif event.type == ServerEventType.RESPONSE_TEXT_DONE:
            print(f"Agent text:\t{event.get('text', '')}")
            agent_text = event.get("text", "")
            await write_conversation_log(f"Agent Text Response:\t{agent_text}")

        elif event.type == ServerEventType.RESPONSE_AUDIO_TRANSCRIPT_DONE:
            print(f"Agent said:\t{event.get('transcript', '')}")
            agent_transcript = event.get("transcript", "")
            await write_conversation_log(f"Agent Audio Response:\t{agent_transcript}")

        elif event.type == ServerEventType.INPUT_AUDIO_BUFFER_SPEECH_STARTED:
            print("Listening...")
            ap.skip_pending_audio()
            if self._active_response and not self._response_api_done:
                try:
                    await conn.response.cancel()
                except Exception as exc:
                    if "no active response" in str(exc).lower():
                        logger.debug("Cancel ignored - response already completed")
                    else:
                        logger.warning("Cancel failed: %s", exc)

        elif event.type == ServerEventType.INPUT_AUDIO_BUFFER_SPEECH_STOPPED:
            print("Processing...")

        elif event.type == ServerEventType.RESPONSE_CREATED:
            self._active_response = True
            self._response_api_done = False
            # Mute mic as soon as a response starts to prevent echo.
            # Cancel any pending unmute from a previous response.
            if self._unmute_task and not self._unmute_task.done():
                self._unmute_task.cancel()
            ap.muted = True
            logger.info("Mic muted – response started")

        elif event.type == ServerEventType.RESPONSE_AUDIO_DELTA:
            ap.queue_audio(event.delta)

        elif event.type == ServerEventType.RESPONSE_AUDIO_DONE:
            print("Ready for next input...")

        elif event.type == ServerEventType.RESPONSE_DONE:
            self._active_response = False
            self._response_api_done = True

            async def _unmute_after_playback():
                # Wait until the speaker has actually finished playing
                # all buffered audio (queue drained AND remaining bytes consumed).
                await ap.wait_until_playback_idle()
                # Extra safety margin for room reverb / speaker tail
                await asyncio.sleep(0.5)
                # Discard any echo that leaked into the server buffer
                try:
                    await conn.input_audio_buffer.clear()
                except Exception:
                    pass
                ap.muted = False
                logger.info("Mic unmuted – speaker idle")

            if self._unmute_task and not self._unmute_task.done():
                self._unmute_task.cancel()
            self._unmute_task = asyncio.create_task(_unmute_after_playback())

        elif event.type == ServerEventType.ERROR:
            msg = event.error.message
            if "Cancellation failed: no active response" in msg:
                logger.debug("Benign cancellation error: %s", msg)
            else:
                logger.error("VoiceLive error: %s", msg)
                print(f"Error: {msg}")
            if "Conversation already has an active response" in msg:
                self._active_response = True


async def write_conversation_log(message: str) -> None:
    def _write_to_file() -> None:
        with open(os.path.join(LOG_DIR, logfilename), "a", encoding="utf-8") as conversation_log:
            conversation_log.write(message + "\n")

    await asyncio.to_thread(_write_to_file)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="VoiceLive order assistant with Foundry agent",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--api-key",
        help="Azure VoiceLive API key (fallback if not using token credential).",
        type=str,
        default=os.environ.get("AZURE_VOICELIVE_API_KEY"),
    )

    parser.add_argument(
        "--endpoint",
        help="Azure VoiceLive endpoint",
        type=str,
        default=os.environ.get("AZURE_VOICELIVE_ENDPOINT", ""),
    )

    parser.add_argument(
        "--agent-id",
        help="Foundry agent ID to use",
        type=str,
        default=os.environ.get("AZURE_VOICELIVE_AGENT_ID")
        or os.environ.get("AZURE_EXISTING_AGENT_ID", ""),
    )

    parser.add_argument(
        "--project-name",
        help="Foundry project name to use",
        type=str,
        default=os.environ.get("AZURE_VOICELIVE_PROJECT_NAME", ""),
    )

    parser.add_argument(
        "--voice",
        help="Voice to use for the assistant. E.g. alloy, echo, en-US-AvaNeural",
        type=str,
        default=os.environ.get("AZURE_VOICELIVE_VOICE", "en-US-Ava:DragonHDLatestNeural"),
    )

    parser.add_argument(
        "--use-api-key",
        help="Use API key authentication instead of token credentials",
        action="store_true",
    )

    parser.add_argument(
        "--verify-agent",
        help="Run a quick text-only agent check before starting audio",
        action="store_true",
    )


    parser.add_argument("--verbose", help="Enable verbose logging", action="store_true")

    return parser.parse_args()


def normalize_endpoint(endpoint: str) -> str:
    if not endpoint:
        return endpoint
    return endpoint if endpoint.endswith("/") else endpoint + "/"


def verify_agent_access(project_endpoint: str, agent_id: str) -> None:
    if not project_endpoint or not agent_id:
        print("Missing AZURE_EXISTING_AIPROJECT_ENDPOINT or AZURE_EXISTING_AGENT_ID for verification.")
        return

    agent_name, _, agent_version = agent_id.partition(":")
    project_client = AIProjectClient(endpoint=project_endpoint, credential=SyncDefaultAzureCredential())
    if agent_version:
        agent = project_client.agents.get_version(agent_name=agent_name, version=agent_version)
    else:
        agent = project_client.agents.get(agent_name=agent_name)
    print(f"Retrieved agent: {agent.name}")

    openai_client = project_client.get_openai_client()
    response = openai_client.responses.create(
        input=[{"role": "user", "content": "Show my order."}],
        extra_body={"agent": {"name": agent.name, "type": "agent_reference"}},
    )

    print(f"Agent response: {response.output_text}")


def check_audio_devices() -> None:
    try:
        p = pyaudio.PyAudio()
        input_devices = [
            i
            for i in range(p.get_device_count())
            if cast(Union[int, float], p.get_device_info_by_index(i).get("maxInputChannels", 0) or 0) > 0
        ]
        output_devices = [
            i
            for i in range(p.get_device_count())
            if cast(Union[int, float], p.get_device_info_by_index(i).get("maxOutputChannels", 0) or 0) > 0
        ]
        p.terminate()

        if not input_devices:
            print("No audio input devices found. Please check your microphone.")
            sys.exit(1)
        if not output_devices:
            print("No audio output devices found. Please check your speakers.")
            sys.exit(1)
    except Exception as exc:
        print(f"Audio system check failed: {exc}")
        sys.exit(1)


def main() -> None:
    args = parse_arguments()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    endpoint = normalize_endpoint(args.endpoint)
    if not endpoint:
        print("Missing AZURE_VOICELIVE_ENDPOINT")
        sys.exit(1)

    if not args.agent_id:
        print("Missing AZURE_VOICELIVE_AGENT_ID or AZURE_EXISTING_AGENT_ID")
        sys.exit(1)

    if not args.project_name:
        print("Missing AZURE_VOICELIVE_PROJECT_NAME")
        sys.exit(1)

    if args.use_api_key and not args.api_key:
        print("Missing AZURE_VOICELIVE_API_KEY for API key authentication")
        sys.exit(1)

    if args.verify_agent:
        verify_agent_access(
            os.environ.get("AZURE_EXISTING_AIPROJECT_ENDPOINT", ""),
            os.environ.get("AZURE_EXISTING_AGENT_ID", args.agent_id),
        )

    credential: Union[AzureKeyCredential, AsyncTokenCredential]
    if args.use_api_key:
        credential = AzureKeyCredential(args.api_key)
        logger.info("Using API key credential")
    else:
        credential = AzureCliCredential()
        logger.info("Using Azure token credential")

    assistant = VoiceLiveOrderAssistant(
        endpoint=endpoint,
        credential=credential,
        agent_id=args.agent_id,
        foundry_project_name=args.project_name,
        voice=args.voice,
    )

    def signal_handler(_sig, _frame):
        logger.info("Received shutdown signal")
        raise KeyboardInterrupt()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    check_audio_devices()
    print("VoiceLive Order Assistant")
    print("=" * 50)

    try:
        asyncio.run(assistant.start())
    except KeyboardInterrupt:
        print("\nVoice assistant shut down. Goodbye!")
    except Exception as exc:
        print("Fatal Error:", exc)


if __name__ == "__main__":
    main()
