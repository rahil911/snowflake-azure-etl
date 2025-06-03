"""
Gemini 2.0 Live API Client
=========================

Handles real-time voice conversations using Google Gemini 2.0 Live API.
Provides bidirectional streaming audio communication with sub-second latency.

Features:
- Real-time voice input processing
- Streaming audio output synthesis
- Bidirectional conversation handling
- Interruption detection and handling
- Audio format conversion
- Session management
- Error recovery
"""

import asyncio
import json
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional, Callable, AsyncGenerator
from enum import Enum
import base64
import logging

# Google GenAI SDK imports
from google import genai
from google.genai import types

from shared.config.logging_config import setup_logging
from shared.utils.validation import validate_input


class AudioFormat(Enum):
    """Supported audio formats."""
    PCM16 = "pcm16"
    OPUS = "opus"
    WEBM = "webm"
    MP3 = "mp3"


class SessionState(Enum):
    """Live API session states."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    STREAMING = "streaming"
    ERROR = "error"


class LiveAPIMessage:
    """Represents a Live API message."""
    
    def __init__(
        self,
        message_type: str,
        data: Dict[str, Any] = None,
        audio_data: bytes = None,
        timestamp: datetime = None
    ):
        self.message_type = message_type
        self.data = data or {}
        self.audio_data = audio_data
        self.timestamp = timestamp or datetime.utcnow()
        self.message_id = str(uuid.uuid4())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "type": self.message_type,
            "id": self.message_id,
            "timestamp": self.timestamp.isoformat(),
            "data": self.data,
            "has_audio": self.audio_data is not None
        }


class LiveAPISession:
    """Manages a Live API session."""
    
    def __init__(
        self,
        session_id: str,
        live_session: Any,  # Google GenAI Live session
        audio_format: AudioFormat = AudioFormat.PCM16
    ):
        self.session_id = session_id
        self.live_session = live_session
        self.audio_format = audio_format
        self.state = SessionState.CONNECTED
        self.created_at = datetime.utcnow()
        self.last_activity = datetime.utcnow()
        
        # Audio buffers
        self.input_buffer = bytearray()
        self.output_buffer = bytearray()
        
        # Conversation state
        self.is_user_speaking = False
        self.is_assistant_speaking = False
        self.turn_detection_threshold = 0.5  # seconds of silence
        
        # Statistics
        self.messages_sent = 0
        self.messages_received = 0
        self.audio_bytes_processed = 0
    
    def update_activity(self) -> None:
        """Update last activity timestamp."""
        self.last_activity = datetime.utcnow()
    
    def is_active(self) -> bool:
        """Check if session is still active."""
        return self.state in [SessionState.CONNECTED, SessionState.STREAMING]
    
    def get_duration(self) -> float:
        """Get session duration in seconds."""
        return (datetime.utcnow() - self.created_at).total_seconds()


class GeminiLiveAPIClient:
    """
    Client for Google Gemini 2.0 Live API with real-time voice conversations.
    """
    
    def __init__(self, settings):
        self.settings = settings
        self.logger = setup_logging("GeminiLiveAPIClient")
        
        # API configuration
        self.api_key = settings.google_api_key
        self.model_name = "gemini-2.0-flash-live-preview-04-09"  # Live API model
        
        # Initialize Google GenAI client
        self.client = genai.Client(api_key=self.api_key)
        
        # Session management
        self.active_sessions: Dict[str, Any] = {}
        self.max_sessions = 100
        
        # Audio configuration
        self.sample_rate = 16000  # 16kHz input
        self.output_sample_rate = 24000  # 24kHz output
        self.channels = 1  # Mono
        self.bit_depth = 16  # 16-bit
        self.chunk_size = 1024  # Audio chunk size
        
        # Event handlers
        self.on_audio_received: Optional[Callable] = None
        self.on_text_received: Optional[Callable] = None
        self.on_session_started: Optional[Callable] = None
        self.on_session_ended: Optional[Callable] = None
        self.on_error: Optional[Callable] = None
        
        # Statistics
        self.total_sessions = 0
        self.total_audio_processed = 0
        
        self.logger.info("GeminiLiveAPIClient initialized")
    
    async def initialize(self) -> None:
        """Initialize the Live API client."""
        try:
            # Validate API key
            if not self.api_key:
                raise ValueError("Google API key not provided")
            
            # Start session cleanup task
            asyncio.create_task(self._cleanup_inactive_sessions())
            
            self.logger.info("GeminiLiveAPIClient initialization completed")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize GeminiLiveAPIClient: {str(e)}")
            raise
    
    async def create_session(
        self,
        session_id: Optional[str] = None,
        voice_name: str = "Aoede",
        system_instruction: Optional[str] = None,
        response_modalities: List[str] = None
    ) -> str:
        """
        Create a new Live API session.
        
        Args:
            session_id: Optional session ID (generated if not provided)
            voice_name: Voice for audio responses
            system_instruction: Optional system instruction for the model
            response_modalities: Response modalities (TEXT, AUDIO)
            
        Returns:
            Session ID
        """
        try:
            if session_id is None:
                session_id = str(uuid.uuid4())
            
            # Check session limit
            if len(self.active_sessions) >= self.max_sessions:
                await self._cleanup_oldest_sessions(10)
            
            # Configure Live API session
            config = types.LiveConnectConfig(
                response_modalities=response_modalities or ["AUDIO", "TEXT"],
                speech_config=types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(
                            voice_name=voice_name
                        )
                    )
                ),
                system_instruction=system_instruction
            )
            
            # Create Live API session
            session = await self.client.aio.live.connect(
                model=self.model_name,
                config=config
            )
            
            # Store session
            self.active_sessions[session_id] = {
                "session": session,
                "created_at": datetime.utcnow(),
                "last_activity": datetime.utcnow(),
                "config": config
            }
            
            self.total_sessions += 1
            
            if self.on_session_started:
                await self.on_session_started(session_id)
            
            self.logger.info(f"Created Live API session: {session_id}")
            return session_id
            
        except Exception as e:
            self.logger.error(f"Error creating session: {str(e)}")
            raise
    
    async def send_audio(
        self,
        session_id: str,
        audio_data: bytes,
        mime_type: str = "audio/pcm;rate=16000"
    ) -> None:
        """
        Send audio data to a session.
        
        Args:
            session_id: Session ID
            audio_data: Raw audio bytes
            mime_type: MIME type of audio data
        """
        try:
            session_info = self.active_sessions.get(session_id)
            if not session_info:
                raise ValueError(f"Session {session_id} not found")
            
            session = session_info["session"]
            
            # Send audio using Live API
            await session.send_realtime_input(
                audio=types.Blob(data=audio_data, mime_type=mime_type)
            )
            
            # Update activity
            session_info["last_activity"] = datetime.utcnow()
            self.total_audio_processed += len(audio_data)
            
            self.logger.debug(f"Sent audio chunk to session {session_id}: {len(audio_data)} bytes")
            
        except Exception as e:
            self.logger.error(f"Error sending audio: {str(e)}")
            raise
    
    async def send_text(
        self,
        session_id: str,
        text: str,
        interrupt_audio: bool = False
    ) -> None:
        """
        Send text input to a session.
        
        Args:
            session_id: Session ID
            text: Text message
            interrupt_audio: Whether to interrupt ongoing audio output
        """
        try:
            session_info = self.active_sessions.get(session_id)
            if not session_info:
                raise ValueError(f"Session {session_id} not found")
            
            session = session_info["session"]
            
            # Send text using Live API
            await session.send_text(text)
            
            # Update activity
            session_info["last_activity"] = datetime.utcnow()
            
            self.logger.debug(f"Sent text to session {session_id}: {text[:50]}...")
            
        except Exception as e:
            self.logger.error(f"Error sending text: {str(e)}")
            raise
    
    async def interrupt_audio(self, session_id: str) -> None:
        """
        Interrupt ongoing audio output in a session.
        
        Args:
            session_id: Session ID
        """
        try:
            session_info = self.active_sessions.get(session_id)
            if not session_info:
                return
            
            session = session_info["session"]
            
            # Interrupt using Live API
            await session.interrupt()
            
            # Update activity
            session_info["last_activity"] = datetime.utcnow()
            
            self.logger.debug(f"Interrupted audio in session {session_id}")
            
        except Exception as e:
            self.logger.error(f"Error interrupting audio: {str(e)}")
    
    async def close_session(self, session_id: str) -> None:
        """
        Close a Live API session.
        
        Args:
            session_id: Session ID to close
        """
        try:
            session_info = self.active_sessions.get(session_id)
            if not session_info:
                self.logger.warning(f"Attempted to close non-existent session: {session_id}")
                return
            
            session = session_info["session"]
            
            # Close the Live API session
            try:
                await session.close()
            except Exception as e:
                self.logger.warning(f"Error closing Live API session: {str(e)}")
            
            # Remove from active sessions
            del self.active_sessions[session_id]
            
            # Trigger event
            if self.on_session_ended:
                await self.on_session_ended(session_id)
            
            self.logger.info(f"Closed Live API session: {session_id}")
            
        except Exception as e:
            self.logger.error(f"Error closing session: {str(e)}")
    
    async def _send_session_config(
        self,
        session: LiveAPISession,
        system_instruction: Optional[str] = None
    ) -> None:
        """Send initial session configuration."""
        try:
            config = {
                "type": "session_config",
                "id": str(uuid.uuid4()),
                "model": self.model_name,
                "audio_config": {
                    "input_format": session.audio_format.value,
                    "output_format": session.audio_format.value,
                    "sample_rate": self.sample_rate,
                    "channels": self.channels
                },
                "generation_config": {
                    "temperature": 0.7,
                    "max_output_tokens": 1000,
                    "response_modalities": ["text", "audio"]
                }
            }
            
            if system_instruction:
                config["system_instruction"] = system_instruction
            
            await session.websocket.send(json.dumps(config))
            self.logger.debug(f"Sent session config for {session.session_id}")
            
        except Exception as e:
            self.logger.error(f"Error sending session config: {str(e)}")
            raise
    
    async def _handle_session(self, session: LiveAPISession) -> None:
        """Handle messages for a session."""
        try:
            session.state = SessionState.STREAMING
            
            async for raw_message in session.websocket:
                try:
                    message = json.loads(raw_message)
                    await self._process_message(session, message)
                    session.messages_received += 1
                    session.update_activity()
                    
                except json.JSONDecodeError as e:
                    self.logger.error(f"Invalid JSON message: {str(e)}")
                except Exception as e:
                    self.logger.error(f"Error processing message: {str(e)}")
            
        except websockets.exceptions.ConnectionClosed:
            self.logger.info(f"Session {session.session_id} connection closed")
        except Exception as e:
            self.logger.error(f"Session handler error: {str(e)}")
            session.state = SessionState.ERROR
            
            if self.on_error:
                await self.on_error(session.session_id, str(e))
        
        finally:
            # Clean up session
            if session.session_id in self.active_sessions:
                del self.active_sessions[session.session_id]
    
    async def _process_message(
        self,
        session: LiveAPISession,
        message: Dict[str, Any]
    ) -> None:
        """Process a message from the Live API."""
        try:
            message_type = message.get("type")
            
            if message_type == "audio_output":
                await self._handle_audio_output(session, message)
            
            elif message_type == "text_output":
                await self._handle_text_output(session, message)
            
            elif message_type == "turn_detection":
                await self._handle_turn_detection(session, message)
            
            elif message_type == "session_update":
                await self._handle_session_update(session, message)
            
            elif message_type == "error":
                await self._handle_error(session, message)
            
            else:
                self.logger.debug(f"Unknown message type: {message_type}")
                
        except Exception as e:
            self.logger.error(f"Error processing message: {str(e)}")
    
    async def _handle_audio_output(
        self,
        session: LiveAPISession,
        message: Dict[str, Any]
    ) -> None:
        """Handle audio output from the model."""
        try:
            audio_data_b64 = message.get("audio_data")
            if not audio_data_b64:
                return
            
            # Decode audio data
            audio_data = base64.b64decode(audio_data_b64)
            session.output_buffer.extend(audio_data)
            
            # Update speaking state
            session.is_assistant_speaking = True
            
            # Trigger callback
            if self.on_audio_received:
                await self.on_audio_received(
                    session.session_id,
                    audio_data,
                    message.get("is_final", False)
                )
            
            self.logger.debug(f"Received audio output: {len(audio_data)} bytes")
            
        except Exception as e:
            self.logger.error(f"Error handling audio output: {str(e)}")
    
    async def _handle_text_output(
        self,
        session: LiveAPISession,
        message: Dict[str, Any]
    ) -> None:
        """Handle text output from the model."""
        try:
            text = message.get("text")
            if not text:
                return
            
            # Trigger callback
            if self.on_text_received:
                await self.on_text_received(
                    session.session_id,
                    text,
                    message.get("is_final", False)
                )
            
            self.logger.debug(f"Received text output: {text[:50]}...")
            
        except Exception as e:
            self.logger.error(f"Error handling text output: {str(e)}")
    
    async def _handle_turn_detection(
        self,
        session: LiveAPISession,
        message: Dict[str, Any]
    ) -> None:
        """Handle turn detection events."""
        try:
            event_type = message.get("event")
            
            if event_type == "speech_start":
                session.is_user_speaking = True
                self.logger.debug(f"User started speaking in session {session.session_id}")
            
            elif event_type == "speech_end":
                session.is_user_speaking = False
                self.logger.debug(f"User stopped speaking in session {session.session_id}")
            
            elif event_type == "audio_end":
                session.is_assistant_speaking = False
                self.logger.debug(f"Assistant finished speaking in session {session.session_id}")
                
        except Exception as e:
            self.logger.error(f"Error handling turn detection: {str(e)}")
    
    async def _handle_session_update(
        self,
        session: LiveAPISession,
        message: Dict[str, Any]
    ) -> None:
        """Handle session update messages."""
        try:
            update_type = message.get("update_type")
            
            if update_type == "session_ready":
                session.state = SessionState.CONNECTED
                self.logger.info(f"Session {session.session_id} is ready")
            
            elif update_type == "session_expired":
                session.state = SessionState.DISCONNECTED
                self.logger.warning(f"Session {session.session_id} expired")
                
        except Exception as e:
            self.logger.error(f"Error handling session update: {str(e)}")
    
    async def _handle_error(
        self,
        session: LiveAPISession,
        message: Dict[str, Any]
    ) -> None:
        """Handle error messages."""
        try:
            error_code = message.get("code")
            error_message = message.get("message", "Unknown error")
            
            session.state = SessionState.ERROR
            
            self.logger.error(f"Live API error in session {session.session_id}: {error_code} - {error_message}")
            
            if self.on_error:
                await self.on_error(session.session_id, f"{error_code}: {error_message}")
                
        except Exception as e:
            self.logger.error(f"Error handling error message: {str(e)}")
    
    async def _cleanup_inactive_sessions(self) -> None:
        """Background task to cleanup inactive sessions."""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute
                
                inactive_sessions = []
                current_time = datetime.utcnow()
                
                for session_id, session in self.active_sessions.items():
                    # Close sessions inactive for more than 30 minutes
                    if (current_time - session.last_activity).total_seconds() > 1800:
                        inactive_sessions.append(session_id)
                
                for session_id in inactive_sessions:
                    await self.close_session(session_id)
                
                if inactive_sessions:
                    self.logger.info(f"Cleaned up {len(inactive_sessions)} inactive sessions")
                
            except Exception as e:
                self.logger.error(f"Error in cleanup task: {str(e)}")
                await asyncio.sleep(10)
    
    async def _cleanup_oldest_sessions(self, count: int) -> None:
        """Remove the oldest sessions to make room for new ones."""
        try:
            # Sort by creation time
            sessions_by_age = sorted(
                self.active_sessions.items(),
                key=lambda x: x[1].created_at
            )
            
            removed_count = 0
            for session_id, session in sessions_by_age:
                if removed_count >= count:
                    break
                
                await self.close_session(session_id)
                removed_count += 1
            
            self.logger.info(f"Cleaned up {removed_count} oldest sessions")
            
        except Exception as e:
            self.logger.error(f"Error cleaning up oldest sessions: {str(e)}")
    
    def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a session."""
        try:
            session = self.active_sessions.get(session_id)
            if not session:
                return None
            
            return {
                "session_id": session.session_id,
                "state": session.state.value,
                "audio_format": session.audio_format.value,
                "created_at": session.created_at.isoformat(),
                "last_activity": session.last_activity.isoformat(),
                "duration": session.get_duration(),
                "is_user_speaking": session.is_user_speaking,
                "is_assistant_speaking": session.is_assistant_speaking,
                "messages_sent": session.messages_sent,
                "messages_received": session.messages_received,
                "audio_bytes_processed": session.audio_bytes_processed
            }
            
        except Exception as e:
            self.logger.error(f"Error getting session info: {str(e)}")
            return None
    
    def get_client_stats(self) -> Dict[str, Any]:
        """Get client statistics."""
        return {
            "active_sessions": len(self.active_sessions),
            "total_sessions": self.total_sessions,
            "total_audio_processed": self.total_audio_processed,
            "max_sessions": self.max_sessions,
            "session_states": {
                state.value: len([s for s in self.active_sessions.values() if s.state == state])
                for state in SessionState
            }
        }
    
    async def get_health(self) -> Dict[str, Any]:
        """Get health status of the Live API client."""
        try:
            return {
                "status": "healthy",
                "active_sessions": len(self.active_sessions),
                "total_sessions": self.total_sessions,
                "websocket_url": self.websocket_url,
                "model": self.model_name
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    async def shutdown(self) -> None:
        """Shutdown the Live API client."""
        try:
            self.logger.info("Shutting down GeminiLiveAPIClient...")
            
            # Close all active sessions
            session_ids = list(self.active_sessions.keys())
            for session_id in session_ids:
                await self.close_session(session_id)
            
            self.logger.info("GeminiLiveAPIClient shutdown completed")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {str(e)}")
            raise 