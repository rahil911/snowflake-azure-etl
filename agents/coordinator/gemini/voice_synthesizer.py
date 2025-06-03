"""
Voice Synthesizer
================

Handles voice response generation for Gemini 2.0 integration.
Converts text responses to natural-sounding speech with customizable voices.

Features:
- Text-to-speech synthesis
- Voice customization (tone, speed, pitch)
- Audio format conversion
- SSML support for enhanced speech
- Real-time streaming synthesis
- Voice cloning capabilities (future)
- Emotional tone adaptation
"""

import asyncio
import base64
import json
import io
from datetime import datetime
from typing import Dict, Any, List, Optional, Union, Callable
from enum import Enum
import logging

from shared.config.logging_config import setup_logging
from shared.utils.validation import validate_input


class VoiceProfile(Enum):
    """Available voice profiles."""
    PROFESSIONAL = "professional"
    FRIENDLY = "friendly"
    CASUAL = "casual"
    TECHNICAL = "technical"
    ENTHUSIASTIC = "enthusiastic"
    CALM = "calm"
    AUTHORITATIVE = "authoritative"


class SpeechRate(Enum):
    """Speech rate options."""
    VERY_SLOW = "very_slow"
    SLOW = "slow"
    NORMAL = "normal"
    FAST = "fast"
    VERY_FAST = "very_fast"


class EmotionalTone(Enum):
    """Emotional tone options."""
    NEUTRAL = "neutral"
    HAPPY = "happy"
    SAD = "sad"
    EXCITED = "excited"
    CONCERNED = "concerned"
    CONFIDENT = "confident"
    APOLOGETIC = "apologetic"


class AudioFormat(Enum):
    """Output audio formats."""
    WAV = "wav"
    MP3 = "mp3"
    OPUS = "opus"
    WEBM = "webm"
    PCM16 = "pcm16"


class VoiceSettings:
    """Voice synthesis settings."""
    
    def __init__(
        self,
        voice_profile: VoiceProfile = VoiceProfile.PROFESSIONAL,
        speech_rate: SpeechRate = SpeechRate.NORMAL,
        emotional_tone: EmotionalTone = EmotionalTone.NEUTRAL,
        pitch: float = 0.0,  # -20.0 to 20.0 semitones
        volume: float = 0.0,  # -96.0 to 16.0 dB
        audio_format: AudioFormat = AudioFormat.WAV,
        sample_rate: int = 16000,
        use_ssml: bool = False
    ):
        self.voice_profile = voice_profile
        self.speech_rate = speech_rate
        self.emotional_tone = emotional_tone
        self.pitch = pitch
        self.volume = volume
        self.audio_format = audio_format
        self.sample_rate = sample_rate
        self.use_ssml = use_ssml
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "voice_profile": self.voice_profile.value,
            "speech_rate": self.speech_rate.value,
            "emotional_tone": self.emotional_tone.value,
            "pitch": self.pitch,
            "volume": self.volume,
            "audio_format": self.audio_format.value,
            "sample_rate": self.sample_rate,
            "use_ssml": self.use_ssml
        }


class SynthesisRequest:
    """Voice synthesis request."""
    
    def __init__(
        self,
        text: str,
        settings: VoiceSettings,
        request_id: Optional[str] = None,
        metadata: Dict[str, Any] = None
    ):
        self.text = text
        self.settings = settings
        self.request_id = request_id or f"req_{int(datetime.utcnow().timestamp())}"
        self.metadata = metadata or {}
        self.created_at = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "request_id": self.request_id,
            "text": self.text,
            "settings": self.settings.to_dict(),
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat()
        }


class SynthesisResult:
    """Voice synthesis result."""
    
    def __init__(
        self,
        request_id: str,
        audio_data: bytes,
        format_info: Dict[str, Any],
        duration_seconds: float,
        success: bool = True,
        error_message: Optional[str] = None,
        metadata: Dict[str, Any] = None
    ):
        self.request_id = request_id
        self.audio_data = audio_data
        self.format_info = format_info
        self.duration_seconds = duration_seconds
        self.success = success
        self.error_message = error_message
        self.metadata = metadata or {}
        self.completed_at = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "request_id": self.request_id,
            "format_info": self.format_info,
            "duration_seconds": self.duration_seconds,
            "success": self.success,
            "error_message": self.error_message,
            "metadata": self.metadata,
            "completed_at": self.completed_at.isoformat(),
            "has_audio_data": self.audio_data is not None
        }


class VoiceSynthesizer:
    """
    Voice response generation for Gemini 2.0 integration.
    """
    
    def __init__(self, settings):
        self.settings = settings
        self.logger = setup_logging("VoiceSynthesizer")
        
        # Synthesis configuration
        self.default_voice_settings = VoiceSettings()
        
        # Active synthesis requests
        self.active_requests: Dict[str, SynthesisRequest] = {}
        
        # Voice profiles configuration
        self.voice_profiles = self._load_voice_profiles()
        
        # Audio processing settings
        self.max_text_length = 5000  # Max characters for single synthesis
        self.chunk_size = 500  # Characters per chunk for long text
        
        # Event handlers
        self.on_synthesis_started: Optional[Callable] = None
        self.on_synthesis_completed: Optional[Callable] = None
        self.on_synthesis_error: Optional[Callable] = None
        self.on_audio_chunk: Optional[Callable] = None
        
        # Statistics
        self.total_requests = 0
        self.successful_syntheses = 0
        self.failed_syntheses = 0
        self.total_audio_duration = 0.0
        
        self.logger.info("VoiceSynthesizer initialized")
    
    async def initialize(self) -> None:
        """Initialize the voice synthesizer."""
        try:
            # Validate text-to-speech dependencies
            # In a real implementation, this would check for TTS engines
            
            self.logger.info("VoiceSynthesizer initialization completed")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize VoiceSynthesizer: {str(e)}")
            raise
    
    def _load_voice_profiles(self) -> Dict[VoiceProfile, Dict[str, Any]]:
        """Load voice profile configurations."""
        return {
            VoiceProfile.PROFESSIONAL: {
                "base_pitch": 0.0,
                "pitch_variance": 2.0,
                "speaking_rate": 1.0,
                "emphasis_strength": 0.7,
                "pause_duration": 1.2
            },
            VoiceProfile.FRIENDLY: {
                "base_pitch": 2.0,
                "pitch_variance": 4.0,
                "speaking_rate": 1.1,
                "emphasis_strength": 0.8,
                "pause_duration": 1.0
            },
            VoiceProfile.CASUAL: {
                "base_pitch": 1.0,
                "pitch_variance": 3.0,
                "speaking_rate": 1.2,
                "emphasis_strength": 0.6,
                "pause_duration": 0.8
            },
            VoiceProfile.TECHNICAL: {
                "base_pitch": -1.0,
                "pitch_variance": 1.5,
                "speaking_rate": 0.9,
                "emphasis_strength": 0.9,
                "pause_duration": 1.4
            },
            VoiceProfile.ENTHUSIASTIC: {
                "base_pitch": 3.0,
                "pitch_variance": 5.0,
                "speaking_rate": 1.3,
                "emphasis_strength": 1.0,
                "pause_duration": 0.6
            },
            VoiceProfile.CALM: {
                "base_pitch": -2.0,
                "pitch_variance": 1.0,
                "speaking_rate": 0.8,
                "emphasis_strength": 0.5,
                "pause_duration": 1.6
            },
            VoiceProfile.AUTHORITATIVE: {
                "base_pitch": -3.0,
                "pitch_variance": 1.5,
                "speaking_rate": 0.9,
                "emphasis_strength": 0.8,
                "pause_duration": 1.3
            }
        }
    
    async def synthesize_speech(
        self,
        text: str,
        voice_settings: Optional[VoiceSettings] = None,
        request_id: Optional[str] = None,
        metadata: Dict[str, Any] = None
    ) -> SynthesisResult:
        """
        Synthesize speech from text.
        
        Args:
            text: Text to synthesize
            voice_settings: Voice synthesis settings
            request_id: Optional request ID
            metadata: Optional metadata
            
        Returns:
            SynthesisResult with audio data
        """
        try:
            self.total_requests += 1
            
            # Use default settings if not provided
            if voice_settings is None:
                voice_settings = self.default_voice_settings
            
            # Create synthesis request
            request = SynthesisRequest(
                text=text,
                settings=voice_settings,
                request_id=request_id,
                metadata=metadata
            )
            
            # Store active request
            self.active_requests[request.request_id] = request
            
            # Trigger start event
            if self.on_synthesis_started:
                await self.on_synthesis_started(request.request_id)
            
            # Validate text length
            if len(text) > self.max_text_length:
                return await self._synthesize_long_text(request)
            else:
                return await self._synthesize_single_chunk(request)
                
        except Exception as e:
            self.failed_syntheses += 1
            self.logger.error(f"Error in speech synthesis: {str(e)}")
            
            # Trigger error event
            if self.on_synthesis_error:
                await self.on_synthesis_error(request_id or "unknown", str(e))
            
            return SynthesisResult(
                request_id=request_id or "unknown",
                audio_data=b"",
                format_info={},
                duration_seconds=0.0,
                success=False,
                error_message=str(e)
            )
        
        finally:
            # Clean up active request
            if request_id and request_id in self.active_requests:
                del self.active_requests[request_id]
    
    async def _synthesize_single_chunk(self, request: SynthesisRequest) -> SynthesisResult:
        """Synthesize a single chunk of text."""
        try:
            # Prepare text for synthesis
            synthesis_text = await self._prepare_text_for_synthesis(
                request.text, 
                request.settings
            )
            
            # Simulate audio synthesis (in real implementation, would use TTS engine)
            audio_data, duration = await self._generate_audio(synthesis_text, request.settings)
            
            # Create format info
            format_info = {
                "format": request.settings.audio_format.value,
                "sample_rate": request.settings.sample_rate,
                "channels": 1,
                "bit_depth": 16
            }
            
            # Update statistics
            self.successful_syntheses += 1
            self.total_audio_duration += duration
            
            # Create result
            result = SynthesisResult(
                request_id=request.request_id,
                audio_data=audio_data,
                format_info=format_info,
                duration_seconds=duration,
                success=True,
                metadata=request.metadata
            )
            
            # Trigger completion event
            if self.on_synthesis_completed:
                await self.on_synthesis_completed(request.request_id, result)
            
            self.logger.debug(f"Synthesized speech: {request.request_id} ({duration:.2f}s)")
            return result
            
        except Exception as e:
            self.logger.error(f"Error synthesizing single chunk: {str(e)}")
            raise
    
    async def _synthesize_long_text(self, request: SynthesisRequest) -> SynthesisResult:
        """Synthesize long text by chunking."""
        try:
            # Split text into chunks
            chunks = self._split_text_into_chunks(request.text, self.chunk_size)
            
            # Synthesize each chunk
            audio_chunks = []
            total_duration = 0.0
            
            for i, chunk in enumerate(chunks):
                chunk_request = SynthesisRequest(
                    text=chunk,
                    settings=request.settings,
                    request_id=f"{request.request_id}_chunk_{i}",
                    metadata={"chunk_index": i, "total_chunks": len(chunks)}
                )
                
                chunk_result = await self._synthesize_single_chunk(chunk_request)
                
                if chunk_result.success:
                    audio_chunks.append(chunk_result.audio_data)
                    total_duration += chunk_result.duration_seconds
                    
                    # Trigger chunk event
                    if self.on_audio_chunk:
                        await self.on_audio_chunk(
                            request.request_id, 
                            chunk_result.audio_data, 
                            i, 
                            len(chunks)
                        )
                else:
                    raise ValueError(f"Failed to synthesize chunk {i}: {chunk_result.error_message}")
            
            # Combine audio chunks
            combined_audio = await self._combine_audio_chunks(audio_chunks, request.settings)
            
            # Create format info
            format_info = {
                "format": request.settings.audio_format.value,
                "sample_rate": request.settings.sample_rate,
                "channels": 1,
                "bit_depth": 16,
                "chunks_combined": len(chunks)
            }
            
            # Update statistics
            self.successful_syntheses += 1
            self.total_audio_duration += total_duration
            
            # Create result
            result = SynthesisResult(
                request_id=request.request_id,
                audio_data=combined_audio,
                format_info=format_info,
                duration_seconds=total_duration,
                success=True,
                metadata={**request.metadata, "chunked_synthesis": True}
            )
            
            # Trigger completion event
            if self.on_synthesis_completed:
                await self.on_synthesis_completed(request.request_id, result)
            
            self.logger.debug(f"Synthesized long speech: {request.request_id} ({total_duration:.2f}s, {len(chunks)} chunks)")
            return result
            
        except Exception as e:
            self.logger.error(f"Error synthesizing long text: {str(e)}")
            raise
    
    async def _prepare_text_for_synthesis(
        self, 
        text: str, 
        settings: VoiceSettings
    ) -> str:
        """Prepare text for synthesis (SSML, normalization, etc.)."""
        try:
            # Clean and normalize text
            cleaned_text = text.strip()
            
            # Apply SSML if enabled
            if settings.use_ssml:
                cleaned_text = await self._apply_ssml_markup(cleaned_text, settings)
            
            # Apply voice profile adjustments
            cleaned_text = await self._apply_voice_profile_adjustments(cleaned_text, settings)
            
            return cleaned_text
            
        except Exception as e:
            self.logger.error(f"Error preparing text for synthesis: {str(e)}")
            return text  # Return original text as fallback
    
    async def _apply_ssml_markup(self, text: str, settings: VoiceSettings) -> str:
        """Apply SSML markup to enhance speech synthesis."""
        try:
            # Basic SSML wrapper
            ssml_text = f'<speak version="1.0" xml:lang="en-US">'
            
            # Apply speech rate
            rate_map = {
                SpeechRate.VERY_SLOW: "x-slow",
                SpeechRate.SLOW: "slow",
                SpeechRate.NORMAL: "medium",
                SpeechRate.FAST: "fast",
                SpeechRate.VERY_FAST: "x-fast"
            }
            rate = rate_map.get(settings.speech_rate, "medium")
            
            # Apply pitch
            pitch_value = f"{settings.pitch:+.1f}st" if settings.pitch != 0 else "medium"
            
            # Apply emotional tone through prosody
            ssml_text += f'<prosody rate="{rate}" pitch="{pitch_value}">'
            
            # Add emotional markup based on tone
            if settings.emotional_tone == EmotionalTone.HAPPY:
                ssml_text += '<emphasis level="moderate">'
            elif settings.emotional_tone == EmotionalTone.EXCITED:
                ssml_text += '<emphasis level="strong">'
            elif settings.emotional_tone == EmotionalTone.CONCERNED:
                ssml_text += '<prosody rate="slow">'
            
            # Add the actual text
            ssml_text += text
            
            # Close emotional markup
            if settings.emotional_tone in [EmotionalTone.HAPPY, EmotionalTone.EXCITED]:
                ssml_text += '</emphasis>'
            elif settings.emotional_tone == EmotionalTone.CONCERNED:
                ssml_text += '</prosody>'
            
            # Close prosody and speak tags
            ssml_text += '</prosody></speak>'
            
            return ssml_text
            
        except Exception as e:
            self.logger.error(f"Error applying SSML markup: {str(e)}")
            return text
    
    async def _apply_voice_profile_adjustments(self, text: str, settings: VoiceSettings) -> str:
        """Apply voice profile-specific text adjustments."""
        try:
            profile_config = self.voice_profiles.get(settings.voice_profile, {})
            
            # Add pauses for technical profile
            if settings.voice_profile == VoiceProfile.TECHNICAL:
                # Add pauses after technical terms or complex phrases
                text = text.replace(", ", ",<break time=\"500ms\"/> ")
                text = text.replace(". ", ".<break time=\"800ms\"/> ")
            
            # Add emphasis for enthusiastic profile
            elif settings.voice_profile == VoiceProfile.ENTHUSIASTIC:
                # Emphasize certain words
                text = text.replace("amazing", "<emphasis>amazing</emphasis>")
                text = text.replace("excellent", "<emphasis>excellent</emphasis>")
                text = text.replace("great", "<emphasis>great</emphasis>")
            
            # Slow down authoritative profile
            elif settings.voice_profile == VoiceProfile.AUTHORITATIVE:
                text = text.replace(". ", ".<break time=\"700ms\"/> ")
            
            return text
            
        except Exception as e:
            self.logger.error(f"Error applying voice profile adjustments: {str(e)}")
            return text
    
    async def _generate_audio(
        self, 
        text: str, 
        settings: VoiceSettings
    ) -> tuple[bytes, float]:
        """Generate audio from text (simulated for now)."""
        try:
            # In a real implementation, this would:
            # 1. Call Google Cloud Text-to-Speech API
            # 2. Or use a local TTS engine like espeak, festival, etc.
            # 3. Apply the voice settings and format conversion
            
            # For simulation, create dummy audio data
            estimated_duration = len(text) * 0.1  # Rough estimate: 0.1 seconds per character
            
            # Adjust duration based on speech rate
            rate_multipliers = {
                SpeechRate.VERY_SLOW: 1.5,
                SpeechRate.SLOW: 1.2,
                SpeechRate.NORMAL: 1.0,
                SpeechRate.FAST: 0.8,
                SpeechRate.VERY_FAST: 0.6
            }
            duration = estimated_duration * rate_multipliers.get(settings.speech_rate, 1.0)
            
            # Generate dummy audio data (in real implementation, this would be actual audio)
            sample_count = int(duration * settings.sample_rate)
            audio_data = b'\x00' * (sample_count * 2)  # 16-bit audio, so 2 bytes per sample
            
            # Encode as base64 for transmission
            audio_b64 = base64.b64encode(audio_data).decode('utf-8')
            
            self.logger.debug(f"Generated audio: {len(text)} chars -> {duration:.2f}s")
            return audio_data, duration
            
        except Exception as e:
            self.logger.error(f"Error generating audio: {str(e)}")
            raise
    
    def _split_text_into_chunks(self, text: str, chunk_size: int) -> List[str]:
        """Split text into smaller chunks for synthesis."""
        try:
            chunks = []
            
            # Split by sentences first
            sentences = text.split('. ')
            
            current_chunk = ""
            for sentence in sentences:
                # Add sentence to current chunk if it fits
                if len(current_chunk + sentence) <= chunk_size:
                    current_chunk += sentence + ". "
                else:
                    # Save current chunk and start new one
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = sentence + ". "
            
            # Add final chunk
            if current_chunk:
                chunks.append(current_chunk.strip())
            
            return chunks
            
        except Exception as e:
            self.logger.error(f"Error splitting text into chunks: {str(e)}")
            return [text]  # Return original text as single chunk
    
    async def _combine_audio_chunks(
        self, 
        audio_chunks: List[bytes], 
        settings: VoiceSettings
    ) -> bytes:
        """Combine multiple audio chunks into a single audio stream."""
        try:
            # In a real implementation, this would:
            # 1. Parse audio headers
            # 2. Concatenate audio data properly
            # 3. Add appropriate silence between chunks
            # 4. Ensure smooth transitions
            
            # For simulation, just concatenate the data
            combined_audio = b''.join(audio_chunks)
            
            return combined_audio
            
        except Exception as e:
            self.logger.error(f"Error combining audio chunks: {str(e)}")
            raise
    
    async def synthesize_streaming(
        self,
        text: str,
        voice_settings: Optional[VoiceSettings] = None,
        chunk_callback: Optional[Callable] = None
    ) -> None:
        """
        Synthesize speech with streaming output.
        
        Args:
            text: Text to synthesize
            voice_settings: Voice synthesis settings
            chunk_callback: Callback for each audio chunk
        """
        try:
            if voice_settings is None:
                voice_settings = self.default_voice_settings
            
            # Split text into chunks
            chunks = self._split_text_into_chunks(text, self.chunk_size)
            
            # Synthesize and stream each chunk
            for i, chunk in enumerate(chunks):
                chunk_result = await self.synthesize_speech(
                    chunk,
                    voice_settings,
                    request_id=f"stream_{i}_{int(datetime.utcnow().timestamp())}"
                )
                
                if chunk_result.success and chunk_callback:
                    await chunk_callback(
                        chunk_result.audio_data,
                        i,
                        len(chunks),
                        chunk_result.duration_seconds
                    )
            
        except Exception as e:
            self.logger.error(f"Error in streaming synthesis: {str(e)}")
            raise
    
    def get_voice_profiles(self) -> List[Dict[str, Any]]:
        """Get available voice profiles."""
        return [
            {
                "name": profile.value,
                "display_name": profile.value.replace("_", " ").title(),
                "config": config
            }
            for profile, config in self.voice_profiles.items()
        ]
    
    def get_synthesis_stats(self) -> Dict[str, Any]:
        """Get synthesis statistics."""
        return {
            "total_requests": self.total_requests,
            "successful_syntheses": self.successful_syntheses,
            "failed_syntheses": self.failed_syntheses,
            "success_rate": self.successful_syntheses / self.total_requests if self.total_requests > 0 else 0,
            "total_audio_duration": self.total_audio_duration,
            "active_requests": len(self.active_requests),
            "average_duration_per_request": self.total_audio_duration / self.successful_syntheses if self.successful_syntheses > 0 else 0
        }
    
    async def get_health(self) -> Dict[str, Any]:
        """Get health status of the voice synthesizer."""
        try:
            stats = self.get_synthesis_stats()
            
            return {
                "status": "healthy",
                "total_requests": stats["total_requests"],
                "success_rate": stats["success_rate"],
                "active_requests": stats["active_requests"],
                "supported_formats": [fmt.value for fmt in AudioFormat]
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    async def shutdown(self) -> None:
        """Shutdown the voice synthesizer."""
        try:
            self.logger.info("Shutting down VoiceSynthesizer...")
            
            # Cancel active requests
            for request_id in list(self.active_requests.keys()):
                del self.active_requests[request_id]
            
            # Log final statistics
            stats = self.get_synthesis_stats()
            self.logger.info(f"Final synthesis stats: {stats}")
            
            self.logger.info("VoiceSynthesizer shutdown completed")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {str(e)}")
            raise 