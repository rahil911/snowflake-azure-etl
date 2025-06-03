"""
Multimodal Processor
===================

Handles text, audio, and visual inputs for Gemini 2.0 integration.
Processes different modalities and converts them to appropriate formats.

Features:
- Text input processing and normalization
- Audio format detection and conversion
- Image and video processing (future)
- Input validation and sanitization
- Format optimization for Gemini 2.0
- Content type detection
- Batch processing support
"""

import asyncio
import base64
import io
import json
import mimetypes
from datetime import datetime
from typing import Dict, Any, List, Optional, Union, Tuple
from enum import Enum
import logging

from shared.config.logging_config import setup_logging
from shared.utils.validation import validate_input


class ModalityType(Enum):
    """Supported input modalities."""
    TEXT = "text"
    AUDIO = "audio"
    IMAGE = "image"
    VIDEO = "video"
    DOCUMENT = "document"
    UNKNOWN = "unknown"


class ProcessingStatus(Enum):
    """Processing status types."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    ERROR = "error"


class InputFormat:
    """Represents an input format specification."""
    
    def __init__(
        self,
        modality: ModalityType,
        mime_type: str,
        encoding: Optional[str] = None,
        parameters: Dict[str, Any] = None
    ):
        self.modality = modality
        self.mime_type = mime_type
        self.encoding = encoding
        self.parameters = parameters or {}


class ProcessedInput:
    """Represents processed multimodal input."""
    
    def __init__(
        self,
        modality: ModalityType,
        content: Any,
        format_info: InputFormat,
        metadata: Dict[str, Any] = None,
        confidence: float = 1.0
    ):
        self.modality = modality
        self.content = content
        self.format_info = format_info
        self.metadata = metadata or {}
        self.confidence = confidence
        self.timestamp = datetime.utcnow()
        self.processing_id = f"{modality.value}_{int(self.timestamp.timestamp())}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "processing_id": self.processing_id,
            "modality": self.modality.value,
            "format": {
                "mime_type": self.format_info.mime_type,
                "encoding": self.format_info.encoding,
                "parameters": self.format_info.parameters
            },
            "metadata": self.metadata,
            "confidence": self.confidence,
            "timestamp": self.timestamp.isoformat(),
            "has_content": self.content is not None
        }


class MultimodalProcessor:
    """
    Processes multimodal inputs for Gemini 2.0 integration.
    """
    
    def __init__(self, settings):
        self.settings = settings
        self.logger = setup_logging("MultimodalProcessor")
        
        # Supported formats
        self.supported_formats = self._load_supported_formats()
        
        # Processing limits
        self.max_text_length = 100000  # 100K characters
        self.max_audio_duration = 300  # 5 minutes
        self.max_image_size = 10 * 1024 * 1024  # 10MB
        self.max_video_duration = 60  # 1 minute
        
        # Statistics
        self.inputs_processed = 0
        self.processing_errors = 0
        self.modality_counts = {modality: 0 for modality in ModalityType}
        
        self.logger.info("MultimodalProcessor initialized")
    
    async def initialize(self) -> None:
        """Initialize the multimodal processor."""
        try:
            # Validate dependencies (would check for audio/image libraries in real implementation)
            self.logger.info("MultimodalProcessor initialization completed")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize MultimodalProcessor: {str(e)}")
            raise
    
    def _load_supported_formats(self) -> Dict[ModalityType, List[InputFormat]]:
        """Load supported input formats for each modality."""
        return {
            ModalityType.TEXT: [
                InputFormat(ModalityType.TEXT, "text/plain", "utf-8"),
                InputFormat(ModalityType.TEXT, "text/markdown", "utf-8"),
                InputFormat(ModalityType.TEXT, "application/json", "utf-8"),
            ],
            ModalityType.AUDIO: [
                InputFormat(ModalityType.AUDIO, "audio/wav", parameters={"sample_rate": 16000, "channels": 1}),
                InputFormat(ModalityType.AUDIO, "audio/mpeg", parameters={"sample_rate": 16000}),
                InputFormat(ModalityType.AUDIO, "audio/opus", parameters={"sample_rate": 16000}),
                InputFormat(ModalityType.AUDIO, "audio/webm", parameters={"sample_rate": 16000}),
            ],
            ModalityType.IMAGE: [
                InputFormat(ModalityType.IMAGE, "image/jpeg"),
                InputFormat(ModalityType.IMAGE, "image/png"),
                InputFormat(ModalityType.IMAGE, "image/webp"),
                InputFormat(ModalityType.IMAGE, "image/gif"),
            ],
            ModalityType.VIDEO: [
                InputFormat(ModalityType.VIDEO, "video/mp4"),
                InputFormat(ModalityType.VIDEO, "video/webm"),
                InputFormat(ModalityType.VIDEO, "video/avi"),
            ],
            ModalityType.DOCUMENT: [
                InputFormat(ModalityType.DOCUMENT, "application/pdf"),
                InputFormat(ModalityType.DOCUMENT, "application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
                InputFormat(ModalityType.DOCUMENT, "text/csv"),
            ]
        }
    
    async def process_input(
        self,
        content: Any,
        content_type: Optional[str] = None,
        metadata: Dict[str, Any] = None
    ) -> ProcessedInput:
        """
        Process multimodal input content.
        
        Args:
            content: Raw input content (text, bytes, etc.)
            content_type: Optional MIME type hint
            metadata: Optional metadata about the content
            
        Returns:
            ProcessedInput object with processed content
        """
        try:
            self.inputs_processed += 1
            metadata = metadata or {}
            
            # Detect modality and format
            modality, format_info = await self._detect_modality_and_format(content, content_type)
            
            # Validate input
            await self._validate_input(content, modality, format_info)
            
            # Process based on modality
            if modality == ModalityType.TEXT:
                processed_content = await self._process_text(content, format_info, metadata)
            elif modality == ModalityType.AUDIO:
                processed_content = await self._process_audio(content, format_info, metadata)
            elif modality == ModalityType.IMAGE:
                processed_content = await self._process_image(content, format_info, metadata)
            elif modality == ModalityType.VIDEO:
                processed_content = await self._process_video(content, format_info, metadata)
            elif modality == ModalityType.DOCUMENT:
                processed_content = await self._process_document(content, format_info, metadata)
            else:
                raise ValueError(f"Unsupported modality: {modality}")
            
            # Update statistics
            self.modality_counts[modality] += 1
            
            # Create processed input
            processed_input = ProcessedInput(
                modality=modality,
                content=processed_content,
                format_info=format_info,
                metadata=metadata,
                confidence=0.95  # High confidence for successful processing
            )
            
            self.logger.debug(f"Processed {modality.value} input: {processed_input.processing_id}")
            return processed_input
            
        except Exception as e:
            self.processing_errors += 1
            self.logger.error(f"Error processing input: {str(e)}")
            
            # Return error result
            return ProcessedInput(
                modality=ModalityType.UNKNOWN,
                content=None,
                format_info=InputFormat(ModalityType.UNKNOWN, "application/octet-stream"),
                metadata={"error": str(e)},
                confidence=0.0
            )
    
    async def _detect_modality_and_format(
        self,
        content: Any,
        content_type: Optional[str] = None
    ) -> Tuple[ModalityType, InputFormat]:
        """Detect the modality and format of input content."""
        try:
            # If content type is provided, use it
            if content_type:
                modality = self._mime_type_to_modality(content_type)
                if modality != ModalityType.UNKNOWN:
                    format_info = InputFormat(modality, content_type)
                    return modality, format_info
            
            # Detect based on content type
            if isinstance(content, str):
                # Text content
                if self._looks_like_json(content):
                    return ModalityType.TEXT, InputFormat(ModalityType.TEXT, "application/json", "utf-8")
                elif self._looks_like_markdown(content):
                    return ModalityType.TEXT, InputFormat(ModalityType.TEXT, "text/markdown", "utf-8")
                else:
                    return ModalityType.TEXT, InputFormat(ModalityType.TEXT, "text/plain", "utf-8")
            
            elif isinstance(content, bytes):
                # Binary content - need to detect format
                detected_type = self._detect_binary_format(content)
                modality = self._mime_type_to_modality(detected_type)
                return modality, InputFormat(modality, detected_type)
            
            elif hasattr(content, 'read'):
                # File-like object
                # Try to read a small chunk to detect format
                chunk = content.read(1024)
                content.seek(0)  # Reset position
                
                detected_type = self._detect_binary_format(chunk)
                modality = self._mime_type_to_modality(detected_type)
                return modality, InputFormat(modality, detected_type)
            
            # Default to unknown
            return ModalityType.UNKNOWN, InputFormat(ModalityType.UNKNOWN, "application/octet-stream")
            
        except Exception as e:
            self.logger.error(f"Error detecting modality: {str(e)}")
            return ModalityType.UNKNOWN, InputFormat(ModalityType.UNKNOWN, "application/octet-stream")
    
    def _mime_type_to_modality(self, mime_type: str) -> ModalityType:
        """Convert MIME type to modality."""
        mime_type = mime_type.lower()
        
        if mime_type.startswith('text/'):
            return ModalityType.TEXT
        elif mime_type.startswith('audio/'):
            return ModalityType.AUDIO
        elif mime_type.startswith('image/'):
            return ModalityType.IMAGE
        elif mime_type.startswith('video/'):
            return ModalityType.VIDEO
        elif mime_type in ['application/pdf', 'application/msword', 'text/csv']:
            return ModalityType.DOCUMENT
        elif mime_type == 'application/json':
            return ModalityType.TEXT
        else:
            return ModalityType.UNKNOWN
    
    def _looks_like_json(self, text: str) -> bool:
        """Check if text looks like JSON."""
        try:
            text = text.strip()
            return (text.startswith('{') and text.endswith('}')) or \
                   (text.startswith('[') and text.endswith(']'))
        except Exception:
            return False
    
    def _looks_like_markdown(self, text: str) -> bool:
        """Check if text looks like Markdown."""
        try:
            markdown_indicators = ['#', '**', '*', '`', '![', '](', '- ', '1. ']
            return any(indicator in text for indicator in markdown_indicators)
        except Exception:
            return False
    
    def _detect_binary_format(self, data: bytes) -> str:
        """Detect binary format from data headers."""
        try:
            if not data:
                return "application/octet-stream"
            
            # Check for common file signatures
            if data.startswith(b'\xff\xfb') or data.startswith(b'ID3'):
                return "audio/mpeg"
            elif data.startswith(b'RIFF') and b'WAVE' in data[:12]:
                return "audio/wav"
            elif data.startswith(b'OggS'):
                return "audio/ogg"
            elif data.startswith(b'\xff\xd8\xff'):
                return "image/jpeg"
            elif data.startswith(b'\x89PNG'):
                return "image/png"
            elif data.startswith(b'GIF8'):
                return "image/gif"
            elif data.startswith(b'RIFF') and b'WEBP' in data[:12]:
                return "image/webp"
            elif data.startswith(b'\x00\x00\x00 ftypmp4') or data.startswith(b'\x00\x00\x00\x18ftypmp4'):
                return "video/mp4"
            elif data.startswith(b'%PDF'):
                return "application/pdf"
            else:
                return "application/octet-stream"
                
        except Exception:
            return "application/octet-stream"
    
    async def _validate_input(
        self,
        content: Any,
        modality: ModalityType,
        format_info: InputFormat
    ) -> None:
        """Validate input based on modality and format."""
        try:
            if modality == ModalityType.TEXT:
                if isinstance(content, str) and len(content) > self.max_text_length:
                    raise ValueError(f"Text input too long: {len(content)} > {self.max_text_length}")
            
            elif modality == ModalityType.AUDIO:
                if isinstance(content, bytes) and len(content) > self.max_audio_duration * 16000 * 2:
                    # Rough estimate: duration * sample_rate * bytes_per_sample
                    raise ValueError("Audio input too long")
            
            elif modality == ModalityType.IMAGE:
                if isinstance(content, bytes) and len(content) > self.max_image_size:
                    raise ValueError(f"Image too large: {len(content)} > {self.max_image_size}")
            
            elif modality == ModalityType.VIDEO:
                # Video validation would require more complex analysis
                pass
            
        except Exception as e:
            self.logger.error(f"Input validation failed: {str(e)}")
            raise
    
    async def _process_text(
        self,
        content: str,
        format_info: InputFormat,
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process text input."""
        try:
            processed = {
                "text": content.strip(),
                "length": len(content),
                "encoding": format_info.encoding or "utf-8",
                "language": metadata.get("language", "auto-detect")
            }
            
            # Add basic text analysis
            processed["word_count"] = len(content.split())
            processed["line_count"] = len(content.splitlines())
            
            # Check for special formats
            if format_info.mime_type == "application/json":
                try:
                    parsed_json = json.loads(content)
                    processed["json_valid"] = True
                    processed["json_structure"] = type(parsed_json).__name__
                except json.JSONDecodeError:
                    processed["json_valid"] = False
            
            return processed
            
        except Exception as e:
            self.logger.error(f"Error processing text: {str(e)}")
            raise
    
    async def _process_audio(
        self,
        content: bytes,
        format_info: InputFormat,
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process audio input."""
        try:
            processed = {
                "audio_data": base64.b64encode(content).decode('utf-8'),
                "size_bytes": len(content),
                "format": format_info.mime_type,
                "encoding": "base64"
            }
            
            # Add format-specific parameters
            if format_info.parameters:
                processed.update(format_info.parameters)
            
            # Estimate duration (rough calculation)
            if format_info.mime_type == "audio/wav":
                # WAV files have header information we could parse
                # For now, use a rough estimate
                sample_rate = format_info.parameters.get("sample_rate", 16000)
                channels = format_info.parameters.get("channels", 1)
                bytes_per_sample = 2  # 16-bit
                
                estimated_duration = len(content) / (sample_rate * channels * bytes_per_sample)
                processed["estimated_duration_seconds"] = estimated_duration
            
            return processed
            
        except Exception as e:
            self.logger.error(f"Error processing audio: {str(e)}")
            raise
    
    async def _process_image(
        self,
        content: bytes,
        format_info: InputFormat,
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process image input."""
        try:
            processed = {
                "image_data": base64.b64encode(content).decode('utf-8'),
                "size_bytes": len(content),
                "format": format_info.mime_type,
                "encoding": "base64"
            }
            
            # In a real implementation, we would use PIL or similar to extract:
            # - Image dimensions
            # - Color mode
            # - EXIF data
            # For now, we'll add placeholder analysis
            processed["analysis_placeholder"] = "Image analysis would be performed here"
            
            return processed
            
        except Exception as e:
            self.logger.error(f"Error processing image: {str(e)}")
            raise
    
    async def _process_video(
        self,
        content: bytes,
        format_info: InputFormat,
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process video input."""
        try:
            processed = {
                "video_data": base64.b64encode(content).decode('utf-8'),
                "size_bytes": len(content),
                "format": format_info.mime_type,
                "encoding": "base64"
            }
            
            # In a real implementation, we would use ffmpeg or similar to extract:
            # - Video duration
            # - Frame rate
            # - Resolution
            # - Audio tracks
            processed["analysis_placeholder"] = "Video analysis would be performed here"
            
            return processed
            
        except Exception as e:
            self.logger.error(f"Error processing video: {str(e)}")
            raise
    
    async def _process_document(
        self,
        content: bytes,
        format_info: InputFormat,
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process document input."""
        try:
            processed = {
                "document_data": base64.b64encode(content).decode('utf-8'),
                "size_bytes": len(content),
                "format": format_info.mime_type,
                "encoding": "base64"
            }
            
            # In a real implementation, we would extract text content:
            # - PDF: use PyPDF2 or similar
            # - Word: use python-docx
            # - CSV: use pandas
            processed["analysis_placeholder"] = "Document text extraction would be performed here"
            
            return processed
            
        except Exception as e:
            self.logger.error(f"Error processing document: {str(e)}")
            raise
    
    async def process_batch(
        self,
        inputs: List[Tuple[Any, Optional[str], Optional[Dict[str, Any]]]]
    ) -> List[ProcessedInput]:
        """
        Process multiple inputs in batch.
        
        Args:
            inputs: List of (content, content_type, metadata) tuples
            
        Returns:
            List of ProcessedInput objects
        """
        try:
            results = []
            
            # Process inputs concurrently
            tasks = [
                self.process_input(content, content_type, metadata)
                for content, content_type, metadata in inputs
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle any exceptions
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    self.logger.error(f"Error processing input {i}: {str(result)}")
                    processed_results.append(ProcessedInput(
                        modality=ModalityType.UNKNOWN,
                        content=None,
                        format_info=InputFormat(ModalityType.UNKNOWN, "application/octet-stream"),
                        metadata={"error": str(result), "batch_index": i},
                        confidence=0.0
                    ))
                else:
                    processed_results.append(result)
            
            return processed_results
            
        except Exception as e:
            self.logger.error(f"Error processing batch: {str(e)}")
            raise
    
    def get_supported_formats(self, modality: Optional[ModalityType] = None) -> Dict[str, List[str]]:
        """Get supported formats, optionally filtered by modality."""
        try:
            if modality:
                if modality in self.supported_formats:
                    return {
                        modality.value: [fmt.mime_type for fmt in self.supported_formats[modality]]
                    }
                else:
                    return {}
            
            # Return all supported formats
            result = {}
            for mod, formats in self.supported_formats.items():
                result[mod.value] = [fmt.mime_type for fmt in formats]
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error getting supported formats: {str(e)}")
            return {}
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return {
            "inputs_processed": self.inputs_processed,
            "processing_errors": self.processing_errors,
            "error_rate": self.processing_errors / self.inputs_processed if self.inputs_processed > 0 else 0,
            "modality_counts": {mod.value: count for mod, count in self.modality_counts.items()},
            "supported_modalities": len(self.supported_formats)
        }
    
    async def get_health(self) -> Dict[str, Any]:
        """Get health status of the multimodal processor."""
        try:
            stats = self.get_processing_stats()
            
            return {
                "status": "healthy",
                "inputs_processed": stats["inputs_processed"],
                "error_rate": stats["error_rate"],
                "supported_modalities": list(self.supported_formats.keys())
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    async def shutdown(self) -> None:
        """Shutdown the multimodal processor."""
        try:
            self.logger.info("Shutting down MultimodalProcessor...")
            
            # Log final statistics
            stats = self.get_processing_stats()
            self.logger.info(f"Final processing stats: {stats}")
            
            self.logger.info("MultimodalProcessor shutdown completed")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {str(e)}")
            raise 