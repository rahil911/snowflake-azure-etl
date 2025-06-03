"""
Secrets management for the multi-agent data intelligence platform.

This module provides secure handling of credentials and sensitive configuration
with support for environment variables, encrypted storage, and secret rotation.
"""

import os
import base64
import json
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Union, List
from pathlib import Path
from functools import lru_cache

from pydantic import BaseModel, Field, ConfigDict, SecretStr, validator
from cryptography.fernet import Fernet


class SecretValue(BaseModel):
    """Container for secret values with metadata."""
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra='forbid'
    )
    
    name: str = Field(..., description="Secret name/identifier")
    value: SecretStr = Field(..., description="Secret value")
    description: Optional[str] = Field(default=None, description="Secret description")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = Field(default=None, description="Expiration time")
    tags: List[str] = Field(default_factory=list, description="Secret tags")
    
    @property
    def is_expired(self) -> bool:
        """Check if secret has expired."""
        if self.expires_at is None:
            return False
        return datetime.utcnow() > self.expires_at
    
    @property
    def expires_soon(self, warning_days: int = 7) -> bool:
        """Check if secret expires within warning period."""
        if self.expires_at is None:
            return False
        warning_time = datetime.utcnow() + timedelta(days=warning_days)
        return self.expires_at <= warning_time
    
    def get_secret_value(self) -> str:
        """Get the actual secret value."""
        return self.value.get_secret_value()
    
    def to_safe_dict(self) -> Dict[str, Any]:
        """Convert to dictionary without exposing secret value."""
        return {
            'name': self.name,
            'description': self.description,
            'created_at': self.created_at.isoformat(),
            'expires_at': self.expires_at.isoformat() if self.expires_at else None,
            'tags': self.tags,
            'is_expired': self.is_expired,
            'expires_soon': self.expires_soon
        }


class SecretsManager:
    """
    Centralized secrets management with encryption and rotation support.
    
    Provides secure storage and retrieval of sensitive configuration values
    with support for multiple backends (environment, files, external services).
    """
    
    def __init__(
        self,
        encryption_key: Optional[str] = None,
        storage_path: Optional[str] = None,
        enable_file_storage: bool = False
    ):
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Encryption setup
        self._fernet = None
        if encryption_key:
            try:
                key_bytes = base64.urlsafe_b64decode(encryption_key.encode())
                self._fernet = Fernet(key_bytes)
            except Exception as e:
                self.logger.warning(f"Failed to initialize encryption: {e}")
        elif enable_file_storage:
            # Generate a new key if file storage is enabled but no key provided
            self._fernet = Fernet(Fernet.generate_key())
            self.logger.warning("Generated new encryption key - store this securely!")
        
        # Storage setup
        self.enable_file_storage = enable_file_storage
        self.storage_path = Path(storage_path) if storage_path else Path("secrets/secrets.json")
        self._secrets_cache: Dict[str, SecretValue] = {}
        
        # Load existing secrets if file storage is enabled
        if self.enable_file_storage:
            self._load_secrets_from_file()
    
    def get_secret(
        self,
        name: str,
        default: Optional[str] = None,
        required: bool = False
    ) -> Optional[str]:
        """
        Get secret value by name.
        
        Args:
            name: Secret name
            default: Default value if secret not found
            required: If True, raises exception if secret not found
            
        Returns:
            Secret value or default
            
        Raises:
            ValueError: If required secret is not found
        """
        # Try cache first
        if name in self._secrets_cache:
            secret = self._secrets_cache[name]
            if secret.is_expired:
                self.logger.warning(f"Secret '{name}' has expired")
                if required:
                    raise ValueError(f"Required secret '{name}' has expired")
                return default
            
            if secret.expires_soon:
                self.logger.warning(f"Secret '{name}' expires soon: {secret.expires_at}")
            
            return secret.get_secret_value()
        
        # Try environment variables
        env_value = os.getenv(name)
        if env_value:
            # Cache environment secret
            self._secrets_cache[name] = SecretValue(
                name=name,
                value=SecretStr(env_value),
                description="From environment variable"
            )
            return env_value
        
        # Try common environment variable patterns
        env_patterns = [
            name.upper(),
            name.lower(),
            name.replace('-', '_').upper(),
            name.replace('_', '-').upper()
        ]
        
        for pattern in env_patterns:
            env_value = os.getenv(pattern)
            if env_value:
                self._secrets_cache[name] = SecretValue(
                    name=name,
                    value=SecretStr(env_value),
                    description=f"From environment variable {pattern}"
                )
                return env_value
        
        # Secret not found
        if required:
            raise ValueError(f"Required secret '{name}' not found")
        
        self.logger.debug(f"Secret '{name}' not found, using default")
        return default
    
    def set_secret(
        self,
        name: str,
        value: str,
        description: Optional[str] = None,
        expires_at: Optional[datetime] = None,
        tags: Optional[List[str]] = None,
        save_to_file: bool = None
    ) -> None:
        """
        Set secret value.
        
        Args:
            name: Secret name
            value: Secret value
            description: Optional description
            expires_at: Optional expiration time
            tags: Optional tags
            save_to_file: Whether to save to file (defaults to manager setting)
        """
        secret = SecretValue(
            name=name,
            value=SecretStr(value),
            description=description,
            expires_at=expires_at,
            tags=tags or []
        )
        
        self._secrets_cache[name] = secret
        
        # Save to file if enabled
        if save_to_file is True or (save_to_file is None and self.enable_file_storage):
            self._save_secrets_to_file()
        
        self.logger.info(f"Secret '{name}' updated")
    
    def delete_secret(self, name: str, remove_from_file: bool = None) -> bool:
        """
        Delete secret.
        
        Args:
            name: Secret name
            remove_from_file: Whether to remove from file (defaults to manager setting)
            
        Returns:
            True if secret was deleted, False if not found
        """
        if name not in self._secrets_cache:
            return False
        
        del self._secrets_cache[name]
        
        # Update file if enabled
        if remove_from_file is True or (remove_from_file is None and self.enable_file_storage):
            self._save_secrets_to_file()
        
        self.logger.info(f"Secret '{name}' deleted")
        return True
    
    def list_secrets(self, include_expired: bool = False) -> List[Dict[str, Any]]:
        """
        List all secrets (without values).
        
        Args:
            include_expired: Whether to include expired secrets
            
        Returns:
            List of secret metadata
        """
        secrets = []
        for secret in self._secrets_cache.values():
            if not include_expired and secret.is_expired:
                continue
            secrets.append(secret.to_safe_dict())
        
        return secrets
    
    def check_expiring_secrets(self, warning_days: int = 7) -> List[Dict[str, Any]]:
        """
        Check for secrets that are expiring soon.
        
        Args:
            warning_days: Days ahead to check for expiration
            
        Returns:
            List of expiring secrets
        """
        expiring = []
        for secret in self._secrets_cache.values():
            if secret.expires_soon(warning_days):
                expiring.append(secret.to_safe_dict())
        
        return expiring
    
    def rotate_secret(
        self,
        name: str,
        new_value: str,
        description: Optional[str] = None
    ) -> None:
        """
        Rotate a secret to a new value.
        
        Args:
            name: Secret name
            new_value: New secret value
            description: Optional description for the rotation
        """
        if name not in self._secrets_cache:
            raise ValueError(f"Secret '{name}' not found for rotation")
        
        old_secret = self._secrets_cache[name]
        
        # Create new secret with updated timestamp
        new_secret = SecretValue(
            name=name,
            value=SecretStr(new_value),
            description=description or f"Rotated from previous secret",
            expires_at=old_secret.expires_at,
            tags=old_secret.tags
        )
        
        self._secrets_cache[name] = new_secret
        
        if self.enable_file_storage:
            self._save_secrets_to_file()
        
        self.logger.info(f"Secret '{name}' rotated")
    
    def encrypt_value(self, value: str) -> str:
        """
        Encrypt a value using the configured encryption key.
        
        Args:
            value: Value to encrypt
            
        Returns:
            Encrypted value as base64 string
            
        Raises:
            RuntimeError: If encryption is not configured
        """
        if not self._fernet:
            raise RuntimeError("Encryption not configured")
        
        encrypted = self._fernet.encrypt(value.encode())
        return base64.urlsafe_b64encode(encrypted).decode()
    
    def decrypt_value(self, encrypted_value: str) -> str:
        """
        Decrypt a value using the configured encryption key.
        
        Args:
            encrypted_value: Encrypted value as base64 string
            
        Returns:
            Decrypted value
            
        Raises:
            RuntimeError: If encryption is not configured
        """
        if not self._fernet:
            raise RuntimeError("Encryption not configured")
        
        try:
            encrypted_bytes = base64.urlsafe_b64decode(encrypted_value.encode())
            decrypted = self._fernet.decrypt(encrypted_bytes)
            return decrypted.decode()
        except Exception as e:
            raise ValueError(f"Failed to decrypt value: {e}")
    
    def _load_secrets_from_file(self) -> None:
        """Load secrets from encrypted file storage."""
        if not self.storage_path.exists():
            self.logger.debug("No secrets file found")
            return
        
        try:
            with open(self.storage_path, 'r') as f:
                data = json.load(f)
            
            # Decrypt and load secrets
            for secret_data in data.get('secrets', []):
                try:
                    # Decrypt value if encryption is enabled
                    value = secret_data['value']
                    if self._fernet and secret_data.get('encrypted', False):
                        value = self.decrypt_value(value)
                    
                    secret = SecretValue(
                        name=secret_data['name'],
                        value=SecretStr(value),
                        description=secret_data.get('description'),
                        created_at=datetime.fromisoformat(secret_data['created_at']),
                        expires_at=datetime.fromisoformat(secret_data['expires_at']) if secret_data.get('expires_at') else None,
                        tags=secret_data.get('tags', [])
                    )
                    
                    self._secrets_cache[secret.name] = secret
                    
                except Exception as e:
                    self.logger.error(f"Failed to load secret '{secret_data.get('name', 'unknown')}': {e}")
            
            self.logger.info(f"Loaded {len(self._secrets_cache)} secrets from file")
            
        except Exception as e:
            self.logger.error(f"Failed to load secrets file: {e}")
    
    def _save_secrets_to_file(self) -> None:
        """Save secrets to encrypted file storage."""
        if not self.enable_file_storage:
            return
        
        try:
            # Ensure directory exists
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Prepare data for saving
            secrets_data = []
            for secret in self._secrets_cache.values():
                value = secret.get_secret_value()
                encrypted = False
                
                # Encrypt value if encryption is enabled
                if self._fernet:
                    value = self.encrypt_value(value)
                    encrypted = True
                
                secret_data = {
                    'name': secret.name,
                    'value': value,
                    'encrypted': encrypted,
                    'description': secret.description,
                    'created_at': secret.created_at.isoformat(),
                    'expires_at': secret.expires_at.isoformat() if secret.expires_at else None,
                    'tags': secret.tags
                }
                secrets_data.append(secret_data)
            
            # Save to file
            data = {
                'version': '1.0',
                'created_at': datetime.utcnow().isoformat(),
                'secrets': secrets_data
            }
            
            with open(self.storage_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            # Set restrictive permissions
            self.storage_path.chmod(0o600)
            
            self.logger.debug(f"Saved {len(secrets_data)} secrets to file")
            
        except Exception as e:
            self.logger.error(f"Failed to save secrets file: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get secrets manager status."""
        secrets_count = len(self._secrets_cache)
        expired_count = sum(1 for s in self._secrets_cache.values() if s.is_expired)
        expiring_count = sum(1 for s in self._secrets_cache.values() if s.expires_soon)
        
        return {
            'total_secrets': secrets_count,
            'expired_secrets': expired_count,
            'expiring_secrets': expiring_count,
            'encryption_enabled': self._fernet is not None,
            'file_storage_enabled': self.enable_file_storage,
            'storage_path': str(self.storage_path) if self.enable_file_storage else None
        }


# Global secrets manager instance
_secrets_manager: Optional[SecretsManager] = None


@lru_cache()
def get_secrets_manager() -> SecretsManager:
    """
    Get global secrets manager instance.
    
    Returns:
        Configured secrets manager
    """
    global _secrets_manager
    
    if _secrets_manager is None:
        # Initialize with environment-based configuration
        encryption_key = os.getenv('SECRETS_ENCRYPTION_KEY')
        storage_path = os.getenv('SECRETS_STORAGE_PATH', 'secrets/secrets.json')
        enable_file_storage = os.getenv('SECRETS_ENABLE_FILE_STORAGE', '').lower() in ('true', '1', 'yes')
        
        _secrets_manager = SecretsManager(
            encryption_key=encryption_key,
            storage_path=storage_path,
            enable_file_storage=enable_file_storage
        )
    
    return _secrets_manager


def get_secret(name: str, default: Optional[str] = None, required: bool = False) -> Optional[str]:
    """
    Convenience function to get secret value.
    
    Args:
        name: Secret name
        default: Default value if secret not found
        required: If True, raises exception if secret not found
        
    Returns:
        Secret value or default
    """
    return get_secrets_manager().get_secret(name, default, required)


def set_secret(
    name: str,
    value: str,
    description: Optional[str] = None,
    expires_at: Optional[datetime] = None,
    tags: Optional[List[str]] = None
) -> None:
    """
    Convenience function to set secret value.
    
    Args:
        name: Secret name
        value: Secret value
        description: Optional description
        expires_at: Optional expiration time
        tags: Optional tags
    """
    get_secrets_manager().set_secret(name, value, description, expires_at, tags)


def generate_encryption_key() -> str:
    """
    Generate a new encryption key for secrets.
    
    Returns:
        Base64-encoded encryption key
    """
    key = Fernet.generate_key()
    return base64.urlsafe_b64encode(key).decode() 