"""Cryptographic utilities for secrets management."""

import base64
import logging
import os
from pathlib import Path

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, hmac
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

_logger = logging.getLogger(__name__)


class SecretManager:
    """
    Handles encryption, decryption, and hashing for secrets.

    This class uses Fernet symmetric encryption for secret values and HMAC-SHA256
    for deterministic name hashing. It derives separate keys for encryption and
    hashing from a master key using HKDF.

    The master key can be provided via:
    - MLFLOW_SECRET_MASTER_KEY environment variable (direct key)
    - MLFLOW_SECRET_MASTER_KEY_FILE environment variable (path to file containing key)
    - Auto-generated temporary key (for testing only)
    """

    def __init__(self):
        master_key = self._get_or_create_master_key()

        encryption_key = self._derive_key(
            master_key, salt=b"mlflow_secret_encryption", info=b"encryption_key"
        )
        self._cipher = Fernet(base64.urlsafe_b64encode(encryption_key))

        self._hash_key = self._derive_key(
            master_key, salt=b"mlflow_secret_hashing", info=b"hash_key"
        )

    def _derive_key(self, master_key: str, salt: bytes, info: bytes) -> bytes:
        """
        Derive a key from the master key using HKDF.

        Args:
            master_key: The master key to derive from.
            salt: Salt for key derivation.
            info: Context-specific info for key derivation.

        Returns:
            32-byte derived key.
        """
        kdf = HKDF(algorithm=hashes.SHA256(), length=32, salt=salt, info=info)
        return kdf.derive(master_key.encode())

    def _get_or_create_master_key(self) -> str:
        """
        Get the master key from environment variables or create a temporary one.

        Returns:
            str: The master key.

        Raises:
            ValueError: If both MLFLOW_SECRET_MASTER_KEY and MLFLOW_SECRET_MASTER_KEY_FILE
                       are set, or if the key file is empty or missing.
        """
        env_key = os.environ.get("MLFLOW_SECRET_MASTER_KEY")
        key_file = os.environ.get("MLFLOW_SECRET_MASTER_KEY_FILE")

        if env_key and key_file:
            raise ValueError(
                "Both MLFLOW_SECRET_MASTER_KEY and MLFLOW_SECRET_MASTER_KEY_FILE are set. "
                "Please set only one."
            )

        if env_key:
            return env_key

        if key_file:
            key_file_path = Path(key_file).expanduser().resolve()
            if not key_file_path.exists():
                raise ValueError(f"MLFLOW_SECRET_MASTER_KEY_FILE '{key_file_path}' does not exist.")
            key = key_file_path.read_text().strip()
            if not key:
                raise ValueError(f"MLFLOW_SECRET_MASTER_KEY_FILE '{key_file_path}' is empty.")
            return key

        new_key = Fernet.generate_key().decode()
        _logger.warning(
            "Neither MLFLOW_SECRET_MASTER_KEY nor MLFLOW_SECRET_MASTER_KEY_FILE is set. "
            "Generated a temporary master key for this session. "
            "Secrets will NOT be recoverable after restart. "
            "Set MLFLOW_SECRET_MASTER_KEY or MLFLOW_SECRET_MASTER_KEY_FILE for production use."
        )
        return new_key

    def encrypt(self, plaintext: str) -> str:
        """
        Encrypt a plaintext string.

        Args:
            plaintext: The string to encrypt.

        Returns:
            Base64-encoded encrypted string.
        """
        return self._cipher.encrypt(plaintext.encode()).decode()

    def decrypt(self, ciphertext: str) -> str:
        """
        Decrypt an encrypted string.

        Args:
            ciphertext: Base64-encoded encrypted string.

        Returns:
            Decrypted plaintext string.
        """
        return self._cipher.decrypt(ciphertext.encode()).decode()

    def hash_name(self, name: str) -> str:
        """
        Create a deterministic hash of a secret name for lookup.

        Uses HMAC-SHA256 for secure, deterministic hashing.

        Args:
            name: The secret name to hash.

        Returns:
            Hex-encoded hash (64 characters).
        """
        h = hmac.HMAC(self._hash_key, hashes.SHA256())
        h.update(name.encode())
        return h.finalize().hex()

    def generate_dek(self) -> bytes:
        """
        Generate a random Data Encryption Key (DEK) for envelope encryption.

        Returns:
            32-byte random key suitable for Fernet encryption.
        """
        return Fernet.generate_key()

    def encrypt_with_dek(self, plaintext: str, dek: bytes) -> str:
        """
        Encrypt plaintext with a Data Encryption Key (DEK).

        Args:
            plaintext: The string to encrypt.
            dek: The Data Encryption Key (32 bytes).

        Returns:
            Base64-encoded encrypted string.
        """
        cipher = Fernet(dek)
        return cipher.encrypt(plaintext.encode()).decode()

    def decrypt_with_dek(self, ciphertext: str, dek: bytes) -> str:
        """
        Decrypt ciphertext with a Data Encryption Key (DEK).

        Args:
            ciphertext: Base64-encoded encrypted string.
            dek: The Data Encryption Key (32 bytes).

        Returns:
            Decrypted plaintext string.
        """
        cipher = Fernet(dek)
        return cipher.decrypt(ciphertext.encode()).decode()

    def encrypt_dek(self, dek: bytes) -> str:
        """
        Encrypt a DEK with the master key (for envelope encryption).

        Args:
            dek: The Data Encryption Key to encrypt (32 bytes).

        Returns:
            Base64-encoded encrypted DEK.
        """
        return self._cipher.encrypt(dek).decode()

    def decrypt_dek(self, encrypted_dek: str) -> bytes:
        """
        Decrypt a DEK that was encrypted with the master key.

        Args:
            encrypted_dek: Base64-encoded encrypted DEK.

        Returns:
            Decrypted DEK (32 bytes).
        """
        return self._cipher.decrypt(encrypted_dek.encode())
