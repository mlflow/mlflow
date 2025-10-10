import os

import pytest
from cryptography.fernet import InvalidToken

from mlflow.secrets.crypto import SecretManager


@pytest.fixture
def secret_manager():
    os.environ["MLFLOW_SECRET_MASTER_KEY"] = "test_master_key_1234567890123456789012"
    manager = SecretManager()
    yield manager
    os.environ.pop("MLFLOW_SECRET_MASTER_KEY", None)


def test_generate_dek(secret_manager):
    dek1 = secret_manager.generate_dek()
    dek2 = secret_manager.generate_dek()

    assert isinstance(dek1, bytes)
    assert isinstance(dek2, bytes)
    assert len(dek1) == 44
    assert len(dek2) == 44
    assert dek1 != dek2


def test_encrypt_with_dek(secret_manager):
    dek = secret_manager.generate_dek()
    plaintext = "my_secret_value"

    encrypted = secret_manager.encrypt_with_dek(plaintext, dek)

    assert isinstance(encrypted, str)
    assert encrypted != plaintext


def test_decrypt_with_dek(secret_manager):
    dek = secret_manager.generate_dek()
    plaintext = "my_secret_value"

    encrypted = secret_manager.encrypt_with_dek(plaintext, dek)
    decrypted = secret_manager.decrypt_with_dek(encrypted, dek)

    assert decrypted == plaintext


def test_encrypt_decrypt_with_different_deks(secret_manager):
    dek1 = secret_manager.generate_dek()
    dek2 = secret_manager.generate_dek()
    plaintext = "my_secret_value"

    encrypted = secret_manager.encrypt_with_dek(plaintext, dek1)

    with pytest.raises(InvalidToken):  # noqa: PT011
        secret_manager.decrypt_with_dek(encrypted, dek2)


def test_encrypt_dek(secret_manager):
    dek = secret_manager.generate_dek()

    encrypted_dek = secret_manager.encrypt_dek(dek)

    assert isinstance(encrypted_dek, str)
    assert encrypted_dek != dek.decode()


def test_decrypt_dek(secret_manager):
    dek = secret_manager.generate_dek()

    encrypted_dek = secret_manager.encrypt_dek(dek)
    decrypted_dek = secret_manager.decrypt_dek(encrypted_dek)

    assert decrypted_dek == dek


def test_envelope_encryption_full_flow(secret_manager):
    plaintext = "my_secret_value"

    dek = secret_manager.generate_dek()

    encrypted_value = secret_manager.encrypt_with_dek(plaintext, dek)
    encrypted_dek = secret_manager.encrypt_dek(dek)

    decrypted_dek = secret_manager.decrypt_dek(encrypted_dek)
    decrypted_value = secret_manager.decrypt_with_dek(encrypted_value, decrypted_dek)

    assert decrypted_value == plaintext


def test_key_rotation_simulation(secret_manager, monkeypatch):
    plaintext = "my_secret_value"

    dek = secret_manager.generate_dek()
    encrypted_value = secret_manager.encrypt_with_dek(plaintext, dek)
    old_encrypted_dek = secret_manager.encrypt_dek(dek)

    monkeypatch.setenv("MLFLOW_SECRET_MASTER_KEY", "new_master_key_9876543210987654321098")
    new_manager = SecretManager()

    decrypted_dek = secret_manager.decrypt_dek(old_encrypted_dek)
    new_encrypted_dek = new_manager.encrypt_dek(decrypted_dek)

    final_dek = new_manager.decrypt_dek(new_encrypted_dek)
    final_value = new_manager.decrypt_with_dek(encrypted_value, final_dek)

    assert final_value == plaintext


def test_envelope_encryption_with_long_value(secret_manager):
    plaintext = "a" * 10000

    dek = secret_manager.generate_dek()
    encrypted_value = secret_manager.encrypt_with_dek(plaintext, dek)
    encrypted_dek = secret_manager.encrypt_dek(dek)

    decrypted_dek = secret_manager.decrypt_dek(encrypted_dek)
    decrypted_value = secret_manager.decrypt_with_dek(encrypted_value, decrypted_dek)

    assert decrypted_value == plaintext


def test_envelope_encryption_with_unicode(secret_manager):
    plaintext = "🔐 Secret with émojis and spëcial çharacters 你好"

    dek = secret_manager.generate_dek()
    encrypted_value = secret_manager.encrypt_with_dek(plaintext, dek)
    encrypted_dek = secret_manager.encrypt_dek(dek)

    decrypted_dek = secret_manager.decrypt_dek(encrypted_dek)
    decrypted_value = secret_manager.decrypt_with_dek(encrypted_value, decrypted_dek)

    assert decrypted_value == plaintext


def test_different_managers_same_master_key(monkeypatch):
    monkeypatch.setenv("MLFLOW_SECRET_MASTER_KEY", "shared_master_key_1234567890123456789")

    manager1 = SecretManager()
    manager2 = SecretManager()

    dek = manager1.generate_dek()
    encrypted_dek = manager1.encrypt_dek(dek)

    decrypted_dek = manager2.decrypt_dek(encrypted_dek)

    assert decrypted_dek == dek


def test_different_managers_different_master_keys(monkeypatch):
    monkeypatch.setenv("MLFLOW_SECRET_MASTER_KEY", "master_key_1_1234567890123456789012")
    manager1 = SecretManager()

    dek = manager1.generate_dek()
    encrypted_dek = manager1.encrypt_dek(dek)

    monkeypatch.setenv("MLFLOW_SECRET_MASTER_KEY", "master_key_2_9876543210987654321098")
    manager2 = SecretManager()

    with pytest.raises(InvalidToken):  # noqa: PT011
        manager2.decrypt_dek(encrypted_dek)
