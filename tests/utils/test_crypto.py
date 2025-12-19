import json
import os

import pytest

from mlflow.exceptions import MlflowException
from mlflow.utils.crypto import (
    AES_256_KEY_LENGTH,
    GCM_NONCE_LENGTH,
    KEKManager,
    _create_aad,
    _decrypt_secret,
    _encrypt_secret,
    _encrypt_with_aes_gcm,
    _generate_dek,
    _mask_secret_value,
    _mask_string_value,
    decrypt_with_aes_gcm,
    rotate_secret_encryption,
    unwrap_dek,
    wrap_dek,
)

TEST_PASSPHRASE = "test-passphrase-for-kek-derivation"


@pytest.fixture
def kek_manager():
    return KEKManager(passphrase=TEST_PASSPHRASE)


def test_kek_manager_from_env_var(monkeypatch):
    monkeypatch.setenv("MLFLOW_CRYPTO_KEK_PASSPHRASE", "env-passphrase")
    kek_manager = KEKManager()
    kek = kek_manager.get_kek()
    assert len(kek) == AES_256_KEY_LENGTH
    assert isinstance(kek, bytes)


def test_kek_manager_from_parameter():
    kek_manager = KEKManager(passphrase=TEST_PASSPHRASE)
    kek = kek_manager.get_kek()
    assert len(kek) == AES_256_KEY_LENGTH
    assert isinstance(kek, bytes)


def test_kek_manager_deterministic():
    kek1 = KEKManager(passphrase=TEST_PASSPHRASE).get_kek()
    kek2 = KEKManager(passphrase=TEST_PASSPHRASE).get_kek()
    assert kek1 == kek2


@pytest.mark.parametrize(
    ("passphrase1", "passphrase2"),
    [
        ("passphrase-one", "passphrase-two"),
        ("short", "very-long-passphrase-with-many-characters"),
        ("pass123", "pass456"),
    ],
)
def test_kek_manager_different_passphrases(passphrase1, passphrase2):
    kek1 = KEKManager(passphrase=passphrase1).get_kek()
    kek2 = KEKManager(passphrase=passphrase2).get_kek()
    assert kek1 != kek2


def test_kek_manager_no_passphrase_uses_default(monkeypatch):
    monkeypatch.delenv("MLFLOW_CRYPTO_KEK_PASSPHRASE", raising=False)
    kek_manager = KEKManager()
    assert kek_manager.using_default_passphrase is True
    assert kek_manager.get_kek() is not None


def test_kek_manager_empty_passphrase_uses_default(monkeypatch):
    monkeypatch.setenv("MLFLOW_CRYPTO_KEK_PASSPHRASE", "")
    kek_manager = KEKManager()
    assert kek_manager.using_default_passphrase is True
    assert kek_manager.get_kek() is not None


def test_kek_manager_custom_passphrase_not_default():
    kek_manager = KEKManager(passphrase=TEST_PASSPHRASE)
    assert kek_manager.using_default_passphrase is False


def test_kek_manager_version_defaults_to_1():
    kek_manager = KEKManager(passphrase=TEST_PASSPHRASE)
    assert kek_manager.kek_version == 1


def test_kek_manager_version_from_parameter():
    kek_manager = KEKManager(passphrase=TEST_PASSPHRASE, kek_version=2)
    assert kek_manager.kek_version == 2


def test_kek_manager_version_from_env_var(monkeypatch):
    monkeypatch.setenv("MLFLOW_CRYPTO_KEK_PASSPHRASE", "env-passphrase")
    monkeypatch.setenv("MLFLOW_CRYPTO_KEK_VERSION", "3")
    kek_manager = KEKManager()
    assert kek_manager.kek_version == 3


def test_kek_manager_version_parameter_overrides_env(monkeypatch):
    monkeypatch.setenv("MLFLOW_CRYPTO_KEK_PASSPHRASE", "env-passphrase")
    monkeypatch.setenv("MLFLOW_CRYPTO_KEK_VERSION", "3")
    kek_manager = KEKManager(kek_version=5)
    assert kek_manager.kek_version == 5


def test_kek_manager_same_passphrase_different_versions_produces_different_keks():
    passphrase = "same-passphrase-for-both"
    kek_v1 = KEKManager(passphrase=passphrase, kek_version=1).get_kek()
    kek_v2 = KEKManager(passphrase=passphrase, kek_version=2).get_kek()
    assert kek_v1 != kek_v2


def test_encrypt_secret_includes_kek_version():
    kek_manager = KEKManager(passphrase=TEST_PASSPHRASE, kek_version=2)
    secret_value = "test-secret"
    secret_id = "test-id"
    secret_name = "test-name"

    result = _encrypt_secret(secret_value, kek_manager, secret_id, secret_name)

    assert result.kek_version == 2


def test_generate_dek_length():
    dek = _generate_dek()
    assert len(dek) == AES_256_KEY_LENGTH
    assert isinstance(dek, bytes)


def test_generate_dek_random():
    dek1 = _generate_dek()
    dek2 = _generate_dek()
    assert dek1 != dek2


def test_encrypt_decrypt_roundtrip():
    dek = _generate_dek()
    plaintext = b"Hello, World!"

    result = _encrypt_with_aes_gcm(plaintext, dek)

    assert len(result.nonce) == GCM_NONCE_LENGTH
    assert len(result.ciphertext) > len(plaintext)

    combined = result.nonce + result.ciphertext
    decrypted = decrypt_with_aes_gcm(combined, dek)
    assert decrypted == plaintext


def test_encrypt_with_custom_nonce():
    dek = _generate_dek()
    plaintext = b"Test data"
    custom_nonce = os.urandom(GCM_NONCE_LENGTH)

    result = _encrypt_with_aes_gcm(plaintext, dek, _nonce_for_testing=custom_nonce)
    assert result.nonce == custom_nonce


def test_encrypt_decrypt_with_aad():
    dek = _generate_dek()
    plaintext = b"Secret message"
    aad = b"metadata"

    result = _encrypt_with_aes_gcm(plaintext, dek, aad=aad)
    combined = result.nonce + result.ciphertext

    decrypted = decrypt_with_aes_gcm(combined, dek, aad=aad)
    assert decrypted == plaintext


def test_decrypt_with_wrong_aad_fails():
    dek = _generate_dek()
    plaintext = b"Secret message"
    aad = b"correct-metadata"

    result = _encrypt_with_aes_gcm(plaintext, dek, aad=aad)
    combined = result.nonce + result.ciphertext

    with pytest.raises(MlflowException, match="AES-GCM decryption failed"):
        decrypt_with_aes_gcm(combined, dek, aad=b"wrong-metadata")


def test_decrypt_with_missing_aad_fails():
    dek = _generate_dek()
    plaintext = b"Secret message"
    aad = b"metadata"

    result = _encrypt_with_aes_gcm(plaintext, dek, aad=aad)
    combined = result.nonce + result.ciphertext

    with pytest.raises(MlflowException, match="AES-GCM decryption failed"):
        decrypt_with_aes_gcm(combined, dek, aad=None)


@pytest.mark.parametrize("bad_key", [b"short", b"", b"a" * 16])
def test_encrypt_with_wrong_key_length_raises(bad_key):
    plaintext = b"Test"
    with pytest.raises(ValueError, match="Key must be 32 bytes"):
        _encrypt_with_aes_gcm(plaintext, bad_key)


@pytest.mark.parametrize("bad_key", [b"short", b"", b"a" * 16])
def test_decrypt_with_wrong_key_length_raises(bad_key):
    ciphertext = os.urandom(100)
    with pytest.raises(ValueError, match="Key must be 32 bytes"):
        decrypt_with_aes_gcm(ciphertext, bad_key)


@pytest.mark.parametrize("bad_nonce", [b"short", b"", b"a" * 5])
def test_encrypt_with_wrong_nonce_length_raises(bad_nonce):
    dek = _generate_dek()
    plaintext = b"Test"
    with pytest.raises(ValueError, match="Nonce must be between"):
        _encrypt_with_aes_gcm(plaintext, dek, _nonce_for_testing=bad_nonce)


def test_decrypt_with_wrong_key_fails():
    key1 = _generate_dek()
    key2 = _generate_dek()
    plaintext = b"Secret"

    result = _encrypt_with_aes_gcm(plaintext, key1)
    combined = result.nonce + result.ciphertext

    with pytest.raises(MlflowException, match="AES-GCM decryption failed"):
        decrypt_with_aes_gcm(combined, key2)


def test_decrypt_with_tampered_ciphertext_fails():
    dek = _generate_dek()
    plaintext = b"Secret"

    result = _encrypt_with_aes_gcm(plaintext, dek)
    combined = result.nonce + result.ciphertext

    tampered = combined[:-1] + bytes([combined[-1] ^ 0xFF])

    with pytest.raises(MlflowException, match="AES-GCM decryption failed"):
        decrypt_with_aes_gcm(tampered, dek)


def test_decrypt_with_short_ciphertext_raises():
    dek = _generate_dek()
    short_ciphertext = os.urandom(5)

    with pytest.raises(ValueError, match="Ciphertext too short"):
        decrypt_with_aes_gcm(short_ciphertext, dek)


def test_wrap_unwrap_dek_roundtrip():
    dek = _generate_dek()
    kek = _generate_dek()

    wrapped_dek = wrap_dek(dek, kek)
    unwrapped_dek = unwrap_dek(wrapped_dek, kek)

    assert unwrapped_dek == dek


def test_unwrap_with_wrong_kek_fails():
    dek = _generate_dek()
    kek1 = _generate_dek()
    kek2 = _generate_dek()

    wrapped_dek = wrap_dek(dek, kek1)

    with pytest.raises(MlflowException, match="Failed to unwrap DEK"):
        unwrap_dek(wrapped_dek, kek2)


def test__create_aad():
    secret_id = "123e4567-e89b-12d3-a456-426614174000"
    secret_name = "my-api-key"

    aad = _create_aad(secret_id, secret_name)

    assert isinstance(aad, bytes)
    assert secret_id.encode() in aad
    assert secret_name.encode() in aad
    assert b"|" in aad


def test_create_aad_deterministic():
    secret_id = "test-id"
    secret_name = "test-name"

    aad1 = _create_aad(secret_id, secret_name)
    aad2 = _create_aad(secret_id, secret_name)

    assert aad1 == aad2


@pytest.mark.parametrize(
    ("id1", "name1", "id2", "name2"),
    [
        ("id1", "name1", "id2", "name2"),
        ("same", "name1", "same", "name2"),
        ("id1", "same", "id2", "same"),
    ],
)
def test_create_aad_different_inputs(id1, name1, id2, name2):
    aad1 = _create_aad(id1, name1)
    aad2 = _create_aad(id2, name2)
    assert aad1 != aad2


@pytest.mark.parametrize(
    ("secret", "expected_mask"),
    [
        ("sk-proj-1234567890abcdef", "sk-...cdef"),
        ("pk-test-1234567890abcdef", "pk-...cdef"),
        ("ghp_1234567890abcdef1234567890abcdef", "ghp...cdef"),
        ("gho_1234567890abcdef1234567890abcdef", "gho...cdef"),
        ("ghu_1234567890abcdef1234567890abcdef", "ghu...cdef"),
        ("api-key-abc123def456", "api...f456"),
        ("12345678", "123...5678"),
    ],
)
def test__mask_string_value(secret, expected_mask):
    masked = _mask_string_value(secret)
    assert masked == expected_mask


@pytest.mark.parametrize("short_secret", ["short", "1234567", "abc", ""])
def test_mask_string_value_short(short_secret):
    if short_secret:
        masked = _mask_string_value(short_secret)
        assert masked == "***"


@pytest.mark.parametrize(
    ("secret_dict", "expected_masked"),
    [
        ({"api_key": "sk-proj-1234567890abcdef"}, {"api_key": "sk-...cdef"}),
        (
            {"username": "admin-user", "password": "secret123"},
            {"username": "adm...user", "password": "sec...t123"},
        ),
        ({"token": "ghp_1234567890abcdef"}, {"token": "ghp...cdef"}),
        (
            {"config": {"host": "localhost", "port": 8080}},
            {"config": "***"},
        ),
        ({"short": "abc"}, {"short": "***"}),
        ({}, {}),
        (
            {"key1": "val1", "key2": "val2", "key3": "val3", "key4": "val4"},
            {"key1": "***", "key2": "***", "key3": "***", "key4": "***"},
        ),
    ],
)
def test_mask_secret_value_dict(secret_dict, expected_masked):
    masked = _mask_secret_value(secret_dict)
    assert masked == expected_masked


def test_mask_secret_value_nested_dict():
    secret = {"outer": {"inner": {"api_key": "sk-abc123xyz", "enabled": True}}}
    masked = _mask_secret_value(secret)
    assert masked == {"outer": "***"}


def test_mask_secret_value_dict_preserves_keys():
    long_key = "a" * 200
    secret = {long_key: "test-longer-value-here"}
    masked = _mask_secret_value(secret)

    assert long_key in masked
    assert masked[long_key] == "tes...here"

    secret_multiple = {
        "key1_long": "val1_long_value",
        "key2_long": "val2_long_value",
        "key3_long": "val3_long_value",
    }
    masked_multiple = _mask_secret_value(secret_multiple)
    assert len(masked_multiple) == 3
    assert all(k in masked_multiple for k in secret_multiple)


@pytest.mark.parametrize(
    "secret_value",
    [
        "sk-abc123",
        "my-api-key-value",
        "password123",
    ],
)
def test_encrypt_decrypt_secret_roundtrip_string(kek_manager, secret_value):
    secret_id = "test-uuid-123"
    secret_name = "test-secret"

    result = _encrypt_secret(secret_value, kek_manager, secret_id, secret_name)

    decrypted_value = _decrypt_secret(
        result.encrypted_value,
        result.wrapped_dek,
        kek_manager,
        secret_id,
        secret_name,
    )

    assert decrypted_value == secret_value
    assert isinstance(decrypted_value, str)


@pytest.mark.parametrize(
    "secret_value",
    [
        {"api_key": "sk-abc123"},
        {"username": "admin", "password": "secret"},
        {"config": {"host": "localhost", "port": 8080}},
        {"value": "simple-string"},
    ],
)
def test_encrypt_decrypt_secret_roundtrip_dict(kek_manager, secret_value):
    secret_id = "test-uuid-123"
    secret_name = "test-secret"

    result = _encrypt_secret(secret_value, kek_manager, secret_id, secret_name)

    decrypted_value = _decrypt_secret(
        result.encrypted_value,
        result.wrapped_dek,
        kek_manager,
        secret_id,
        secret_name,
    )

    assert decrypted_value == secret_value
    assert isinstance(decrypted_value, dict)


def test_encrypt_secret_dict_is_json_serialized(kek_manager):
    secret_dict = {"key1": "value1", "key2": "value2"}
    secret_id = "test-uuid-123"
    secret_name = "test-secret"

    result = _encrypt_secret(secret_dict, kek_manager, secret_id, secret_name)

    kek = kek_manager.get_kek()
    dek = unwrap_dek(result.wrapped_dek, kek)
    aad = _create_aad(secret_id, secret_name)
    decrypted_bytes = decrypt_with_aes_gcm(result.encrypted_value, dek, aad=aad)

    decrypted_json = decrypted_bytes.decode("utf-8")
    parsed = json.loads(decrypted_json)

    assert parsed == secret_dict


def test_decrypt_with_wrong_secret_id_fails(kek_manager):
    secret_value = "my-secret-key"

    result = _encrypt_secret(secret_value, kek_manager, secret_id="id1", secret_name="name1")

    with pytest.raises(MlflowException, match="Failed to decrypt secret"):
        _decrypt_secret(
            result.encrypted_value,
            result.wrapped_dek,
            kek_manager,
            secret_id="id2",
            secret_name="name1",
        )


def test_decrypt_with_wrong_secret_name_fails(kek_manager):
    secret_value = "my-secret-key"

    result = _encrypt_secret(secret_value, kek_manager, secret_id="id1", secret_name="name1")

    with pytest.raises(MlflowException, match="Failed to decrypt secret"):
        _decrypt_secret(
            result.encrypted_value,
            result.wrapped_dek,
            kek_manager,
            secret_id="id1",
            secret_name="name2",
        )


def test_decrypt_with_wrong_passphrase_fails():
    kek_manager1 = KEKManager(passphrase="passphrase1")
    kek_manager2 = KEKManager(passphrase="passphrase2")
    secret_value = "my-secret-key"

    result = _encrypt_secret(secret_value, kek_manager1, secret_id="id1", secret_name="name1")

    with pytest.raises(MlflowException, match="Failed to decrypt secret"):
        _decrypt_secret(
            result.encrypted_value,
            result.wrapped_dek,
            kek_manager2,
            secret_id="id1",
            secret_name="name1",
        )


def test_encrypt_secret_unicode(kek_manager):
    secret_value = "🔐 Secret with emoji 密钥"
    secret_id = "id1"
    secret_name = "unicode-secret"

    result = _encrypt_secret(secret_value, kek_manager, secret_id, secret_name)

    decrypted_value = _decrypt_secret(
        result.encrypted_value, result.wrapped_dek, kek_manager, secret_id, secret_name
    )

    assert decrypted_value == secret_value


def test_rotate_secret_encryption():
    old_kek_manager = KEKManager(passphrase="old-passphrase")
    new_kek_manager = KEKManager(passphrase="new-passphrase")
    secret_value = "my-secret-key"
    secret_id = "id1"
    secret_name = "name1"

    encrypt_result = _encrypt_secret(secret_value, old_kek_manager, secret_id, secret_name)

    rotate_result = rotate_secret_encryption(
        encrypt_result.encrypted_value,
        encrypt_result.wrapped_dek,
        old_kek_manager,
        new_kek_manager,
    )

    assert rotate_result.encrypted_value == encrypt_result.encrypted_value
    assert rotate_result.wrapped_dek != encrypt_result.wrapped_dek

    decrypted_value = _decrypt_secret(
        rotate_result.encrypted_value,
        rotate_result.wrapped_dek,
        new_kek_manager,
        secret_id,
        secret_name,
    )

    assert decrypted_value == secret_value


def test_rotate_cannot_decrypt_with_old_kek():
    old_kek_manager = KEKManager(passphrase="old-passphrase")
    new_kek_manager = KEKManager(passphrase="new-passphrase")
    secret_value = "my-secret-key"
    secret_id = "id1"
    secret_name = "name1"

    encrypt_result = _encrypt_secret(secret_value, old_kek_manager, secret_id, secret_name)

    rotate_result = rotate_secret_encryption(
        encrypt_result.encrypted_value,
        encrypt_result.wrapped_dek,
        old_kek_manager,
        new_kek_manager,
    )

    with pytest.raises(MlflowException, match="Failed to decrypt secret"):
        _decrypt_secret(
            encrypt_result.encrypted_value,
            rotate_result.wrapped_dek,
            old_kek_manager,
            secret_id,
            secret_name,
        )


def test_rotate_with_wrong_old_kek_fails():
    wrong_kek_manager = KEKManager(passphrase="wrong-passphrase")
    new_kek_manager = KEKManager(passphrase="new-passphrase")
    correct_kek_manager = KEKManager(passphrase="correct-passphrase")
    secret_value = "my-secret-key"
    secret_id = "id1"
    secret_name = "name1"

    result = _encrypt_secret(secret_value, correct_kek_manager, secret_id, secret_name)

    with pytest.raises(MlflowException, match="Failed to rotate secret encryption"):
        rotate_secret_encryption(
            result.encrypted_value, result.wrapped_dek, wrong_kek_manager, new_kek_manager
        )


@pytest.mark.parametrize(
    "secrets",
    [
        [("secret1", "id1", "name1"), ("secret2", "id2", "name2"), ("secret3", "id3", "name3")],
        [("value1", "uuid-1", "api-key"), ("value2", "uuid-2", "password")],
    ],
)
def test_rotate_multiple_secrets(secrets):
    old_kek_manager = KEKManager(passphrase="old-passphrase")
    new_kek_manager = KEKManager(passphrase="new-passphrase")

    encrypted_secrets = []
    for secret_value, secret_id, secret_name in secrets:
        result = _encrypt_secret(secret_value, old_kek_manager, secret_id, secret_name)
        encrypted_secrets.append(
            (result.encrypted_value, result.wrapped_dek, secret_id, secret_name)
        )

    for encrypted_value, old_wrapped_dek, secret_id, secret_name in encrypted_secrets:
        rotate_result = rotate_secret_encryption(
            encrypted_value, old_wrapped_dek, old_kek_manager, new_kek_manager
        )

        original_value = next(s[0] for s in secrets if s[1] == secret_id and s[2] == secret_name)
        decrypted_value = _decrypt_secret(
            encrypted_value,
            rotate_result.wrapped_dek,
            new_kek_manager,
            secret_id,
            secret_name,
        )
        assert decrypted_value == original_value
