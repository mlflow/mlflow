import hashlib
import hmac
import os

_SALT_LENGTH = 16
_ITERATIONS = 600000
_HASH_METHOD = "sha256"


def generate_password_hash(password: str) -> str:
    salt = os.urandom(_SALT_LENGTH).hex()
    h = hashlib.pbkdf2_hmac(
        _HASH_METHOD, password.encode("utf-8"), salt.encode("utf-8"), _ITERATIONS
    )
    return f"pbkdf2:{_HASH_METHOD}:{_ITERATIONS}${salt}${h.hex()}"


def check_password_hash(pwhash: str, password: str) -> bool:
    if not pwhash.startswith("pbkdf2:"):
        return False
    method_part, _, rest = pwhash.partition("$")
    salt, _, hash_hex = rest.partition("$")
    parts = method_part.split(":")
    if len(parts) != 3:
        return False
    _, hash_method, iterations_str = parts
    try:
        iterations = int(iterations_str)
    except ValueError:
        return False
    h = hashlib.pbkdf2_hmac(hash_method, password.encode("utf-8"), salt.encode("utf-8"), iterations)
    return hmac.compare_digest(h.hex(), hash_hex)
