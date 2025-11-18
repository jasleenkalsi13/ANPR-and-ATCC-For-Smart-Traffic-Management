import secrets

# hex (64 hex chars => 32 bytes)
secret_key = secrets.token_hex(32)
print(secret_key)

# or URL-safe base64-like string
secret_key2 = secrets.token_urlsafe(32)
print(secret_key2)