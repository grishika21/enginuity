"""Demo identity: HS256 JWTs carrying a user name and roles.

The orchestrator issues the token at login. It forwards the caller's token on every agent call,
and each agent checks it again against the roles its card allows (defence in depth).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import jwt

from .config import get_settings

ISSUER = "fielddesk-demo"
TOKEN_TTL_S = 8 * 3600


@dataclass(frozen=True)
class Identity:
    user: str
    roles: tuple[str, ...]
    token: str


class AuthError(Exception):
    pass


def issue(user: str, roles: list[str], ttl_s: int = TOKEN_TTL_S) -> str:
    now = int(time.time())
    claims = {"sub": user, "roles": roles, "iss": ISSUER, "iat": now, "exp": now + ttl_s}
    return jwt.encode(claims, get_settings().jwt_secret, algorithm="HS256")


def verify(token: str | None) -> Identity:
    if not token:
        raise AuthError("missing bearer token")
    try:
        claims = jwt.decode(token, get_settings().jwt_secret, algorithms=["HS256"], issuer=ISSUER)
    except jwt.PyJWTError as exc:
        raise AuthError("invalid or expired token") from exc
    roles = claims.get("roles") or []
    if not isinstance(roles, list):
        raise AuthError("invalid roles claim")
    return Identity(user=str(claims["sub"]), roles=tuple(str(r) for r in roles), token=token)


def bearer(header: str | None) -> str | None:
    if header and header.lower().startswith("bearer "):
        return header[7:].strip()
    return None


def allowed(roles: tuple[str, ...] | list[str], allowed_roles: list[str]) -> bool:
    return "admin" in roles or bool(set(roles) & set(allowed_roles))
