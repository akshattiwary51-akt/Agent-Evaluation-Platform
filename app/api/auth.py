"""Optional HTTP Basic protection for everything except /health.

Set APP_PASSWORD (and optionally APP_USER, default "admin") to require credentials. When unset
the app is open, which is fine on localhost and NOT fine on a public network. This is a minimal
gate, not a user system: one shared credential, no rate limiting, and Basic auth must only be
used over HTTPS.
"""
import os
import secrets

from fastapi import HTTPException, Request
from fastapi.security.utils import get_authorization_scheme_param
import base64


def require_auth(request: Request) -> None:
    password = os.environ.get("APP_PASSWORD")
    if not password:
        return
    scheme, param = get_authorization_scheme_param(request.headers.get("Authorization"))
    ok = False
    if scheme.lower() == "basic" and param:
        try:
            user, _, supplied = base64.b64decode(param).decode().partition(":")
            ok = (secrets.compare_digest(user, os.environ.get("APP_USER", "admin"))
                  & secrets.compare_digest(supplied, password))
        except Exception:
            ok = False
    if not ok:
        raise HTTPException(401, "authentication required",
                            headers={"WWW-Authenticate": 'Basic realm="agent-eval"'})
