from fastapi import Header, HTTPException
import secrets

from app.config import settings


def verify_api_key(x_api_key: str | None = Header(default=None)):
    if not settings.API_KEY:
        raise HTTPException(
            status_code=500,
            detail="API key is not configured"
        )

    if x_api_key is None or not secrets.compare_digest(
        x_api_key,
        settings.API_KEY
    ):
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key"
        )

    return x_api_key