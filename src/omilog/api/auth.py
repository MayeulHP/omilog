from typing import Annotated

from fastapi import APIRouter, Form, HTTPException, status
from pydantic import BaseModel

from ..auth import authenticate, create_access_token

router = APIRouter(tags=["auth"])


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


@router.post("/auth/jwt/login", response_model=Token)
async def login(
    username: Annotated[str, Form()],
    password: Annotated[str, Form()],
) -> Token:
    if not authenticate(username, password):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid credentials")
    return Token(access_token=create_access_token(username))
