"""Cookie-based auth for the browser UI.

Shares the same JWT issuance as the API — issued by `/auth/jwt/login`, also
set on a httpOnly cookie at the UI `/login` form. So API tools (curl, future
MCP clients) keep working unchanged.

Unauthenticated UI requests are redirected to /login; HTMX-originated requests
get a 401 with `HX-Redirect` so the htmx client triggers the full redirect
instead of swapping in the login page mid-row.
"""

from typing import Annotated

from fastapi import Depends, HTTPException, Request, Response
from starlette.responses import RedirectResponse

from ..auth import decode_token
from ..config import settings


class _UnauthenticatedUI(Exception):
    pass


def cookie_user(request: Request) -> str:
    token = request.cookies.get(settings.cookie_name)
    if not token:
        raise _UnauthenticatedUI()
    try:
        return decode_token(token)
    except HTTPException:
        raise _UnauthenticatedUI() from None


def install_handler(app) -> None:
    """Register the FastAPI exception handler that turns _UnauthenticatedUI
    into a redirect for browsers or an HX-Redirect for HTMX."""
    @app.exception_handler(_UnauthenticatedUI)
    async def _on_unauth(request: Request, _: _UnauthenticatedUI):
        if request.headers.get("hx-request") == "true":
            return Response(status_code=401, headers={"HX-Redirect": "/login"})
        return RedirectResponse("/login", status_code=303)


UIUser = Annotated[str, Depends(cookie_user)]
