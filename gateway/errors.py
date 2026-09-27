"""Errors in the OpenAI error format: {"error": {"message", "type", "code", ...}}."""

from __future__ import annotations

from typing import Any

from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse


class GatewayError(Exception):
    def __init__(
        self,
        status_code: int,
        message: str,
        *,
        type: str = "gateway_error",
        code: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.message = message
        self.type = type
        self.code = code

    def body(self, request_id: str | None) -> dict[str, Any]:
        return {
            "error": {
                "message": self.message,
                "type": self.type,
                "code": self.code,
                "param": None,
                "request_id": request_id,
            }
        }


def _request_id(request: Request) -> str | None:
    return getattr(request.state, "request_id", None)


async def gateway_error_handler(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, GatewayError)
    return JSONResponse(exc.body(_request_id(request)), status_code=exc.status_code)


async def validation_error_handler(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, RequestValidationError)
    first = exc.errors()[0] if exc.errors() else {}
    where = ".".join(str(p) for p in first.get("loc", ()) if p != "body")
    message = f"{where}: {first.get('msg', 'invalid request')}" if where else "invalid request"
    err = GatewayError(400, message, type="invalid_request_error", code="invalid_request")
    return JSONResponse(err.body(_request_id(request)), status_code=400)
