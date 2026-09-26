"""Where finished request records go."""

from __future__ import annotations

import logging

from gateway.telemetry.logging import log_fields
from gateway.telemetry.record import RequestRecord

logger = logging.getLogger("gateway.request")


class Telemetry:
    def emit(self, record: RequestRecord) -> None:
        log_fields(logger, "request", **record.to_dict())
