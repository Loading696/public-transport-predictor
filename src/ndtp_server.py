"""Async TCP receiver for the NDTP emulator stream.

Accepts one connection per unitId, performs framing
([NPL 15][NPH+body = dataSize]), verifies CRC, decodes the Nav00 cell
and buffers TelemetryEvent-style rows. Rows are keyed by unitId;
pass tr_map={unitId: tr_id} when the vehicle mapping is known
(default: tr_id = unitId).

Buffered rows feed the same causal pipeline as CSV telemetry
(InferenceService.predict_frame cuts event_time <= T per point).
"""

from __future__ import annotations

import asyncio
from collections import deque
from typing import Any

from src.ndtp import (
    NPL_SIZE,
    NPH_SIZE,
    SERVICE_HANDSHAKE,
    SERVICE_NAVDATA,
    TYPE_CONN_REQUEST,
    TYPE_REALTIME,
    NDTPError,
    nav_to_telemetry,
    parse_handshake_body,
    parse_nph,
    parse_npl,
    parse_realtime,
    verify_crc,
)


class NDTPReceiver:
    def __init__(self, tr_map: dict[int, int] | None = None, max_rows: int = 200_000) -> None:
        self._tr_map = dict(tr_map or {})
        self._rows: deque[dict[str, Any]] = deque(maxlen=max_rows)
        self._lock = asyncio.Lock()
        self.stats = {
            "connections": 0,
            "handshakes": 0,
            "realtime": 0,
            "rows": 0,
            "crc_errors": 0,
            "frame_errors": 0,
        }
        self._server: asyncio.AbstractServer | None = None

    async def start(self, host: str = "0.0.0.0", port: int = 9201) -> asyncio.AbstractServer:
        self._server = await asyncio.start_server(self._handle, host, port)
        return self._server

    async def stop(self) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

    @property
    def bound_port(self) -> int | None:
        if self._server and self._server.sockets:
            return self._server.sockets[0].getsockname()[1]
        return None

    async def snapshot_rows(self) -> list[dict[str, Any]]:
        async with self._lock:
            return list(self._rows)

    def set_tr_map(self, tr_map: dict[int, int] | None) -> None:
        """Обновить маппинг unitId -> tr_id без перезапуска TCP-сервера."""
        self._tr_map = dict(tr_map or {})

    def status(self) -> dict[str, Any]:
        return {
            "enabled": self._server is not None,
            "port": self.bound_port,
            "buffered_rows": len(self._rows),
            **self.stats,
        }

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        self.stats["connections"] += 1
        try:
            while True:
                try:
                    npl_raw = await reader.readexactly(NPL_SIZE)
                except asyncio.IncompleteReadError:
                    break
                try:
                    npl = parse_npl(npl_raw)
                    payload = await reader.readexactly(npl.data_size)
                    nph = parse_nph(payload[:NPH_SIZE])
                    body = payload[NPH_SIZE:]
                    verify_crc(npl, payload)
                except (NDTPError, asyncio.IncompleteReadError) as exc:
                    if isinstance(exc, NDTPError) and "CRC" in str(exc):
                        self.stats["crc_errors"] += 1
                    else:
                        self.stats["frame_errors"] += 1
                    break
                key = (nph.service_id, nph.pkt_type)
                if key == (SERVICE_HANDSHAKE, TYPE_CONN_REQUEST):
                    parse_handshake_body(body)
                    self.stats["handshakes"] += 1
                elif key == (SERVICE_NAVDATA, TYPE_REALTIME):
                    packet = parse_realtime(npl, nph, body)
                    self.stats["realtime"] += 1
                    if packet.nav is not None:
                        row = nav_to_telemetry(
                            npl.peer_address, packet.nav, self._tr_map,
                            door_open=packet.door_open, cells=packet.cells,
                        )
                        # Детали дверей для витрины/отладки (не ломают схему telemetry).
                        if packet.doors:
                            row["doors"] = packet.doors
                        async with self._lock:
                            self._rows.append(row)
                        self.stats["rows"] += 1
                        if packet.door_open:
                            self.stats["door_open_frames"] = self.stats.get("door_open_frames", 0) + 1
                else:
                    self.stats["frame_errors"] += 1
        except (ConnectionResetError, BrokenPipeError):
            pass
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except (ConnectionResetError, BrokenPipeError):
                pass
