"""NDTP protocol parser (client -> server direction).

Implements the framing from dataset/docs/Emulator-and-Telematic-Packets-Specification.md:
  [ NPL 15 bytes ][ NPH 10 bytes ][ body ], little-endian, packed.

  - NPL: u16 signature (0x7E7E), u16 dataSize (NPH+body), u16 flags,
         u16 crc (CRC-16/Modbus over NPH+body, bytes swapped),
         u8 type (0x02 = NPH), u32 peerAddress (unitId), u16 requestId.
  - NPH: u16 serviceId, u16 type, u16 flags, u32 requestId.
  - Handshake NPH_SGC_CONN_REQUEST (service 0 / type 100): 18-byte body.
  - Realtime NPH_SND_REALTIME (service 1 / type 101): cell sequence
    [type u8][number u8][payload], G6CellNav00 (type 0, 26 bytes) first.

Only the navigation cell is decoded into telemetry rows; other known cells
are carried as raw payload, unknown cell types stop cell parsing.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from datetime import datetime, timezone

SIGNATURE = 0x7E7E
NPL_SIZE = 15
NPH_SIZE = 10
NPL_TYPE_NPH = 0x02

SERVICE_HANDSHAKE = 0
TYPE_CONN_REQUEST = 100
SERVICE_NAVDATA = 1
TYPE_REALTIME = 101

CELL_NAV00 = 0
NAV00_PAYLOAD = 26

CELL_SIZES = {0: 26, 2: 26, 8: 6, 10: 37, 16: 8, 15: 50}

_NPL_STRUCT = struct.Struct("<HHHHBIH")
_NPH_STRUCT = struct.Struct("<HHHI")
_HANDSHAKE_STRUCT = struct.Struct("<HHHIII")
_NAV00_STRUCT = struct.Struct("<III BBHHHHHBB")


class NDTPError(ValueError):
    """Malformed NDTP frame or failed integrity check."""


def crc16_modbus(data: bytes) -> int:
    crc = 0xFFFF
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = ((crc >> 1) ^ 0xA001) if (crc & 1) else (crc >> 1)
    return crc & 0xFFFF


def _swap16(value: int) -> int:
    return ((value & 0xFF) << 8) | ((value >> 8) & 0xFF)


@dataclass
class NPLHeader:
    signature: int
    data_size: int
    flags: int
    crc: int
    pkt_type: int
    peer_address: int
    request_id: int


@dataclass
class NPHHeader:
    service_id: int
    pkt_type: int
    flags: int
    request_id: int


@dataclass
class RealtimePacket:
    npl: NPLHeader
    nph: NPHHeader
    nav: dict | None = None
    cells: list = field(default_factory=list)


def parse_npl(raw: bytes) -> NPLHeader:
    if len(raw) != NPL_SIZE:
        raise NDTPError(f"NPL must be {NPL_SIZE} bytes, got {len(raw)}")
    signature, data_size, flags, crc, pkt_type, peer, request_id = _NPL_STRUCT.unpack(raw)
    if signature != SIGNATURE:
        raise NDTPError(f"bad signature 0x{signature:04X}")
    if pkt_type != NPL_TYPE_NPH:
        raise NDTPError(f"unsupported NPL type 0x{pkt_type:02X}")
    return NPLHeader(signature, data_size, flags, crc, pkt_type, peer, request_id)


def parse_nph(raw: bytes) -> NPHHeader:
    if len(raw) != NPH_SIZE:
        raise NDTPError(f"NPH must be {NPH_SIZE} bytes, got {len(raw)}")
    service_id, pkt_type, flags, request_id = _NPH_STRUCT.unpack(raw)
    return NPHHeader(service_id, pkt_type, flags, request_id)


def verify_crc(npl: NPLHeader, nph_and_body: bytes) -> None:
    expected = _swap16(crc16_modbus(nph_and_body))
    if expected != npl.crc:
        raise NDTPError(
            f"CRC mismatch for unit {npl.peer_address}: "
            f"got 0x{npl.crc:04X}, want 0x{expected:04X}"
        )


def decode_nav00(payload: bytes) -> dict:
    if len(payload) != NAV00_PAYLOAD:
        raise NDTPError(f"Nav00 payload must be {NAV00_PAYLOAD} bytes, got {len(payload)}")
    (ts, lon_raw, lat_raw, extra, bat, sp_avg, sp_max, course, track, alt, nsat, pdop) = (
        _NAV00_STRUCT.unpack(payload)
    )
    lon = lon_raw / 1e7 * (1.0 if (extra >> 6) & 1 else -1.0)
    lat = lat_raw / 1e7 * (1.0 if (extra >> 5) & 1 else -1.0)
    return {
        "timestamp": ts,
        "lon": lon,
        "lat": lat,
        "location_valid": bool((extra >> 7) & 1),
        "extra": extra,
        "bat_voltage_mv": bat * 20,
        "speed_avg": float(sp_avg),
        "speed_max": float(sp_max),
        "course": float(course),
        "track_m": track,
        "alt": float(alt),
        "nsat": nsat,
        "pdop": pdop,
    }


def parse_cells(body: bytes) -> tuple[list, dict | None]:
    """Split a realtime body into [(type, number, payload)] and decoded Nav00."""
    cells: list = []
    nav: dict | None = None
    pos = 0
    while pos + 2 <= len(body):
        cell_type, number = body[pos], body[pos + 1]
        size = CELL_SIZES.get(cell_type)
        if size is None:
            break
        end = pos + 2 + size
        if end > len(body):
            raise NDTPError(f"truncated cell type={cell_type} number={number}")
        payload = body[pos + 2 : end]
        cells.append((cell_type, number, payload))
        if cell_type == CELL_NAV00 and nav is None:
            nav = decode_nav00(payload)
        pos = end
    return cells, nav


def parse_realtime(npl: NPLHeader, nph: NPHHeader, body: bytes) -> RealtimePacket:
    if (nph.service_id, nph.pkt_type) != (SERVICE_NAVDATA, TYPE_REALTIME):
        raise NDTPError(f"not a realtime packet: service={nph.service_id} type={nph.pkt_type}")
    cells, nav = parse_cells(body)
    return RealtimePacket(npl=npl, nph=nph, nav=nav, cells=cells)


def parse_handshake_body(body: bytes) -> dict:
    if len(body) != 18:
        raise NDTPError(f"handshake body must be 18 bytes, got {len(body)}")
    high, low, flags, peer, max_size, _reserved = _HANDSHAKE_STRUCT.unpack(body)
    return {
        "proto": (high, low),
        "flags": flags,
        "peer_address": peer,
        "max_packet_size": max_size,
    }


def nav_to_telemetry(peer_address: int, nav: dict, tr_map: dict | None = None) -> dict:
    """Convert a decoded Nav00 into a TelemetryEvent-style row (naive UTC)."""
    tr_id = (tr_map or {}).get(peer_address, peer_address)
    event_time = datetime.fromtimestamp(nav["timestamp"], tz=timezone.utc).replace(tzinfo=None)
    return {
        "tr_id": int(tr_id),
        "unit_id": int(peer_address),
        "event_time": event_time,
        "location_valid": bool(nav["location_valid"]),
        "lon": float(nav["lon"]),
        "lat": float(nav["lat"]),
        "alt": float(nav["alt"]),
        "speed": float(nav["speed_avg"]),
        "heading": float(nav["course"]),
    }


def build_handshake_frame(unit_id: int, request_id: int = 1) -> bytes:
    body = _HANDSHAKE_STRUCT.pack(6, 2, 0, unit_id, 65535, 0)
    nph = _NPH_STRUCT.pack(SERVICE_HANDSHAKE, TYPE_CONN_REQUEST, 0x0001, request_id)
    npl = _NPL_STRUCT.pack(
        SIGNATURE, len(nph) + len(body), 0, _swap16(crc16_modbus(nph + body)),
        NPL_TYPE_NPH, unit_id, 0,
    )
    return npl + nph + body


def build_realtime_frame(
    unit_id: int,
    request_id: int,
    *,
    timestamp: int,
    lon: float,
    lat: float,
    valid: bool = True,
    speed_avg: float = 0.0,
    speed_max: float = 0.0,
    course: float = 0.0,
    altitude: float = 0.0,
) -> bytes:
    extra = (0x80 if valid else 0x00) | 0x60
    payload = _NAV00_STRUCT.pack(
        int(timestamp),
        int(round(abs(lon) * 1e7)),
        int(round(abs(lat) * 1e7)),
        extra,
        200,
        int(speed_avg),
        int(speed_max),
        int(course),
        0,
        int(altitude),
        8,
        1,
    )
    body = bytes([CELL_NAV00, 0]) + payload
    nph = _NPH_STRUCT.pack(SERVICE_NAVDATA, TYPE_REALTIME, 0x0001, request_id)
    npl = _NPL_STRUCT.pack(
        SIGNATURE, len(nph) + len(body), 0, _swap16(crc16_modbus(nph + body)),
        NPL_TYPE_NPH, unit_id, 0,
    )
    return npl + nph + body
