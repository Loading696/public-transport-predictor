"""Round-trip test for the NDTP layer (no emulator needed).

Builds synthetic handshake + realtime frames per the spec, parses them back,
checks decoded navigation values, CRC rejection and a live loopback
TCP exchange into NDTPReceiver's buffer.

Usage:
    py ml/test_ndtp.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ndtp import (  # noqa: E402
    build_handshake_frame,
    build_realtime_frame,
    decode_nav00,
    parse_cells,
    parse_handshake_body,
    parse_nph,
    parse_npl,
    verify_crc,
    NPH_SIZE,
    NDTPError,
)
from src.ndtp_server import NDTPReceiver  # noqa: E402

UNIT_ID = 1166336
TS = 1767670500
LON, LAT = 37.6173210, 55.7551234


def test_handshake() -> None:
    raw = build_handshake_frame(UNIT_ID, request_id=1)
    npl = parse_npl(raw[:15])
    assert npl.peer_address == UNIT_ID, npl
    nph = parse_nph(raw[15 : 15 + NPH_SIZE])
    assert (nph.service_id, nph.pkt_type) == (0, 100), nph
    verify_crc(npl, raw[15:])
    info = parse_handshake_body(raw[15 + NPH_SIZE :])
    assert info["proto"] == (6, 2) and info["peer_address"] == UNIT_ID, info
    print("[PASS] handshake round-trip")


def test_realtime_decode() -> None:
    raw = build_realtime_frame(
        UNIT_ID, 2, timestamp=TS, lon=LON, lat=LAT,
        valid=True, speed_avg=32.0, speed_max=45.0, course=159.0, altitude=165.0,
    )
    npl = parse_npl(raw[:15])
    verify_crc(npl, raw[15:])
    nph = parse_nph(raw[15 : 15 + NPH_SIZE])
    assert (nph.service_id, nph.pkt_type) == (1, 101), nph
    cells, nav = parse_cells(raw[15 + NPH_SIZE :])
    assert nav is not None and cells[0][:2] == (0, 0)
    assert abs(nav["lon"] - LON) < 1e-6, nav
    assert abs(nav["lat"] - LAT) < 1e-6, nav
    assert nav["location_valid"] is True, nav
    assert nav["speed_avg"] == 32.0 and nav["course"] == 159.0, nav
    assert nav["timestamp"] == TS, nav
    assert decode_nav00(cells[0][2]) == nav
    print("[PASS] realtime decode (coords/signs/valid/speed/course)")


def test_crc_reject() -> None:
    raw = bytearray(build_realtime_frame(UNIT_ID, 3, timestamp=TS, lon=LON, lat=LAT))
    raw[-1] ^= 0xFF
    npl = parse_npl(bytes(raw[:15]))
    try:
        verify_crc(npl, bytes(raw[15:]))
    except NDTPError:
        print("[PASS] corrupted frame rejected by CRC")
        return
    raise AssertionError("corrupted frame passed CRC")


def test_loopback() -> None:
    async def scenario() -> None:
        receiver = NDTPReceiver()
        await receiver.start("127.0.0.1", 0)
        port = receiver.bound_port
        assert port, "no bound port"
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        writer.write(build_handshake_frame(UNIT_ID, 1))
        writer.write(
            build_realtime_frame(UNIT_ID, 2, timestamp=TS, lon=LON, lat=LAT, speed_avg=32.0, course=159.0)
        )
        await writer.drain()
        writer.close()
        await writer.wait_closed()
        for _ in range(100):
            if receiver.stats["rows"] >= 1:
                break
            await asyncio.sleep(0.05)
        rows = await receiver.snapshot_rows()
        await receiver.stop()
        assert len(rows) == 1, rows
        row = rows[0]
        assert row["unit_id"] == UNIT_ID, row
        assert abs(row["lon"] - LON) < 1e-6 and abs(row["lat"] - LAT) < 1e-6, row
        assert row["location_valid"] is True and row["speed"] == 32.0, row

    asyncio.run(scenario())
    print("[PASS] TCP loopback handshake+realtime -> buffer")


def main() -> None:
    test_handshake()
    test_realtime_decode()
    test_crc_reject()
    test_loopback()
    print("ALL NDTP TESTS PASSED")


if __name__ == "__main__":
    main()
