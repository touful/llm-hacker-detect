"""Passive classic-PCAP reader. No sockets, subprocesses or payload persistence."""

import re
import struct
from dataclasses import dataclass, field
from itertools import count, islice

MAGIC = {
    bytes.fromhex(h): (e, scale)
    for h, e, scale in (
        ("d4c3b2a1", "<", 1e6),
        ("a1b2c3d4", ">", 1e6),
        ("4d3cb2a1", "<", 1e9),
        ("a1b23c4d", ">", 1e9),
    )
}
HEADERS = set(
    "host user-agent accept accept-encoding accept-language connection content-type content-length cookie referer".split()
)
METHODS = set("GET HEAD POST PUT DELETE OPTIONS PATCH TRACE CONNECT".split())
TOKEN = re.compile(rb"[!#$%&'*+.^_`|~0-9A-Za-z-]+")
LIMIT = 64 * 1024


def _records(path, limit):
    with open(path, "rb") as f:
        h = f.read(24)
        if len(h) != 24 or h[:4] not in MAGIC:
            raise ValueError("expected classic PCAP")
        endian, scale = MAGIC[h[:4]]
        snaplen, link = struct.unpack_from(endian + "II", h, 16)
        link &= 0xFFFF
        if link not in (1, 113, 276):
            raise ValueError(f"unsupported link type: {link}")
        record = struct.Struct(endian + "IIII")
        for _ in islice(count(), limit):
            h = f.read(16)
            if not h:
                break
            if len(h) != 16:
                raise EOFError("truncated record header")
            sec, frac, incl, wire = record.unpack(h)
            if incl > min(wire, snaplen, 16 << 20):
                raise ValueError("invalid capture length")
            b = f.read(incl)
            if len(b) != incl:
                raise EOFError("truncated record")
            yield sec + frac / scale, wire, b, link


def _transport(b, link):
    off, idx = {1: (14, 12), 113: (16, 14), 276: (20, 0)}[link]
    proto = int.from_bytes(b[idx : idx + 2], "big")
    while proto in (0x8100, 0x88A8) and len(b) >= off + 4:
        proto = int.from_bytes(b[off + 2 : off + 4], "big")
        off += 4
    if proto == 0x0800 and len(b) >= off + 20 and b[off] >> 4 == 4:
        ihl = 4 * (b[off] & 15)
        if ihl < 20 or int.from_bytes(b[off + 6 : off + 8], "big") & 0x3FFF:
            return
        end = off + int.from_bytes(b[off + 2 : off + 4], "big")
        src, dst, proto = b[off + 12 : off + 16], b[off + 16 : off + 20], b[off + 9]
        off += ihl
    elif proto == 0x86DD and len(b) >= off + 40 and b[off] >> 4 == 6:
        end = off + 40 + int.from_bytes(b[off + 4 : off + 6], "big")
        src, dst, proto = b[off + 8 : off + 24], b[off + 24 : off + 40], b[off + 6]
        off += 40
    else:
        return
    if proto not in (6, 17) or min(len(b), end) < off + (20 if proto == 6 else 8):
        return
    sp, dp = struct.unpack_from("!HH", b, off)
    if proto == 17:
        return (src, sp), (dst, dp), proto, 0, 0, b""
    size = (b[off + 12] >> 4) * 4
    if size < 20 or off + size > min(len(b), end):
        return
    return (
        (src, sp),
        (dst, dp),
        proto,
        int.from_bytes(b[off + 4 : off + 8], "big"),
        b[off + 13],
        b[off + size : end],
    )


@dataclass
class Side:
    seq: int | None = None
    buf: bytearray = field(default_factory=bytearray)
    start: float = 0
    skip: int = 0
    closed: bool = False

    def feed(self, t, seq, payload, fid, req, rsp):
        if self.closed or not payload:
            return
        if self.seq is not None:
            gap = (seq - self.seq + 2**31) % 2**32 - 2**31
            if gap > 0:  # do not search arbitrary bytes after a capture gap
                self.closed = True
                self.buf.clear()
                return
            payload = payload[-gap:]
        self.seq = ((seq if self.seq is None else self.seq) + len(payload)) % 2**32
        while payload:
            n = min(self.skip, len(payload))
            self.skip -= n
            payload = payload[n:]
            if not payload:
                return
            if not self.buf:
                self.start = t
            self.buf.extend(payload)
            end = self.buf.find(b"\r\n\r\n")
            if end < 0 and len(self.buf) <= LIMIT and (self.buf[0:1].isalpha()):
                return
            if end < 0 or end + 4 > LIMIT:
                self.closed = True
                self.buf.clear()
                return
            lines = bytes(self.buf[:end]).split(b"\r\n")
            payload = bytes(self.buf[end + 4 :])
            self.buf.clear()
            fields = []
            for line in lines[1:]:
                k, sep, v = line.partition(b":")
                if not sep or not TOKEN.fullmatch(k):
                    self.closed = True
                    return
                fields.append((k.lower(), v.strip()))
            first = lines[0].split(b" ", 2)
            status = None
            if (
                len(first) >= 2
                and first[0] in (b"HTTP/1.0", b"HTTP/1.1")
                and re.fullmatch(rb"[1-5][0-9]{2}", first[1])
            ):
                status = int(first[1])
                rsp.append((self.start, fid, status))
            elif (
                len(first) == 3
                and TOKEN.fullmatch(first[0])
                and first[2] in (b"HTTP/1.1", b"HTTP/1.0")
            ):
                method = first[0].decode("ascii").upper()
                target, _, q = first[1].partition(b"?")
                headers = tuple(
                    k.decode("ascii") if k.decode("ascii") in HEADERS else "other"
                    for k, v in fields
                )
                ua = dict(fields).get(b"user-agent", b"").lower()
                client = (
                    "script"
                    if any(k in ua for k in (b"curl", b"python", b"wget", b"go-http"))
                    else "browser"
                    if b"mozilla/" in ua
                    else "other"
                )
                req.append(
                    (
                        self.start,
                        fid,
                        method if method in METHODS else "OTHER",
                        sum(bool(v) for v in target.split(b"/")),
                        sum(bool(v) for v in q.split(b"&")),
                        headers,
                        client,
                    )
                )
            else:
                self.closed = True
                return
            lengths = [v for k, v in fields if k == b"content-length"]
            if (
                any(k in (b"transfer-encoding", b"upgrade") for k, v in fields)
                or len(lengths) > 1
                or lengths
                and not re.fullmatch(rb"\d{1,18}", lengths[0])
            ):
                self.closed = True
                return
            if status and (status < 200 or status in (204, 304)):
                self.skip = 0
            elif lengths:
                self.skip = int(lengths[0])
            elif status:
                self.closed = True
                return  # close-delimited response body


def capture(path, max_packets=None):
    """Return TCP/UDP packet metadata, complete visible request heads and statuses."""
    packets, requests, responses = [], [], []
    flows, sides, syns = {}, {}, {}
    next_id = 1
    for t, wire, b, link in _records(path, max_packets):
        tr = _transport(b, link)
        if tr is None:
            continue
        a, baddr, proto, seq, flags, data = tr
        key = (min(a, baddr), max(a, baddr), proto)
        new_syn = flags & 2 and not flags & 16 and syns.get(key) != seq
        if new_syn:
            syns[key] = seq
        if key not in flows or new_syn:
            flows[key] = (next_id, a)
            next_id += 1
        fid, origin = flows[key]
        d = 1 if origin == a else -1
        packets.append((t, wire, fid, d))
        if proto == 6 and (data or flags & 2):
            side = sides.get((fid, d))
            if side is None:
                side = sides[fid, d] = Side()
            if flags & 2 and side.seq is None:
                side.seq = (seq + 1) % 2**32
            side.feed(
                t, (seq + bool(flags & 2)) % 2**32, data, fid, requests, responses
            )
    packets.sort(key=lambda p: p[0])
    requests.sort(key=lambda p: p[0])
    responses.sort(key=lambda p: p[0])
    return packets, requests, responses
