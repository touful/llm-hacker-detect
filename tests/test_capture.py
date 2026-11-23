"""Black-box tests for capture. All wire data is generated here, offline."""

import struct
import tempfile
import unittest
from pathlib import Path

from agenttrace.capture import capture


ETHERNET, SLL2 = 1, 276
CLIENT = (bytes([192, 0, 2, 29]), 49273)
SERVER = (bytes([192, 0, 2, 7]), 80)


def checksum(data):
    if len(data) % 2:
        data += b"\0"
    total = sum(word for (word,) in struct.iter_unpack("!H", data))
    while total >> 16:
        total = (total & 0xFFFF) + (total >> 16)
    return ~total & 0xFFFF


def datagram(payload=b"", *, tcp=True, reverse=False, seq=0, ack=0, flags=0x18):
    (src, sport), (dst, dport) = (SERVER, CLIENT) if reverse else (CLIENT, SERVER)
    protocol = 6 if tcp else 17
    if tcp:
        segment = (
            struct.pack("!HHIIBBHHH", sport, dport, seq, ack, 0x50, flags, 8192, 0, 0)
            + payload
        )
    else:
        segment = struct.pack("!4H", sport, dport, len(payload) + 8, 0) + payload
    offset = 16 if tcp else 6
    pseudo = src + dst + struct.pack("!BBH", 0, protocol, len(segment))
    check = checksum(pseudo + segment)
    segment = (
        segment[:offset] + struct.pack("!H", check or 0xFFFF) + segment[offset + 2 :]
    )
    header = struct.pack(
        "!BBHHHBBH4s4s", 0x45, 0, len(segment) + 20, 1, 0, 64, protocol, 0, src, dst
    )
    return header[:10] + struct.pack("!H", checksum(header)) + header[12:] + segment


def frame(ip, link):
    if link == SLL2:
        return struct.pack("!HHIHBB8s", 0x0800, 0, 2, 1, 0, 6, b"\x02\0\0\0\0\1") + ip
    assert link == ETHERNET
    return bytes.fromhex("0200000000010200000000020800") + ip


def pcap(records=(), link=ETHERNET, endian="<", nano=False):
    """records contains (microseconds since fixture start, link-layer frame)."""
    magic = 0xA1B23C4D if nano else 0xA1B2C3D4
    data = bytearray(struct.pack(endian + "IHHIIII", magic, 2, 4, 0, 0, 65535, link))
    for us, packet in records:
        sec, frac = divmod(us, 1_000_000)
        data.extend(
            struct.pack(
                endian + "IIII",
                100 + sec,
                frac * (1000 if nano else 1),
                len(packet),
                len(packet),
            )
        )
        data.extend(packet)
    return bytes(data)


class Conversation:
    """Contiguous TCP byte streams; each SYN consumes one sequence number."""

    def __init__(self, link=ETHERNET):
        self.link = link
        self.records = []
        self.next_seq = [18001, 61009]
        self.send(0, flags=0x02)
        self.send(25000, reverse=True, flags=0x12)
        self.send(50000, flags=0x10)

    def send(self, us, payload=b"", *, reverse=False, flags=0x18):
        ip = datagram(
            payload,
            reverse=reverse,
            seq=self.next_seq[reverse],
            ack=self.next_seq[not reverse] if flags & 0x10 else 0,
            flags=flags,
        )
        self.next_seq[reverse] += len(payload) + bool(flags & 2)
        self.records.append((us, frame(ip, self.link)))

    def bytes(self):
        return pcap(self.records, self.link)


class CaptureTests(unittest.TestCase):
    def parse(self, data, **kwargs):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "synthetic.pcap"
            path.write_bytes(data)
            result = capture(str(path), **kwargs)
        self.assertEqual(len(result), 3)
        for rows in result:
            self.assertIsInstance(rows, list)
        return result

    def test_http_metadata_and_privacy(self):
        stream = Conversation(SLL2)
        stream.send(
            150000,
            b"ZCANARY /uri-canary?query-canary HTTP/1.1\r\n"
            b"hOsT: host-canary.invalid\r\nX-Name-Canary: value-canary\r\nAc",
        )
        stream.send(
            340000,
            b"cept: */*\r\nX-Second-Canary: second-canary\r\n"
            b"ACCEPT: text/plain\r\nCookie: cookie-canary\r\n\r\n",
        )
        stream.send(560000, b"HTTP/1.", reverse=True)
        stream.send(
            870000,
            b"1 200 reason-canary\r\nContent-Length: 11\r\n\r\nbody-canary",
            reverse=True,
        )
        result = self.parse(stream.bytes())
        packets, requests, responses = result
        self.assertEqual(len(requests), 1)
        self.assertEqual(len(responses), 1)
        self.assertAlmostEqual(requests[0][0] - packets[0][0], 0.15)
        self.assertAlmostEqual(responses[0][0] - requests[0][0], 0.41)
        self.assertEqual(requests[0][2:5], ("OTHER", 1, 1))
        self.assertEqual(
            tuple(requests[0][5]),
            ("host", "other", "accept", "other", "accept", "cookie"),
        )
        public = repr(result).lower()
        for private in ("canary", "192.0.2.", str(CLIENT[1])):
            self.assertNotIn(private, public)

    def test_link_layers_flow_identity_and_direction(self):
        # UDP uses the same ports as TCP, but is a different conversation. Its
        # first observed side is the server. Port numbers do not set direction.
        for link in (ETHERNET, SLL2):
            for endian in ("<", ">"):
                for nano in (False, True):
                    with self.subTest(link=link, endian=endian, nano=nano):
                        stream = Conversation(link)
                        for us, reverse in ((900000, True), (1250000, False)):
                            packet = datagram(b"udp-canary", tcp=False, reverse=reverse)
                            stream.records.append((us, frame(packet, link)))
                        packets, requests, responses = self.parse(
                            pcap(stream.records, link, endian, nano)
                        )
                        self.assertEqual((requests, responses), ([], []))
                        self.assertEqual(len(packets), len(stream.records))
                        for row, (us, packet) in zip(packets, stream.records):
                            self.assertAlmostEqual(row[0] - packets[0][0], us / 1e6)
                            self.assertEqual(row[1], len(packet))
                        ids = [row[2] * row[3] for row in packets]
                        self.assertTrue(all(type(i) is int for i in ids))
                        self.assertGreater(ids[0], 0)
                        self.assertGreater(ids[3], 0)
                        self.assertNotEqual(ids[0], ids[3])
                        self.assertEqual(
                            ids, [ids[0], -ids[0], ids[0], ids[3], -ids[3]]
                        )

    def test_pcap_errors(self):
        header = pcap()
        record = struct.pack("<IIII", 100, 0, 12, 12)
        for case, data in (
            ("bad magic", b"JUNK" + header[4:]),
            ("pcapng", b"\x0a\x0d\x0d\x0a" + header[4:]),
            ("unsupported link type", pcap(link=147)),
            ("short global header", header[:19]),
            ("short record header", header + record[:14]),
            ("short frame", header + record + bytes(11)),
            (
                "length exceeds snaplen",
                header + struct.pack("<IIII", 100, 0, 65536, 65536) + bytes(65536),
            ),
        ):
            with self.subTest(case=case), self.assertRaises((ValueError, EOFError)):
                self.parse(data)

    def test_cap_stops_before_next_record(self):
        stream = Conversation()
        stream.send(100000, b"GET / HTTP/1.1\r\nHost: example.invalid\r\n\r\n")
        # A capped reader must not parse the intentionally partial next record.
        data = stream.bytes() + b"\xff"
        for n in (0, 1, 3, 4):
            with self.subTest(cap=n):
                packets, requests, _ = self.parse(data, max_packets=n)
                self.assertEqual(len(packets), n)
                self.assertEqual(len(requests), int(n == 4))

    def test_content_length_body_and_next_header_share_segment(self):
        fake = b"GET /not-a-request HTTP/1.1\r\nHost: fake.invalid\r\n\r\n"
        head = (
            b"POST /one HTTP/1.1\r\nHost: example.invalid\r\nContent-Length: "
            + str(len(fake)).encode("ascii")
            + b"\r\n\r\n"
        )
        stream = Conversation()
        stream.send(100000, head)
        stream.send(350000, fake[:17])
        stream.send(500000, fake[17:] + b"GE")
        stream.send(900000, b"T /two HTTP/1.1\r\nHost: example.invalid\r\n\r\n")
        packets, requests, _ = self.parse(stream.bytes())
        self.assertEqual([row[2] for row in requests], ["POST", "GET"])
        self.assertAlmostEqual(requests[0][0] - packets[0][0], 0.1)
        self.assertAlmostEqual(requests[1][0] - requests[0][0], 0.4)

    def test_status_line_inside_response_body_is_not_a_response(self):
        fake = b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\n\r\n"
        stream = Conversation()
        stream.send(
            100000,
            b"GET /one HTTP/1.1\r\nHost: example.invalid\r\n\r\n"
            b"GET /two HTTP/1.1\r\nHost: example.invalid\r\n\r\n",
        )
        stream.send(
            300000,
            b"HTTP/1.1 200 OK\r\nContent-Length: "
            + str(len(fake)).encode("ascii")
            + b"\r\n\r\n",
            reverse=True,
        )
        stream.send(400000, fake, reverse=True)
        stream.send(800000, b"HTTP/1.1 204 No Content\r\n\r\n", reverse=True)
        _, requests, responses = self.parse(stream.bytes())
        self.assertEqual(len(requests), 2)
        self.assertEqual(len(responses), 2)
        self.assertAlmostEqual(responses[1][0] - responses[0][0], 0.5)

    def test_chunk_data_is_not_a_request(self):
        stream = Conversation()
        body = b"GET /not-a-request HTTP/1.1\r\nHost: fake.invalid\r\n\r\n"
        stream.send(
            100000,
            b"POST / HTTP/1.1\r\nHost: example.invalid\r\n"
            b"Transfer-Encoding: chunked\r\n\r\n" + b"%x\r\n" % len(body),
        )
        stream.send(200000, body)
        stream.send(
            300000, b"\r\n0\r\n\r\nHEAD /next HTTP/1.1\r\nHost: example.invalid\r\n\r\n"
        )
        _, requests, _ = self.parse(stream.bytes())
        methods = [row[2] for row in requests]
        self.assertIn(methods, (["POST"], ["POST", "HEAD"]))

    def test_no_resynchronization_inside_unframed_bytes(self):
        for prefix in (b"not a start line\r\n\r\n", b"\x16\x03\x01\x00\x10opaque"):
            with self.subTest(prefix=prefix):
                stream = Conversation()
                stream.send(100000, prefix)
                stream.send(200000, b"GET /fake HTTP/1.1\r\nHost: fake.invalid\r\n\r\n")
                _, requests, _ = self.parse(stream.bytes())
                self.assertEqual(requests, [])

    def test_oversized_unterminated_header_is_not_salvaged(self):
        stream = Conversation()
        stream.send(100000, b"GET / HTTP/1.1\r\nHost: example.invalid\r\nX-Padding: ")
        # Exercise a bounded header accumulator without constructing a giant
        # pcap record. The apparent request starts inside the same field value.
        for i in range(8):
            stream.send(200000 + i, b"a" * 32768)
        stream.send(300000, b"GET /inside-field HTTP/1.1\r\nHost: fake.invalid\r\n\r\n")
        _, requests, _ = self.parse(stream.bytes())
        self.assertEqual(requests, [])

    def test_tcp_payload_ends_at_ip_length_not_end_of_frame(self):
        stream = Conversation()
        stream.send(100000, b"GET / HTTP/1.1\r\nHost: example.invalid\r\n")
        us, packet = stream.records[-1]
        # Ethernet padding is outside IPv4.total_length, not the end of the header.
        stream.records[-1] = us, packet + b"\r\n"
        _, requests, _ = self.parse(stream.bytes())
        self.assertEqual(requests, [])
