"""A small local stand-in for the USGS M2M JSON API plus a synthetic CORONA scan.

Only the endpoints corona_fetch.py uses are implemented, with the request and
response shapes documented at https://m2m.cr.usgs.gov/api/docs/json/. The
synthetic scan is rendered from a known ground->pixel homography, with a
checkerboard of 250 m cells and a bright disc at the target point, so tests can
check that the cut-out lands where it should.
"""

from __future__ import annotations

import io
import json
import tarfile
import threading
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import tifffile

import corona_fetch as cf


@dataclass
class SyntheticScene:
    entity_id: str
    target: tuple[float, float]  # lat, lon of the bright disc
    center: tuple[float, float]  # footprint centre
    length_m: float = 120_000.0  # E-W extent (cross-track, the long film axis)
    width_m: float = 12_000.0  # N-S extent
    rotate_deg: float = 8.0  # footprint rotation from E-W
    width_px: int = 12_000
    height_px: int = 1_200
    image_rotation: int = 0  # 0: TL=NW ; 2: rotated 180 (TL=SE)
    parts: int = 2
    available: bool = True
    acquired: str = "1969-07-21"
    corners: dict = field(default_factory=dict)

    def footprint(self):
        proj = cf.LocalProjection(*self.center)
        a = np.radians(self.rotate_deg)
        hx, hy = self.length_m / 2, self.width_m / 2
        local = {"NW": (-hx, hy), "NE": (hx, hy), "SE": (hx, -hy), "SW": (-hx, -hy)}
        out = {}
        for k, (x, y) in local.items():
            xr = x * np.cos(a) - y * np.sin(a)
            yr = x * np.sin(a) + y * np.cos(a)
            lat, lon = proj.inverse(xr, yr)
            out[k] = (float(lat), float(lon))
        self.corners = out
        return out

    def render(self, out_dir: Path) -> list[Path]:
        corners = self.footprint()
        proj = cf.LocalProjection(*self.target)
        ground = np.column_stack(proj.forward(
            [corners[c][0] for c in cf.CORNERS], [corners[c][1] for c in cf.CORNERS]))
        perm = [(self.image_rotation + i) % 4 for i in range(4)]
        W, H = self.width_px, self.height_px
        img_corners = np.array([[0, 0], [W, 0], [W, H], [0, H]], float)
        Hg2p = cf.fit_homography(ground[perm], img_corners)
        Hp2g = np.linalg.inv(Hg2p)
        img = np.empty((H, W), np.uint8)
        cols = np.arange(W) + 0.5
        for r0 in range(0, H, 200):
            rows = np.arange(r0, min(r0 + 200, H)) + 0.5
            U, V = np.meshgrid(cols, rows)
            gx, gy = cf.apply_homography(Hp2g, U, V)
            checker = ((np.floor(gx / 250) + np.floor(gy / 250)) % 2).astype(np.uint8)
            block = 60 + 60 * checker
            disc = gx ** 2 + gy ** 2 < 100.0 ** 2
            block[disc] = 250
            # A north-pointing bar from the disc, 300 m long, to check orientation.
            bar = (np.abs(gx) < 15) & (gy > 100) & (gy < 400)
            block[bar] = 200
            img[r0 : r0 + len(rows)] = block
        out_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        splits = np.linspace(0, W, self.parts + 1).astype(int)
        for i in range(self.parts):
            p = out_dir / f"{self.entity_id}_{'abcdefgh'[i]}.tif"
            tifffile.imwrite(p, img[:, splits[i] : splits[i + 1]])  # uncompressed, like USGS
            paths.append(p)
        self.ground_to_pixel = Hg2p
        return paths

    def archive_bytes(self, out_dir: Path) -> bytes:
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tf:
            for p in self.render(out_dir):
                tf.add(p, arcname=p.name)
        return buf.getvalue()

    def search_result(self) -> dict:
        c = self.corners or self.footprint()
        md = [
            {"fieldName": "Entity ID", "value": self.entity_id},
            {"fieldName": "Acquisition Date", "value": self.acquired},
            {"fieldName": "Mission", "value": "1107-2"},
            {"fieldName": "Camera Type", "value": "KH-4B"},
            {"fieldName": "Download Available", "value": "Y" if self.available else "N"},
        ]
        for k in cf.CORNERS:
            md.append({"fieldName": f"{k} Corner Lat dec", "value": f"{c[k][0]:.6f}"})
            md.append({"fieldName": f"{k} Corner Long dec", "value": f"{c[k][1]:.6f}"})
            # DMS variants are also present in real metadata; must be ignored in favour of dec.
            md.append({"fieldName": f"{k} Corner Lat", "value": "00°00'00\"N"})
            md.append({"fieldName": f"{k} Corner Long", "value": "000°00'00\"E"})
        ring = [[c[k][1], c[k][0]] for k in ("SW", "SE", "NE", "NW", "SW")]
        return {
            "entityId": self.entity_id,
            "displayId": self.entity_id,
            "temporalCoverage": {"startDate": self.acquired, "endDate": self.acquired},
            "spatialCoverage": {"type": "Polygon", "coordinates": [ring]},
            "browse": [],
            "metadata": md,
        }


class FakeM2M:
    def __init__(self, scenes: list[SyntheticScene], workdir: Path, username="user", token="tok"):
        self.scenes = {s.entity_id: s for s in scenes}
        self.workdir = workdir
        self.username, self.token = username, token
        self.api_key = "APIKEY123"
        self.calls: list[tuple[str, dict]] = []
        self.requests_by_label: dict[str, list[dict]] = {}
        self.retrieve_count: dict[str, int] = {}
        self.archives: dict[str, bytes] = {}
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), self._handler())
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def base(self) -> str:
        return f"http://127.0.0.1:{self.server.server_address[1]}"

    @property
    def api_url(self) -> str:
        return self.base + "/api/api/json/stable/"

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.server.shutdown()

    def endpoints_called(self) -> list[str]:
        return [c[0] for c in self.calls]

    # -- API logic --
    def handle(self, endpoint: str, payload: dict, api_key: str | None):
        self.calls.append((endpoint, payload))
        if endpoint == "login-token":
            if payload.get("username") == self.username and payload.get("token") == self.token:
                return self.api_key
            raise ApiError("AUTH_INVALID", "User credential verification failed")
        if api_key != self.api_key:
            raise ApiError("AUTH_UNAUTHROIZED", "missing or bad X-Auth-Token")
        if endpoint == "logout":
            return None
        if endpoint == "scene-search":
            sf = payload["sceneFilter"]["spatialFilter"]
            ll, ur = sf["lowerLeft"], sf["upperRight"]
            hits = []
            for s in self.scenes.values():
                c = s.corners or s.footprint()
                lats = [v[0] for v in c.values()]
                lons = [v[1] for v in c.values()]
                if (min(lats) <= ur["latitude"] and max(lats) >= ll["latitude"]
                        and min(lons) <= ur["longitude"] and max(lons) >= ll["longitude"]):
                    hits.append(s.search_result())
            start = payload.get("startingNumber", 1) - 1
            page = hits[start : start + payload.get("maxResults", 100)]
            nxt = start + len(page) + 1
            return {"results": page, "recordsReturned": len(page), "totalHits": len(hits),
                    "startingNumber": start + 1, "nextRecord": nxt if nxt <= len(hits) else None}
        if endpoint == "download-options":
            out = []
            for eid in payload["entityIds"]:
                s = self.scenes[eid]
                out.append({"id": "5e83a3efe0d4e2a0", "entityId": eid, "productName": "Browse",
                            "available": True, "filesize": 1000, "downloadSystem": "dds"})
                out.append({"id": "5e839ff8ea1b0ae4", "entityId": eid,
                            "productName": "Declass 1 Film Scan", "available": s.available,
                            "filesize": 900_000_000, "downloadSystem": "dds"})
            return out
        if endpoint == "download-request":
            label = payload["label"]
            self.requests_by_label[label] = payload["downloads"]
            preparing = [{"downloadId": 1000 + i, "eulaCode": None}
                         for i, _ in enumerate(payload["downloads"])]
            return {"availableDownloads": [], "preparingDownloads": preparing, "failed": [],
                    "newRecords": {}, "duplicateProducts": []}
        if endpoint == "download-retrieve":
            label = payload["label"]
            n = self.retrieve_count[label] = self.retrieve_count.get(label, 0) + 1
            dls = self.requests_by_label.get(label, [])
            if n < 2:  # staged on the second poll
                return {"available": [], "requested": [{"downloadId": 1000 + i} for i in range(len(dls))],
                        "queueSize": len(dls)}
            avail = [{"downloadId": 1000 + i, "entityId": d["entityId"], "productId": d["productId"],
                      "url": f"{self.base}/download/{d['entityId']}"} for i, d in enumerate(dls)]
            return {"available": avail, "requested": [], "queueSize": 0}
        if endpoint == "download-order-remove":
            return None
        raise ApiError("UNKNOWN_ENDPOINT", endpoint)

    def archive(self, eid: str) -> bytes:
        if eid not in self.archives:
            self.archives[eid] = self.scenes[eid].archive_bytes(self.workdir / "render" / eid)
        return self.archives[eid]

    def _handler(self):
        fake = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                ep = self.path.rstrip("/").rsplit("/", 1)[-1]
                body = json.loads(self.rfile.read(int(self.headers.get("content-length") or 0)) or b"{}")
                try:
                    data = fake.handle(ep, body, self.headers.get("X-Auth-Token"))
                    resp = {"requestId": 1, "version": "stable", "data": data,
                            "errorCode": None, "errorMessage": None}
                except ApiError as e:
                    resp = {"requestId": 1, "version": "stable", "data": None,
                            "errorCode": e.code, "errorMessage": e.msg}
                raw = json.dumps(resp).encode()
                self.send_response(200)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def _download(self, head: bool):
                eid = self.path.rsplit("/", 1)[-1]
                if not self.path.startswith("/download/") or eid not in fake.scenes:
                    self.send_response(404)
                    self.end_headers()
                    return
                data = fake.archive(eid)
                start = 0
                rng = self.headers.get("Range")
                if rng and rng.startswith("bytes="):
                    start = int(rng[6:].split("-")[0])
                    self.send_response(206)
                else:
                    self.send_response(200)
                self.send_header("content-disposition", f'attachment; filename="{eid}.tgz"')
                self.send_header("content-length", str(len(data) - start))
                self.end_headers()
                if not head:
                    self.wfile.write(data[start:])

            def do_GET(self):
                self._download(head=False)

        return H


class ApiError(Exception):
    def __init__(self, code, msg):
        super().__init__(msg)
        self.code, self.msg = code, msg
