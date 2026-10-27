#!/usr/bin/env python3
"""Download declassified spy-satellite imagery (KH-9 HEXAGON, CORONA, ARGON)
around a set of coordinates.

For every point you give it, this script:

  1. logs in to the USGS EROS M2M API with an application token,
  2. searches the declassified datasets, in order of preference (default
     ``declassiii`` = KH-9 HEXAGON 1971-84, 0.6-1.2 m, then ``corona2`` =
     CORONA KH-4 1960-72, ~2 m, and ARGON, ~30 m), for frames whose footprint
     contains the point,
  3. downloads the full-resolution film scan of the best downloadable frame(s)
     (0.5-7 GB each; cached, so reruns are fast),
  4. finds the photo inside the scan, assembles multi-part scans, and maps the
     point onto the film with the USGS footprint corners and a panoramic camera
     model,
  5. writes the window around the point as an approximately georeferenced
     GeoTIFF (WGS84), a PNG preview, a wider context preview with the window
     drawn on it, and a JSON sidecar.

Accuracy
--------
The scans are not georeferenced; the only geolocation USGS supplies is the four
footprint corners. Measured errors: 1-5 km for KH-4, up to ~10 km for KH-9 near
the ends of its strip. The window therefore has a per-camera margin (1 km ARGON,
5 km KH-4, 10 km KH-9; override with ``--margin-m``) so the target is inside it,
not at its centre. Always check the context preview.

The film's orientation relative to north is not in the metadata. It is taken
from a match against USGS's georeferenced browse overlays when that is
conclusive, else from per-camera conventions verified on real scans (see
CAMERAS). Where no convention is known (KH-9 forward camera), every candidate
orientation is written.

Frames that USGS has not scanned yet cannot be downloaded; they are listed with
``downloadAvailable: false`` and can be ordered in EarthExplorer.

Credentials
-----------
Create an application token at https://ers.cr.usgs.gov/ (Profile ->
"Application Token") with M2M access, then either export

    M2M_USERNAME=<ers username>
    M2M_TOKEN=<application token>

or pass ``--username`` and enter the token at the prompt.

Examples
--------
    # what is available for a list of points in the 1970s, download nothing
    ./corona_fetch.py --points-file examples/morocco_points.csv \
        --start 1970-01-01 --end 1979-12-31 --dry-run

    # download and cut out the best 1970s frame for each point
    ./corona_fetch.py --points-file examples/morocco_points.csv \
        --start 1970-01-01 --end 1979-12-31 --out ./declass_morocco

    # CORONA only, one point
    ./corona_fetch.py --dataset corona2 --point 34.0209,-6.8416
"""

from __future__ import annotations

import argparse
import csv
import getpass
import html
import io
import json
import logging
import math
import os
import re
import shutil
import sys
import tarfile
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import requests
import tifffile
from PIL import Image, ImageDraw

log = logging.getLogger("corona_fetch")

API_URL = "https://m2m.cr.usgs.gov/api/api/json/stable/"
DEFAULT_DATASET = "corona2"
# Searched in this order of preference: KH-9 HEXAGON panoramic (Declass 3,
# 0.6-1.2 m, 1971-84), then CORONA/ARGON (Declass 1, 1960-72).
DEFAULT_DATASETS = "declassiii,corona2"
WMS_URL = "https://ims.cr.usgs.gov/wms.py"
EARTH_RADIUS_M = 6371008.8
CORNERS = ("NW", "NE", "SE", "SW")
TIFF_SUFFIXES = (".tif", ".tiff", ".ntf")
PREVIEW_MAX_PX = 4000  # PNG previews are downsampled to this; the GeoTIFF keeps full resolution


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class LocalProjection:
    """Equirectangular projection to metres around a reference point.

    Accurate to well under 1 % over a single CORONA frame (~15 x 200 km), which
    is far better than the corner-coordinate geolocation itself.
    """

    lat0: float
    lon0: float

    def forward(self, lat, lon):
        lat = np.asarray(lat, dtype=float)
        lon = np.asarray(lon, dtype=float)
        dlon = (lon - self.lon0 + 180.0) % 360.0 - 180.0
        x = EARTH_RADIUS_M * np.radians(dlon) * math.cos(math.radians(self.lat0))
        y = EARTH_RADIUS_M * np.radians(lat - self.lat0)
        return x, y

    def inverse(self, x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        lat = self.lat0 + np.degrees(y / EARTH_RADIUS_M)
        lon = self.lon0 + np.degrees(x / (EARTH_RADIUS_M * math.cos(math.radians(self.lat0))))
        lon = (lon + 180.0) % 360.0 - 180.0
        return lat, lon


def box_around(lat: float, lon: float, size_m: float) -> tuple[float, float, float, float]:
    """(min_lat, min_lon, max_lat, max_lon) of a size_m x size_m square."""
    half = size_m / 2.0
    dlat = math.degrees(half / EARTH_RADIUS_M)
    dlon = math.degrees(half / (EARTH_RADIUS_M * max(math.cos(math.radians(lat)), 1e-6)))
    return lat - dlat, lon - dlon, lat + dlat, lon + dlon


def fit_homography(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """3x3 H with dst ~ H @ src (homogeneous), from >= 4 point pairs."""
    src = np.asarray(src, dtype=float)
    dst = np.asarray(dst, dtype=float)
    if src.shape[0] < 4 or src.shape != dst.shape:
        raise ValueError("need at least 4 matching point pairs")
    rows, rhs = [], []
    for (x, y), (u, v) in zip(src, dst):
        rows.append([x, y, 1, 0, 0, 0, -u * x, -u * y])
        rhs.append(u)
        rows.append([0, 0, 0, x, y, 1, -v * x, -v * y])
        rhs.append(v)
    h, *_ = np.linalg.lstsq(np.array(rows), np.array(rhs), rcond=None)
    return np.append(h, 1.0).reshape(3, 3)


def apply_homography(H: np.ndarray, x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    w = H[2, 0] * x + H[2, 1] * y + H[2, 2]
    u = (H[0, 0] * x + H[0, 1] * y + H[0, 2]) / w
    v = (H[1, 0] * x + H[1, 1] * y + H[1, 2]) / w
    return u, v


def point_in_polygon(x: float, y: float, poly: np.ndarray) -> bool:
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            xi = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < xi:
                inside = not inside
    return inside


# Image corners in cyclic order: top-left, top-right, bottom-right, bottom-left.
IMAGE_CORNER_NAMES = ("TL", "TR", "BR", "BL")
NORM_CORNERS = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype=float)


@dataclass(frozen=True)
class Camera:
    """What we know about the camera behind a frame.

    pan_deg: half scan angle of a panoramic camera (0 for frame cameras).
    orientation: how the scan sits relative to the footprint corners when the
        USGS overlay cannot tell us; None if not yet verified for this camera.
    margin_m: typical error of the corner-based geolocation, used as the
        default crop margin.
    """

    name: str
    pan_deg: float = 0.0
    orientation: str | None = None
    margin_m: float = 1000.0


# Orientations verified on real scans: ARGON against the USGS overlay
# (DS09066A063MC071, r=0.74); KH-4 forward/aft by locating the Pentagon in
# DS1101-2157DF028 and DS1043-2157DA014, where the placement error was 1-5 km
# (hence the 5 km margin); KH-9 aft by matching the El Mansour Eddahbi
# reservoir in D3C1212-100087A023, which landed ~7-10 km off near the strip
# end (hence the 10 km margin).
# KH-9 forward is unverified, so every orientation is written for it.
CAMERAS = {
    "argon": Camera("ARGON (KH-5) frame camera", 0.0, "TL=SE,TR=SW", 1000.0),
    "kh4-F": Camera("KH-4 forward panoramic camera", 35.0, "TL=NW,TR=NE", 5000.0),
    "kh4-A": Camera("KH-4 aft panoramic camera", 35.0, "TL=SE,TR=SW", 5000.0),
    "kh9-F": Camera("KH-9 forward panoramic camera", 60.0, None, 10000.0),
    "kh9-A": Camera("KH-9 aft panoramic camera", 60.0, "TL=SE,TR=SW", 10000.0),
    # Declass 2. KH-7 GAMBIT is a strip camera and the KH-9 mapping camera a
    # frame camera: neither has panoramic distortion.
    "kh7": Camera("KH-7 GAMBIT strip camera", 0.0, None, 2000.0),
    "kh9-map": Camera("KH-9 mapping camera", 0.0, None, 2000.0),
    "unknown": Camera("unknown camera", 0.0, None, 1000.0),
}


def camera_of(scene: "Scene") -> Camera:
    """KH-4 ids look like DS1101-2157DF028, HEXAGON like D3C1217-100011F024 (the
    letter before the frame number is the camera), ARGON like DS09066A063MC071,
    Declass 2 like DZB00402700011H021001 (H = KH-7, L = KH-9 mapping camera)."""
    d = scene.display_id
    if d.startswith("DZB"):
        kind = str(scene.metadata.get("Camera Type") or "")
        if "KH-7" in kind or (not kind and len(d) > 14 and d[14] == "H"):
            return CAMERAS["kh7"]
        if "Mapping" in kind or (not kind and len(d) > 14 and d[14] == "L"):
            return CAMERAS["kh9-map"]
        return CAMERAS["unknown"]
    cam = str(scene.metadata.get("Camera Type") or scene.metadata.get("Camera") or "")[:1].upper()
    if not cam and len(d) > 4:
        cam = d[-4].upper()
    if d.startswith("DS1") and cam in ("A", "F"):
        return CAMERAS[f"kh4-{cam}"]
    if d.startswith("D3C") and cam in ("A", "F"):
        return CAMERAS[f"kh9-{cam}"]
    if d.startswith("DS0") and d[-5:-3] == "MC":
        return CAMERAS["argon"]
    return CAMERAS["unknown"]


@dataclass(frozen=True)
class FilmModel:
    """Scan pixels <-> normalised film coordinates in which the ground maps
    (approximately) projectively.

    For a frame camera that is just the frame rescaled to [-1, 1]. A panoramic
    camera records scan angle theta linearly along the strip, while the ground
    moves as tan(theta) along the strip and the across-strip scale grows as
    1/cos(theta) (the "bow-tie" footprint). Undoing both lets a homography
    fitted to the four footprint corners hold over the whole strip.

    ``gaps`` are (start, end) pixel ranges along the strip that hold no
    ground: the title block and film clips where USGS joined scan parts. They
    are cut out, so the photo on either side is treated as continuous.
    """

    frame: tuple[int, int, int, int]
    pan_deg: float = 0.0
    gaps: tuple[tuple[int, int], ...] = ()

    def _axes(self):
        x0, y0, x1, y1 = self.frame
        cut = sum(b - a for a, b in self.gaps)
        if x1 - x0 >= y1 - y0:
            hx, hy = (x1 - x0 - cut) / 2, (y1 - y0) / 2
            return x0 + hx, y0 + hy, hx, hy, True
        hx, hy = (x1 - x0) / 2, (y1 - y0 - cut) / 2
        return x0 + hx, y0 + hy, hx, hy, False

    def _cut(self, t):
        """Strip pixel -> gap-free strip pixel (a point in a gap goes to its start)."""
        t = np.asarray(t, dtype=float)
        out = t.copy()
        for a, b in self.gaps:
            out = out - np.clip(t - a, 0, b - a)
        return out

    def _uncut(self, t):
        out = np.array(t, dtype=float)
        for a, b in sorted(self.gaps):
            out = np.where(out >= a, out + (b - a), out)
        return out

    def in_gap(self, u, v):
        t = np.asarray(u if self._axes()[4] else v, dtype=float)
        hit = np.zeros(t.shape, dtype=bool)
        for a, b in self.gaps:
            hit |= (t >= a) & (t < b)
        return hit

    def to_norm(self, u, v):
        cx, cy, hx, hy, along_x = self._axes()
        if along_x:
            u = self._cut(u)
        else:
            v = self._cut(v)
        a = (np.asarray(u, dtype=float) - cx) / hx
        b = (np.asarray(v, dtype=float) - cy) / hy
        if self.pan_deg > 0:
            m = math.radians(self.pan_deg)
            t, c = (a, b) if along_x else (b, a)
            th = np.clip(t * m, -1.5, 1.5)
            t, c = np.tan(th) / math.tan(m), c * math.cos(m) / np.cos(th)
            a, b = (t, c) if along_x else (c, t)
        return a, b

    def from_norm(self, a, b):
        cx, cy, hx, hy, along_x = self._axes()
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        if self.pan_deg > 0:
            m = math.radians(self.pan_deg)
            t, c = (a, b) if along_x else (b, a)
            th = np.arctan(t * math.tan(m))
            t, c = th / m, c * np.cos(th) / math.cos(m)
            a, b = (t, c) if along_x else (c, t)
        u, v = cx + a * hx, cy + b * hy
        return (self._uncut(u), v) if along_x else (u, self._uncut(v))


class GroundToScan:
    """Local ground metres <-> scan pixels for one orientation."""

    def __init__(self, ground_xy: np.ndarray, perm, film: FilmModel):
        self.film = film
        self.H = fit_homography(ground_xy[perm], NORM_CORNERS)
        self.Hinv = np.linalg.inv(self.H)

    def to_px(self, x, y):
        return self.film.from_norm(*apply_homography(self.H, x, y))

    def to_ground(self, u, v):
        return apply_homography(self.Hinv, *self.film.to_norm(u, v))


def orientation_candidates(ground_xy: np.ndarray, width: int, height: int):
    """Plausible ways the film scan may sit relative to the ground footprint.

    ``ground_xy`` holds the NW, NE, SE, SW footprint corners (local metres, in
    that cyclic order). The scan can be rotated and/or mirrored relative to
    north, which the metadata does not record. Of the 8 dihedral assignments
    of image corners to ground corners, we keep the 4 that put the image's long
    axis on the footprint's long axis (all 8 when either is close to square,
    e.g. ARGON frames), ordered so the "north-up, unmirrored" reading comes
    first.

    Returns a list of (name, permutation) where permutation[i] is the index of
    the ground corner that sits at image corner i (TL, TR, BR, BL).
    """
    cands = []
    image_aspect = max(width, height) / max(min(width, height), 1)
    for rot in range(4):
        for mirror in (False, True):
            if mirror:
                perm = [(rot - i) % 4 for i in range(4)]
            else:
                perm = [(rot + i) % 4 for i in range(4)]
            # Image TL->TR edge is the "width" edge.
            g = ground_xy[perm]
            top = np.linalg.norm(g[1] - g[0]) + np.linalg.norm(g[2] - g[3])
            side = np.linalg.norm(g[3] - g[0]) + np.linalg.norm(g[2] - g[1])
            ground_aspect = max(top, side) / max(min(top, side), 1e-9)
            elongated = image_aspect > 1.5 and ground_aspect > 1.5
            if elongated and (width >= height) != (top >= side):
                continue
            name = f"TL={CORNERS[perm[0]]},TR={CORNERS[perm[1]]}"
            cands.append((name, perm, rot, mirror))
    # Prefer no mirroring, then the smallest rotation.
    cands.sort(key=lambda c: (c[3], c[2]))
    return [(name, perm) for name, perm, _, _ in cands]


# --------------------------------------------------------------------------- #
# Scene metadata
# --------------------------------------------------------------------------- #


def parse_coordinate(value) -> float | None:
    """Parse decimal degrees or DMS strings like 37°12'34.5"N / 371234N."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    s = html.unescape(str(value)).strip()  # USGS sends e.g. 37&deg;48'39.60"N
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        pass
    hemi = 1.0
    m = re.search(r"(?<![A-Za-z])([NSEW])(?![A-Za-z])", s, re.I)
    if m:
        if m.group(1).upper() in "SW":
            hemi = -1.0
        s = s.replace(m.group(0), "")
    if s.strip().startswith("-"):
        hemi = -hemi
        s = s.replace("-", "", 1)
    nums = re.findall(r"\d+(?:\.\d+)?", s)
    if len(nums) == 1 and len(nums[0].split(".")[0]) >= 5:
        # Packed DDMMSS / DDDMMSS
        packed = nums[0]
        ip, _, frac = packed.partition(".")
        sec = float(ip[-2:] + ("." + frac if frac else ""))
        mins = float(ip[-4:-2])
        deg = float(ip[:-4])
        return hemi * (deg + mins / 60 + sec / 3600)
    if not nums:
        return None
    parts = [float(n) for n in nums[:3]] + [0.0, 0.0]
    return hemi * (parts[0] + parts[1] / 60 + parts[2] / 3600)


@dataclass
class Scene:
    entity_id: str
    display_id: str
    acquired: str
    corners: dict[str, tuple[float, float]]  # corner -> (lat, lon)
    metadata: dict[str, str] = field(default_factory=dict)
    browse_url: str | None = None
    dataset: str = ""
    # CORONA frames whose WMS overlays stand in as a georeferenced reference
    # when the scene has no overlay of its own (HEXAGON).
    reference_layers: list[str] = field(default_factory=list)
    raw: dict = field(default_factory=dict, repr=False)

    @property
    def center(self) -> tuple[float, float]:
        lats = [c[0] for c in self.corners.values()]
        lons = [c[1] for c in self.corners.values()]
        return float(np.mean(lats)), float(np.mean(lons))

    def projection(self) -> LocalProjection:
        return LocalProjection(*self.center)

    def ground_xy(self, proj: LocalProjection) -> np.ndarray:
        lat = [self.corners[c][0] for c in CORNERS]
        lon = [self.corners[c][1] for c in CORNERS]
        x, y = proj.forward(lat, lon)
        return np.column_stack([x, y])

    def contains(self, lat: float, lon: float) -> bool:
        proj = self.projection()
        x, y = proj.forward(lat, lon)
        return point_in_polygon(float(x), float(y), self.ground_xy(proj))

    def download_available(self) -> bool | None:
        for k, v in self.metadata.items():
            if "downloadavailable" in k.lower().replace(" ", ""):  # "Down Load Available"
                if isinstance(v, bool):
                    return v
                return str(v).strip().upper().startswith(("Y", "T"))
        opts = self.raw.get("options") or {}
        if isinstance(opts.get("download"), bool):
            return opts["download"]
        return None


def _metadata_dict(result: dict) -> dict[str, str]:
    md = result.get("metadata") or []
    out = {}
    if isinstance(md, list):
        for item in md:
            name = item.get("fieldName")
            if name:
                out[name] = item.get("value")
    elif isinstance(md, dict):
        out.update(md)
    return out


def _corners_from_metadata(md: dict[str, str]) -> dict[str, tuple[float, float]] | None:
    found: dict[str, dict[str, tuple[int, float]]] = {c: {} for c in CORNERS}
    for key, value in md.items():
        # "Cor[nm]er": USGS sends "NW Cormer Lat dec".
        m = re.match(r"^\s*(NW|NE|SE|SW)\s+Cor[nm]er\s+(Lat|Long|Lon)\w*\s*(dec)?", key, re.I)
        if not m:
            continue
        corner = m.group(1).upper()
        axis = "lat" if m.group(2).lower() == "lat" else "lon"
        priority = 0 if m.group(3) else 1  # prefer the decimal fields
        val = parse_coordinate(value)
        if val is None:
            continue
        prev = found[corner].get(axis)
        if prev is None or priority < prev[0]:
            found[corner][axis] = (priority, val)
    if all("lat" in found[c] and "lon" in found[c] for c in CORNERS):
        return {c: (found[c]["lat"][1], found[c]["lon"][1]) for c in CORNERS}
    return None


def _corners_from_polygon(geom: dict) -> dict[str, tuple[float, float]] | None:
    try:
        ring = geom["coordinates"][0]
    except (KeyError, IndexError, TypeError):
        return None
    pts = [(float(p[1]), float(p[0])) for p in ring]
    if len(pts) > 1 and pts[0] == pts[-1]:
        pts = pts[:-1]
    if len(pts) < 4:
        return None
    lat0 = float(np.mean([p[0] for p in pts]))
    lon0 = float(np.mean([p[1] for p in pts]))
    proj = LocalProjection(lat0, lon0)
    x, y = proj.forward([p[0] for p in pts], [p[1] for p in pts])
    scores = {"NW": -x + y, "NE": x + y, "SE": x - y, "SW": -x - y}
    return {c: pts[int(np.argmax(s))] for c, s in scores.items()}


def scene_from_result(result: dict) -> Scene | None:
    md = _metadata_dict(result)
    corners = _corners_from_metadata(md) or _corners_from_polygon(result.get("spatialCoverage") or {})
    if corners is None:
        log.warning("scene %s has no usable footprint; skipping", result.get("entityId"))
        return None
    temporal = result.get("temporalCoverage") or {}
    acquired = (
        md.get("Acquisition Date")
        or temporal.get("startDate")
        or result.get("acquisitionDate")
        or ""
    )
    browse = None
    for b in result.get("browse") or []:
        if b.get("browsePath"):
            browse = b["browsePath"]
            break
    return Scene(
        entity_id=str(result.get("entityId")),
        display_id=str(result.get("displayId") or result.get("entityId")),
        acquired=str(acquired)[:10],
        corners=corners,
        metadata={k: v for k, v in md.items() if v not in (None, "")},
        browse_url=browse,
        raw=result,
    )


# --------------------------------------------------------------------------- #
# M2M API client
# --------------------------------------------------------------------------- #


class M2MError(RuntimeError):
    pass


class M2MClient:
    def __init__(self, api_url: str = API_URL, timeout: float = 120.0):
        self.api_url = api_url.rstrip("/") + "/"
        self.timeout = timeout
        self.session = requests.Session()
        self.api_key: str | None = None

    def _post(self, endpoint: str, payload: dict | None = None, retries: int = 3):
        url = self.api_url + endpoint
        headers = {"X-Auth-Token": self.api_key} if self.api_key else {}
        for attempt in range(retries):
            try:
                r = self.session.post(url, json=payload or {}, headers=headers, timeout=self.timeout)
            except requests.RequestException as e:
                if attempt == retries - 1:
                    raise M2MError(f"{endpoint}: {e}") from e
                time.sleep(2 ** attempt)
                continue
            if r.status_code in (429, 502, 503, 504) and attempt < retries - 1:
                time.sleep(2 ** (attempt + 1))
                continue
            try:
                body = r.json()
            except ValueError:
                raise M2MError(f"{endpoint}: HTTP {r.status_code}, non-JSON response: {r.text[:200]}")
            if body.get("errorCode"):
                raise M2MError(f"{endpoint}: {body['errorCode']}: {body.get('errorMessage')}")
            if r.status_code >= 400:
                raise M2MError(f"{endpoint}: HTTP {r.status_code}")
            return body.get("data")
        raise M2MError(f"{endpoint}: gave up after {retries} attempts")

    # -- auth --
    def login(self, username: str, token: str) -> None:
        self.api_key = self._post("login-token", {"username": username, "token": token})
        if not self.api_key:
            raise M2MError("login-token returned no API key")

    def logout(self) -> None:
        if self.api_key:
            try:
                self._post("logout", retries=1)
            except M2MError as e:
                log.debug("logout failed: %s", e)
            self.api_key = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.logout()

    # -- search --
    def scene_search(
        self,
        dataset: str,
        bbox: tuple[float, float, float, float],
        start: str | None = None,
        end: str | None = None,
        max_results: int = 100,
    ) -> list[dict]:
        min_lat, min_lon, max_lat, max_lon = bbox
        scene_filter: dict = {
            "spatialFilter": {
                "filterType": "mbr",
                "lowerLeft": {"latitude": min_lat, "longitude": min_lon},
                "upperRight": {"latitude": max_lat, "longitude": max_lon},
            }
        }
        if start or end:
            scene_filter["acquisitionFilter"] = {
                "start": start or "1959-01-01",
                "end": end or "1990-12-31",
            }
        results: list[dict] = []
        starting = 1
        while len(results) < max_results:
            data = self._post(
                "scene-search",
                {
                    "datasetName": dataset,
                    "maxResults": min(100, max_results - len(results)),
                    "startingNumber": starting,
                    "metadataType": "full",
                    "sceneFilter": scene_filter,
                },
            ) or {}
            batch = data.get("results") or []
            results.extend(batch)
            nxt = data.get("nextRecord")
            total = data.get("totalHits") or 0
            if not batch or not nxt or nxt > total or nxt <= starting:
                break
            starting = nxt
        return results

    # -- download --
    def download_options(self, dataset: str, entity_ids: list[str]) -> list[dict]:
        return self._post("download-options", {"datasetName": dataset, "entityIds": entity_ids}) or []

    def download_request(self, downloads: list[dict], label: str) -> dict:
        return self._post("download-request", {"downloads": downloads, "label": label}) or {}

    def download_retrieve(self, label: str) -> dict:
        return self._post("download-retrieve", {"label": label}) or {}

    def download_label_remove(self, label: str) -> None:
        try:
            self._post("download-order-remove", {"label": label}, retries=1)
        except M2MError as e:
            log.debug("download-order-remove failed: %s", e)

    def resolve_download_urls(
        self, downloads: list[dict], label: str, poll_s: float = 15.0, timeout_s: float = 1800.0
    ) -> dict[str, str]:
        """Request downloads and wait until each has a URL. Returns entityId -> url."""
        req = self.download_request(downloads, label)
        for f in req.get("failed") or []:
            log.warning("download request failed: %s", f)
        id_to_entity: dict[str, str] = {}
        urls: dict[str, str] = {}
        for d in (req.get("availableDownloads") or []) + (req.get("preparingDownloads") or []):
            if d.get("entityId"):
                id_to_entity[str(d.get("downloadId"))] = d["entityId"]
        for d in req.get("availableDownloads") or []:
            ent = d.get("entityId") or id_to_entity.get(str(d.get("downloadId")))
            if d.get("url") and ent:
                urls[ent] = d["url"]
        wanted = {d["entityId"] for d in downloads}
        deadline = time.monotonic() + timeout_s
        while not wanted <= set(urls):
            if time.monotonic() > deadline:
                log.warning("timed out waiting for downloads: %s", sorted(wanted - set(urls)))
                break
            data = self.download_retrieve(label)
            for d in data.get("available") or []:
                ent = d.get("entityId") or id_to_entity.get(str(d.get("downloadId")))
                if ent in wanted and d.get("url"):
                    urls[ent] = d["url"]
            if wanted <= set(urls):
                break
            log.info("waiting for %d download(s) to be staged...", len(wanted - set(urls)))
            time.sleep(poll_s)
        return urls


def pick_product(options: list[dict], prefer: str | None = None) -> dict | None:
    """Choose the full-resolution scan from a scene's download options."""
    usable = [o for o in options if o.get("available") and o.get("id")]
    if prefer:
        pref = [o for o in usable if prefer.lower() in str(o.get("productName", "")).lower()]
        if pref:
            usable = pref
    # Avoid browse / metadata-only products when something better exists.
    real = [o for o in usable if not re.search(r"browse|metadata|kml", str(o.get("productName", "")), re.I)]
    usable = real or usable
    if not usable:
        return None
    return max(usable, key=lambda o: float(o.get("filesize") or 0))


# --------------------------------------------------------------------------- #
# Files
# --------------------------------------------------------------------------- #


def http_download(url: str, dest_dir: Path, timeout: float = 120.0) -> Path:
    """Stream ``url`` into ``dest_dir`` (one download per directory).

    A finished file in ``dest_dir`` is reused, and an interrupted download is
    resumed with an HTTP Range request.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    done_files = [p for p in dest_dir.iterdir() if p.is_file() and p.name != "_partial"]
    if done_files:
        log.info("already downloaded: %s", done_files[0])
        return done_files[0]
    part = dest_dir / "_partial"
    have = part.stat().st_size if part.exists() else 0
    headers = {"Range": f"bytes={have}-"} if have else {}
    with requests.get(url, stream=True, timeout=timeout, headers=headers) as r:
        if r.status_code == 416:  # nothing left to fetch, or server rejects the range
            part.unlink(missing_ok=True)
            return http_download(url, dest_dir, timeout)
        r.raise_for_status()
        cd = r.headers.get("content-disposition", "")
        m = re.search(r'filename\*?=(?:UTF-8\'\')?"?([^";]+)"?', cd, re.I)
        name = m.group(1) if m else Path(requests.utils.urlparse(r.url).path).name
        name = Path(name or "download.bin").name  # never trust a path from the server
        if r.status_code != 206:
            have = 0
        total = have + int(r.headers.get("content-length") or 0)
        done = have
        last = time.monotonic()
        with open(part, "ab" if have else "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                done += len(chunk)
                if time.monotonic() - last > 10:
                    pct = f" ({100 * done / total:.0f}%)" if total > have else ""
                    log.info("  %s: %.1f MB%s", name, done / 1e6, pct)
                    last = time.monotonic()
    if total > have and done != total:
        raise IOError(f"download of {name} incomplete: {done} of {total} bytes; rerun to resume")
    final = dest_dir / name
    part.replace(final)
    return final


def _safe_members_tar(tf: tarfile.TarFile, dest: Path):
    dest = dest.resolve()
    for m in tf.getmembers():
        if not (m.isfile() or m.isdir()):
            continue
        target = (dest / m.name).resolve()
        if dest != target and dest not in target.parents:
            raise ValueError(f"unsafe path in archive: {m.name}")
        yield m


def extract_tiffs(archive: Path, dest: Path) -> list[Path]:
    """Extract an archive (or accept a bare TIFF) and return TIFF paths, sorted."""
    dest.mkdir(parents=True, exist_ok=True)
    if archive.suffix.lower() in TIFF_SUFFIXES:
        return [archive]
    if tarfile.is_tarfile(archive):
        with tarfile.open(archive) as tf:
            members = list(_safe_members_tar(tf, dest))
            if hasattr(tarfile, "data_filter"):
                tf.extractall(dest, members=members, filter="data")
            else:
                tf.extractall(dest, members=members)
    elif zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as zf:
            root = dest.resolve()
            for info in zf.infolist():
                target = (root / info.filename).resolve()
                if root != target and root not in target.parents:
                    raise ValueError(f"unsafe path in archive: {info.filename}")
            zf.extractall(dest)
    else:
        raise ValueError(f"don't know how to unpack {archive}")
    tiffs = sorted(p for p in dest.rglob("*") if p.suffix.lower() in TIFF_SUFFIXES)
    if not tiffs:
        raise ValueError(f"no TIFFs found in {archive}")
    return tiffs


def _open_array(path: Path):
    """Memory-map an uncompressed TIFF; fall back to a full read."""
    try:
        return tifffile.memmap(path, mode="r")
    except Exception:
        log.info("%s is not memory-mappable (compressed/tiled?); reading it fully", path.name)
        return tifffile.imread(path)


class StripMosaic:
    """Treat the TIFF parts of one CORONA scan (``*_a.tif``, ``*_b.tif``, ...) as
    one image. Parts are joined end to end along the strip's long axis."""

    def __init__(self, paths: list[Path]):
        self.paths = paths
        self.arrays = [_open_array(p) for p in paths]
        shapes = [a.shape[:2] for a in self.arrays]
        hs = {s[0] for s in shapes}
        ws = {s[1] for s in shapes}
        if len(self.arrays) == 1 or len(hs) == 1 and (len(ws) != 1 or shapes[0][1] >= shapes[0][0]):
            self.axis = 1  # side by side
        elif len(ws) == 1:
            self.axis = 0  # stacked
        else:
            # Heights differ slightly between scanned segments: join along x
            # and pad to the tallest.
            self.axis = 1
        if self.axis == 1:
            self.height = max(s[0] for s in shapes)
            self.offsets = np.cumsum([0] + [s[1] for s in shapes]).tolist()
            self.width = self.offsets[-1]
        else:
            self.width = max(s[1] for s in shapes)
            self.offsets = np.cumsum([0] + [s[0] for s in shapes]).tolist()
            self.height = self.offsets[-1]
        self.extra_shape = self.arrays[0].shape[2:]
        self.dtype = self.arrays[0].dtype

    def read(self, x0: int, y0: int, x1: int, y1: int, step: int = 1) -> np.ndarray:
        """Read [y0:y1:step, x0:x1:step], clamped to the image; outside is 0."""
        x0, x1 = sorted((int(x0), int(x1)))
        y0, y1 = sorted((int(y0), int(y1)))
        out_h = max(0, -(-(y1 - y0) // step))
        out_w = max(0, -(-(x1 - x0) // step))
        out = np.zeros((out_h, out_w) + self.extra_shape, dtype=self.dtype)
        for i, arr in enumerate(self.arrays):
            if self.axis == 1:
                ox, oy = self.offsets[i], 0
            else:
                ox, oy = 0, self.offsets[i]
            ah, aw = arr.shape[:2]
            # Intersection in mosaic coords
            ix0, ix1 = max(x0, ox), min(x1, ox + aw)
            iy0, iy1 = max(y0, oy), min(y1, oy + ah)
            if ix0 >= ix1 or iy0 >= iy1:
                continue
            # Align to the global sampling grid
            ix0 += (-(ix0 - x0)) % step
            iy0 += (-(iy0 - y0)) % step
            if ix0 >= ix1 or iy0 >= iy1:
                continue
            block = np.asarray(arr[iy0 - oy : iy1 - oy : step, ix0 - ox : ix1 - ox : step])
            r0, c0 = (iy0 - y0) // step, (ix0 - x0) // step
            out[r0 : r0 + block.shape[0], c0 : c0 + block.shape[1]] = block
        return out

    def overview(self, long_side: int = 3200) -> tuple[np.ndarray, int]:
        """Decimated grey copy of the whole scan (cached) and its step."""
        step = max(1, max(self.width, self.height) // long_side)
        cache = self.__dict__.setdefault("_overviews", {})
        if step not in cache:
            a = self.read(0, 0, self.width, self.height, step=step).astype(np.float32)
            cache[step] = a.mean(axis=2) if a.ndim == 3 else a
        return cache[step], step


def _largest_component(mask: np.ndarray) -> np.ndarray:
    """Boolean mask of the largest 4-connected True region."""
    labels = np.zeros(mask.shape, dtype=np.int32)
    best, best_size, n = 0, 0, 0
    h, w = mask.shape
    for sy, sx in zip(*np.nonzero(mask)):
        if labels[sy, sx]:
            continue
        n += 1
        labels[sy, sx] = n
        stack, size = [(sy, sx)], 0
        while stack:
            y, x = stack.pop()
            size += 1
            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not labels[ny, nx]:
                    labels[ny, nx] = n
                    stack.append((ny, nx))
        if size > best_size:
            best, best_size = n, size
    return labels == best if best else np.zeros_like(mask)


def _erode(mask: np.ndarray, r: int) -> np.ndarray:
    out = mask.copy()
    for _ in range(r):
        p = np.pad(out, 1, constant_values=False)
        out = out & p[:-2, 1:-1] & p[2:, 1:-1] & p[1:-1, :-2] & p[1:-1, 2:]
    return out


def _runs(flags: np.ndarray, min_frac: float = 0.02) -> list[tuple[int, int]]:
    """[start, end) index ranges of True runs at least min_frac of the length."""
    runs, start = [], None
    for i, v in enumerate(np.append(flags, False)):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if i - start >= min_frac * len(flags):
                runs.append((start, i))
            start = None
    return runs


def detect_gaps(mosaic: StripMosaic, frame: tuple[int, int, int, int]) -> tuple[tuple[int, int], ...]:
    """Non-photo ranges along a panoramic strip (where scan parts were joined)."""
    x0, y0, x1, y1 = frame
    along_x = x1 - x0 >= y1 - y0
    if max(x1 - x0, y1 - y0) < 3 * min(x1 - x0, y1 - y0):
        return ()
    a, step = mosaic.overview()
    band = a[y0 // step : y1 // step, x0 // step : x1 // step]
    lo, hi = np.percentile(band, [1, 99])
    lit = band > lo + 0.06 * max(hi - lo, 1.0)
    runs = _runs(lit.mean(axis=0 if along_x else 1) > 0.5)
    origin = x0 if along_x else y0
    return tuple((origin + runs[i][1] * step, origin + runs[i + 1][0] * step)
                 for i in range(len(runs) - 1))


def detect_frame(mosaic: StripMosaic, cells: int = 400) -> tuple[int, int, int, int] | None:
    """Find the exposed image area inside a film scan.

    USGS scans include the black unexposed film border, a grey data strip, a
    white leader and a USGS banner around the photo. The footprint corners refer
    to the photo itself, so the homography must be fitted to its corners, not
    the scan's. The photo is taken as the largest connected non-black region of
    a block-averaged overview. Returns (x0, y0, x1, y1) in mosaic pixels, or
    None when nothing plausible is found.
    """
    a, step = mosaic.overview(cells * 8)
    b = max(1, int(math.ceil(max(a.shape) / cells)))
    h, w = a.shape[0] // b, a.shape[1] // b
    if h < 8 or w < 8:
        return None
    bm = a[: h * b, : w * b].reshape(h, b, w, b).mean(axis=(1, 3))
    lo, hi = np.percentile(bm, [1, 99])
    lit = bm > lo + 0.06 * max(hi - lo, 1.0)
    core = _largest_component(_erode(lit, 2))
    if not core.any():
        return None
    # Undo the erosion, but only into lit cells.
    for _ in range(2):
        p = np.pad(core, 1, constant_values=False)
        core = lit & (core | p[:-2, 1:-1] | p[2:, 1:-1] | p[1:-1, :-2] | p[1:-1, 2:])
    ys, xs = np.nonzero(core)
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    if max(h, w) > 3 * min(h, w):
        # Panoramic strip: a scan comes in parts, each with its own clips and
        # title block, which cut the photo into pieces along the strip. Span
        # every substantial piece that lies in the photo's cross-strip band.
        along_x = w >= h
        band = lit[y0:y1, :] if along_x else lit[:, x0:x1]
        runs = _runs(band.mean(axis=0 if along_x else 1) > 0.5)
        if runs:
            if along_x:
                x0, x1 = min(x0, runs[0][0]), max(x1, runs[-1][1])
            else:
                y0, y1 = min(y0, runs[0][0]), max(y1, runs[-1][1])
    if (y1 - y0) * (x1 - x0) < 0.25 * h * w:
        return None
    s = b * step
    return (int(x0 * s), int(y0 * s), int(min(x1 * s, mosaic.width)), int(min(y1 * s, mosaic.height)))


def assemble_scan(paths: list[Path]) -> StripMosaic:
    """Join the TIFF parts of a scan in the order that makes one continuous photo.

    USGS names the parts _a, _b, ...; on the KH-4 scans we checked, the parts run
    in reverse order along the strip (_a holds the end with the title block).
    Joined in the wrong order, the title block and film clips land inside the
    photo, so we keep the order that leaves no such gaps (reverse on a tie).
    """
    if len(paths) < 2:
        return StripMosaic(paths)
    best = None
    for order in (paths[::-1], paths):
        m = StripMosaic(order)
        frame = detect_frame(m)
        n_gaps = len(detect_gaps(m, frame)) if frame else 99
        if best is None or n_gaps < best[0]:
            best = (n_gaps, m)
    return best[1]


def write_geotiff(path: Path, data: np.ndarray, geo_affine: np.ndarray | None) -> None:
    """Write ``data`` as a TIFF; with ``geo_affine`` (2x3, pixel -> lon/lat) add
    GeoTIFF tags for WGS84 so GIS tools place it on a map."""
    extratags = []
    if geo_affine is not None:
        a = geo_affine
        transform = (
            a[0, 0], a[0, 1], 0.0, a[0, 2],
            a[1, 0], a[1, 1], 0.0, a[1, 2],
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        )
        geokeys = (1, 1, 0, 3, 1024, 0, 1, 2, 1025, 0, 1, 1, 2048, 0, 1, 4326)
        extratags = [
            (34264, "d", 16, transform, True),  # ModelTransformationTag
            (34735, "H", len(geokeys), geokeys, True),  # GeoKeyDirectoryTag
        ]
    photometric = "minisblack" if data.ndim == 2 or data.shape[-1] == 1 else "rgb"
    tifffile.imwrite(path, data, photometric=photometric, extratags=extratags, compression="zlib")


def to_preview(data: np.ndarray) -> Image.Image:
    """Contrast-stretch to 8 bit for PNG previews."""
    a = np.asarray(data, dtype=float)
    if a.ndim == 3 and a.shape[2] == 1:
        a = a[..., 0]
    valid = a[a > 0] if np.any(a > 0) else a.ravel()
    lo, hi = (np.percentile(valid, [1, 99]) if valid.size else (0, 1))
    if hi <= lo:
        hi = lo + 1
    a = np.clip((a - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(a)


# --------------------------------------------------------------------------- #
# Cropping
# --------------------------------------------------------------------------- #


def _has_wms_overlay(scene: Scene) -> bool:
    return any("wms" in str(b.get("overlayType", "")).lower() and b.get("overlayPath")
               for b in scene.raw.get("browse") or [])


def pick_reference_layers(scene: Scene, others: list[Scene], max_layers: int = 12) -> list[str]:
    """CORONA frames that together cover the scene's centre line, greedily."""
    c = scene.corners
    mid_a = [(c["NW"][i] + c["SW"][i]) / 2 for i in (0, 1)]
    mid_b = [(c["NE"][i] + c["SE"][i]) / 2 for i in (0, 1)]
    samples = [(mid_a[0] + t * (mid_b[0] - mid_a[0]), mid_a[1] + t * (mid_b[1] - mid_a[1]))
               for t in np.linspace(0, 1, 24)]
    cover = {o.entity_id: {i for i, (la, lo) in enumerate(samples) if o.contains(la, lo)}
             for o in others if o.display_id.startswith("DS1")}
    chosen, done = [], set()
    while len(chosen) < max_layers:
        ent = max(cover, key=lambda k: len(cover[k] - done), default=None)
        if ent is None or not cover[ent] - done:
            break
        chosen.append(ent)
        done |= cover.pop(ent)
    return chosen


def fetch_overlay(scene: Scene, long_side: int = 800, timeout: float = 60.0):
    """USGS's own georeferenced rendering of the scene's browse image, or of
    the overlapping CORONA frames in ``scene.reference_layers``.

    Returns (grey array, valid mask, lat grid, lon grid) on a plain lat/lon
    grid covering the footprint, or None if there is no overlay or the server
    cannot be reached.
    """
    url = next((b.get("overlayPath") for b in scene.raw.get("browse") or []
                if b.get("overlayPath") and "wms" in str(b.get("overlayType", "")).lower()), None)
    if url:
        base, _, query = url.partition("?")
        q = dict(kv.partition("=")[::2] for kv in query.split("&"))
        q = {k.lower(): v for k, v in q.items()}
        requests_ = [(base, q.get("map", "declassi"), q.get("layers", ""))]
    else:
        # One request per layer: the server rejects a whole request if any
        # one layer is missing.
        requests_ = [(WMS_URL, "declassi", layer) for layer in scene.reference_layers]
    if not requests_:
        return None
    lats = [c[0] for c in scene.corners.values()]
    lons = [c[1] for c in scene.corners.values()]
    s, n, w, e = min(lats), max(lats), min(lons), max(lons)
    aspect = (e - w) * math.cos(math.radians((s + n) / 2)) / max(n - s, 1e-9)
    width = int(long_side if aspect >= 1 else max(16, long_side * aspect))
    height = int(long_side if aspect < 1 else max(16, long_side / aspect))
    grey = np.zeros((height, width), dtype=np.float32)
    valid = np.zeros((height, width), dtype=bool)
    for base, wms_map, layers in requests_:
        params = {"service": "WMS", "version": "1.1.1", "request": "GetMap", "srs": "EPSG:4326",
                  "bbox": f"{w},{s},{e},{n}", "width": width, "height": height,
                  "format": "image/png", "styles": "", "transparent": "true",
                  "map": wms_map, "layers": layers}
        try:
            r = requests.get(base, params=params, timeout=timeout)
            r.raise_for_status()
            a = np.asarray(Image.open(io.BytesIO(r.content)).convert("RGBA"), dtype=np.float32)
        except (requests.RequestException, OSError) as err:
            log.debug("[%s] overlay %s unavailable: %s", scene.display_id, layers, err)
            continue
        g = a[..., :3].mean(axis=2)
        new = (a[..., 3] > 0) & (g > 0) & ~valid
        grey[new], valid[new] = g[new], True
    if valid.sum() < 100:
        log.info("[%s] no USGS overlay for the orientation check", scene.display_id)
        return None
    lon_g, lat_g = np.meshgrid(np.linspace(w, e, width), np.linspace(n, s, height))
    return grey, valid, lat_g, lon_g


def _box_blur(a: np.ndarray, k: int) -> np.ndarray:
    if k <= 1:
        return a
    c = np.pad(a, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    h, w = a.shape
    y0 = np.clip(np.arange(h) - k // 2, 0, h)
    y1 = np.clip(np.arange(h) + (k + 1) // 2, 0, h)
    x0 = np.clip(np.arange(w) - k // 2, 0, w)
    x1 = np.clip(np.arange(w) + (k + 1) // 2, 0, w)
    total = c[y1][:, x1] - c[y0][:, x1] - c[y1][:, x0] + c[y0][:, x0]
    return total / ((y1 - y0)[:, None] * (x1 - x0)[None, :])


def match_orientation(scene: Scene, mosaic: StripMosaic, film: FilmModel, cands,
                      proj: LocalProjection, ground: np.ndarray,
                      overlay=None) -> tuple[str | None, dict[str, float]]:
    """Score each candidate orientation by correlating the scan with USGS's
    georeferenced browse overlay. Returns (winner or None, scores)."""
    overlay = overlay if overlay is not None else fetch_overlay(scene)
    if overlay is None:
        return None, {}
    grey, valid, lat_g, lon_g = overlay
    ov, step = mosaic.overview()
    fx0, fy0, fx1, fy1 = film.frame
    gx, gy = proj.forward(lat_g, lon_g)
    # Average the scan down to the overlay's pixel size before sampling it, or
    # field-scale texture aliases into noise.
    overlay_m = float(np.hypot(gx[0, 1] - gx[0, 0], gy[0, 1] - gy[0, 0]))
    edges = np.linalg.norm(ground - np.roll(ground, -1, axis=0), axis=1)
    scan_m = max(edges) / max(fx1 - fx0, fy1 - fy0)
    ov = _box_blur(ov, int(round(overlay_m / (scan_m * step))))
    scores = {}
    for name, perm in cands:
        u, v = GroundToScan(ground, perm, film).to_px(gx, gy)
        ok = valid & (u >= fx0) & (u < fx1) & (v >= fy0) & (v < fy1) & ~film.in_gap(u, v)
        if ok.sum() < 100:
            continue
        ui = np.clip((u[ok] / step).astype(int), 0, ov.shape[1] - 1)
        vi = np.clip((v[ok] / step).astype(int), 0, ov.shape[0] - 1)
        c = np.corrcoef(ov[vi, ui], grey[ok])[0, 1]
        scores[name] = float(c) if np.isfinite(c) else 0.0
    ranked = sorted(scores.values(), reverse=True) + [0.0, 0.0]
    best = max(scores, key=scores.get) if scores else None
    if best is None or ranked[0] < 0.3 or ranked[0] - ranked[1] < 0.1:
        log.info("[%s] overlay match inconclusive: %s", scene.display_id,
                 {k: round(v, 2) for k, v in scores.items()})
        return None, scores
    log.info("[%s] orientation %s matches the USGS overlay (r=%.2f, next %.2f)",
             scene.display_id, best, ranked[0], ranked[1])
    return best, scores


@dataclass
class CropResult:
    orientation: str
    pixel_window: tuple[int, int, int, int]
    gsd_m: float
    files: dict[str, str]
    footprint_corner_lat_lon: dict[str, tuple[float, float]]
    rectified: bool
    image_frame_px: tuple[int, int, int, int]
    orientation_source: str
    orientation_scores: dict[str, float]


def crop_scene(
    scene: Scene,
    mosaic: StripMosaic,
    lat: float,
    lon: float,
    size_m: float,
    margin_m: float,
    out_dir: Path,
    stem: str,
    orientation: str = "auto",
    rectify: bool = False,
    context_m: float = 10000.0,
    pan_deg: float | None = None,
) -> list[CropResult]:
    camera = camera_of(scene)
    proj = LocalProjection(lat, lon)
    ground = scene.ground_xy(proj)
    frame = detect_frame(mosaic)
    if frame is None:
        log.warning("[%s] could not find the image frame in the scan; using the whole scan",
                    scene.display_id)
        frame = (0, 0, mosaic.width, mosaic.height)
    else:
        log.info("[%s] image frame x=%d..%d y=%d..%d of %dx%d scan", scene.display_id,
                 frame[0], frame[2], frame[1], frame[3], mosaic.width, mosaic.height)
    fx0, fy0, fx1, fy1 = frame
    gaps = detect_gaps(mosaic, frame)
    if gaps:
        log.info("[%s] cutting %d non-photo gap(s) between scan parts: %s", scene.display_id,
                 len(gaps), gaps)
    film = FilmModel(frame, camera.pan_deg if pan_deg is None else pan_deg, gaps)
    cands = orientation_candidates(ground, fx1 - fx0, fy1 - fy0)
    source, scores = "user", {}
    if orientation in ("auto", "all"):
        best, scores = match_orientation(scene, mosaic, film, cands, proj, ground)
        if best:
            source = "usgs-overlay"
        elif camera.orientation and any(c[0] == camera.orientation for c in cands):
            best, source = camera.orientation, "camera-default"
        else:
            source = "unverified"
        if best:
            cands.sort(key=lambda c: c[0] != best)
    if orientation == "all":
        chosen = cands
    elif orientation == "auto":
        chosen = cands[:1]
        if source == "unverified":
            chosen = cands
            log.warning("[%s] orientation unknown for %s; writing all %d candidates, pick the "
                        "right one from the context previews", scene.display_id, camera.name,
                        len(cands))
        else:
            log.info("[%s] orientation %s (%s)", scene.display_id, chosen[0][0], source)
    else:
        chosen = [c for c in cands if c[0].replace(" ", "").upper() == orientation.replace(" ", "").upper()]
        if not chosen:
            raise ValueError(f"orientation {orientation!r} not valid here; choose from {[c[0] for c in cands]}")

    half = size_m / 2 + margin_m
    box = np.array([[-half, half], [half, half], [half, -half], [-half, -half]])
    results = []
    for name, perm in chosen:
        g2s = GroundToScan(ground, perm, film)
        u, v = g2s.to_px(box[:, 0], box[:, 1])
        x0, x1 = int(math.floor(u.min())), int(math.ceil(u.max()))
        y0, y1 = int(math.floor(v.min())), int(math.ceil(v.max()))
        if x1 <= 0 or y1 <= 0 or x0 >= mosaic.width or y0 >= mosaic.height:
            log.warning("[%s] target falls outside the scan under orientation %s", scene.display_id, name)
            continue
        x0c, y0c = max(x0, 0), max(y0, 0)
        x1c, y1c = min(x1, mosaic.width), min(y1, mosaic.height)
        if (x0c, y0c, x1c, y1c) != (x0, y0, x1, y1):
            log.warning("[%s] window is clipped by the edge of the scan", scene.display_id)

        # Ground sample distance at the target from the mapping's Jacobian.
        eps = 1.0
        u0, v0 = g2s.to_px(0.0, 0.0)
        ux, vx = g2s.to_px(eps, 0.0)
        uy, vy = g2s.to_px(0.0, eps)
        det = abs((ux - u0) * (vy - v0) - (uy - u0) * (vx - v0))
        gsd = float(1.0 / math.sqrt(det)) if det > 0 else float("nan")

        tag = stem if len(chosen) == 1 else f"{stem}_{name.replace('=', '').replace(',', '_')}"
        files = {}
        if rectify:
            n = max(2, min(8000, int(round(2 * half / gsd))))
            gx = np.linspace(-half, half, n)
            gy = np.linspace(half, -half, n)
            GX, GY = np.meshgrid(gx, gy)
            uu, vv = g2s.to_px(GX, GY)
            raw = mosaic.read(x0c, y0c, x1c + 2, y1c + 2)
            data = _bilinear(raw, uu - x0c, vv - y0c)
            pixel_size = 2 * half / (n - 1)
            # pixel (col,row) -> local metres -> lon/lat (approximately affine)
            cols = np.array([0, n - 1, 0])
            rows = np.array([0, 0, n - 1])
            mx = -half + cols * pixel_size
            my = half - rows * pixel_size
        else:
            data = mosaic.read(x0c, y0c, x1c, y1c)
            # Fit a pixel->lon/lat affine over the crop for the GeoTIFF tags.
            cols = np.array([0, x1c - x0c, 0, x1c - x0c], dtype=float)
            rows = np.array([0, 0, y1c - y0c, y1c - y0c], dtype=float)
            mx, my = g2s.to_ground(cols + x0c, rows + y0c)
        glat, glon = proj.inverse(mx, my)
        A = np.column_stack([cols, rows, np.ones_like(cols, dtype=float)])
        coef_lon, *_ = np.linalg.lstsq(A, glon, rcond=None)
        coef_lat, *_ = np.linalg.lstsq(A, glat, rcond=None)
        geo_affine = np.vstack([coef_lon, coef_lat])

        tif = out_dir / f"{tag}.tif"
        write_geotiff(tif, data, geo_affine)
        files["geotiff"] = tif.name
        png = out_dir / f"{tag}.png"
        preview = to_preview(data)
        preview.thumbnail((PREVIEW_MAX_PX, PREVIEW_MAX_PX))
        preview.save(png)
        files["preview"] = png.name

        # Context preview: a wider window, decimated, with the target box drawn.
        if context_m and context_m > 2 * half:
            ch = context_m / 2
            cbox = np.array([[-ch, ch], [ch, ch], [ch, -ch], [-ch, -ch]])
            cu, cv = g2s.to_px(cbox[:, 0], cbox[:, 1])
            cx0, cx1 = max(int(cu.min()), 0), min(int(cu.max()), mosaic.width)
            cy0, cy1 = max(int(cv.min()), 0), min(int(cv.max()), mosaic.height)
            if cx1 > cx0 and cy1 > cy0:
                step = max(1, int(math.ceil(max(cx1 - cx0, cy1 - cy0) / 1500)))
                ctx = to_preview(mosaic.read(cx0, cy0, cx1, cy1, step=step)).convert("RGB")
                d = ImageDraw.Draw(ctx)
                tb = [((ui - cx0) / step, (vi - cy0) / step) for ui, vi in zip(u, v)]
                d.polygon(tb, outline=(255, 40, 40), width=3)
                nu, nv = g2s.to_px([0, 0], [0, 0.6 * ch])
                d.line([((nu[0] - cx0) / step, (nv[0] - cy0) / step),
                        ((nu[1] - cx0) / step, (nv[1] - cy0) / step)], fill=(40, 200, 255), width=3)
                d.text(((nu[1] - cx0) / step + 4, (nv[1] - cy0) / step), "N", fill=(40, 200, 255))
                cpng = out_dir / f"{tag}_context.png"
                ctx.save(cpng)
                files["context"] = cpng.name

        results.append(
            CropResult(
                orientation=name,
                pixel_window=(x0c, y0c, x1c, y1c),
                gsd_m=gsd,
                files=files,
                footprint_corner_lat_lon={c: scene.corners[c] for c in CORNERS},
                rectified=rectify,
                image_frame_px=frame,
                orientation_source=source,
                orientation_scores=scores,
            )
        )
        log.info(
            "[%s] %s: window x=%d..%d y=%d..%d (%dx%d px, ~%.2f m/px) -> %s",
            scene.display_id, name, x0c, x1c, y0c, y1c, x1c - x0c, y1c - y0c, gsd, tif,
        )
    return results


def _bilinear(img: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    u = np.clip(u, 0, w - 1.001)
    v = np.clip(v, 0, h - 1.001)
    x0 = np.floor(u).astype(int)
    y0 = np.floor(v).astype(int)
    fx = u - x0
    fy = v - y0
    if img.ndim == 3:
        fx = fx[..., None]
        fy = fy[..., None]
    a = img[y0, x0].astype(float)
    b = img[y0, x0 + 1].astype(float)
    c = img[y0 + 1, x0].astype(float)
    d = img[y0 + 1, x0 + 1].astype(float)
    out = a * (1 - fx) * (1 - fy) + b * fx * (1 - fy) + c * (1 - fx) * fy + d * fx * fy
    if np.issubdtype(img.dtype, np.integer):
        info = np.iinfo(img.dtype)
        out = np.clip(np.rint(out), info.min, info.max)
    return out.astype(img.dtype)


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #


@dataclass
class Point:
    name: str
    lat: float
    lon: float


def parse_points(args) -> list[Point]:
    pts: list[Point] = []
    for i, s in enumerate(args.point or []):
        try:
            lat_s, lon_s = s.split(",")
            pts.append(Point(f"pt{i + 1}", float(lat_s), float(lon_s)))
        except ValueError:
            raise SystemExit(f"bad --point {s!r}; expected LAT,LON")
    if args.points_file:
        with open(args.points_file, newline="") as f:
            sample = f.read(2048)
            f.seek(0)
            has_header = any(k in sample.lower() for k in ("lat", "lon", "name"))
            if has_header:
                for i, row in enumerate(csv.DictReader(f)):
                    row = {k.strip().lower(): v for k, v in row.items() if k}
                    lat = row.get("lat") or row.get("latitude")
                    lon = row.get("lon") or row.get("lng") or row.get("long") or row.get("longitude")
                    if lat is None or lon is None:
                        raise SystemExit(f"{args.points_file}: need lat/lon columns")
                    pts.append(Point(row.get("name") or row.get("id") or f"row{i + 1}", float(lat), float(lon)))
            else:
                for i, row in enumerate(csv.reader(f)):
                    if not row or row[0].startswith("#"):
                        continue
                    if len(row) >= 3:
                        pts.append(Point(row[0], float(row[1]), float(row[2])))
                    else:
                        pts.append(Point(f"row{i + 1}", float(row[0]), float(row[1])))
    for p in pts:
        if not (-90 <= p.lat <= 90 and -180 <= p.lon <= 180):
            raise SystemExit(f"point {p.name} out of range: {p.lat},{p.lon}")
    if not pts:
        raise SystemExit("give at least one --point LAT,LON or --points-file")
    return pts


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s).strip("_") or "point"


def rank_scenes(scenes: list[Scene], sort: str, datasets: list[str] | None = None) -> list[Scene]:
    """Frames flagged downloadable first, then unknown, then not; within each,
    by dataset preference, then coarse ARGON frames last, then by date."""
    by_date = sorted(scenes, key=lambda s: s.acquired, reverse=(sort == "newest"))
    tier = {True: 0, None: 1, False: 2}
    pref = {d: i for i, d in enumerate(datasets or [])}
    return sorted(by_date, key=lambda s: (tier[s.download_available()], pref.get(s.dataset, 0),
                                          camera_of(s) is CAMERAS["argon"]))


def run(args) -> int:
    points = parse_points(args)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    raw_root = out_root / "_scenes"

    username = args.username or os.environ.get("M2M_USERNAME")
    token = os.environ.get("M2M_TOKEN")
    if not username:
        raise SystemExit("set M2M_USERNAME (or --username)")
    if not token:
        if not sys.stdin.isatty():
            raise SystemExit("set M2M_TOKEN to your ERS application token")
        token = getpass.getpass(f"M2M application token for {username}: ")

    datasets = [d.strip() for d in args.dataset.split(",") if d.strip()]
    summary = []
    with M2MClient(args.api_url) as api:
        api.login(username, token)
        log.info("logged in to M2M as %s", username)
        for p in points:
            bbox = box_around(p.lat, p.lon, args.size_m)
            scenes = []
            for ds in datasets:
                log.info("== %s (%.5f, %.5f): searching %s", p.name, p.lat, p.lon, ds)
                for s in (scene_from_result(r) for r in
                          api.scene_search(ds, bbox, args.start, args.end, args.max_search)):
                    if s and s.contains(p.lat, p.lon):
                        s.dataset = ds
                        scenes.append(s)
            scenes = rank_scenes(scenes, args.sort, datasets)
            log.info("   %d frame(s) cover the point", len(scenes))
            entry = {"point": p.__dict__, "candidates": [], "crops": []}
            summary.append(entry)
            for s in scenes:
                entry["candidates"].append(
                    {"dataset": s.dataset, "entityId": s.entity_id, "displayId": s.display_id,
                     "acquired": s.acquired, "camera": camera_of(s).name,
                     "downloadAvailable": s.download_available(),
                     "mission": s.metadata.get("Mission")}
                )
                log.info("   %-20s %s  %-30s available=%s", s.display_id, s.acquired,
                         camera_of(s).name, s.download_available())
            if args.dry_run or not scenes:
                continue

            # Ask for download options for the top candidates; skip unscanned frames.
            selected: list[tuple[Scene, dict]] = []
            pool = scenes[: max(args.max_scenes * 5, 10)]
            by_entity: dict[str, list[dict]] = {}
            for ds in datasets:
                ids = [s.entity_id for s in pool if s.dataset == ds]
                for o in api.download_options(ds, ids) if ids else []:
                    by_entity.setdefault(str(o.get("entityId")), []).append(o)
            for s in pool:
                prod = pick_product(by_entity.get(s.entity_id, []), args.product)
                if prod is None:
                    log.info("   %s: no downloadable product (not yet scanned? order it in EarthExplorer)",
                             s.display_id)
                    continue
                selected.append((s, prod))
                if len(selected) >= args.max_scenes:
                    break
            if not selected:
                log.warning("   no frame covering %s is downloadable right now", p.name)
                continue

            need = [(s, prod) for s, prod in selected if not _cached_tiffs(raw_root / s.entity_id)]
            urls: dict[str, str] = {}
            if need:
                label = f"corona_fetch_{int(time.time())}"
                for s, prod in need:
                    size = float(prod.get("filesize") or 0) / 1e6
                    log.info("   requesting %s (%s, %.0f MB)", s.display_id, prod.get("productName"), size)
                urls = api.resolve_download_urls(
                    [{"entityId": s.entity_id, "productId": prod["id"]} for s, prod in need],
                    label, poll_s=args.poll_s, timeout_s=args.wait_s,
                )
                api.download_label_remove(label)

            point_dir = out_root / _slug(p.name)
            point_dir.mkdir(parents=True, exist_ok=True)
            for s, prod in selected:
                if not _has_wms_overlay(s) and args.orientation in ("auto", "all"):
                    lats = [c[0] for c in s.corners.values()]
                    lons = [c[1] for c in s.corners.values()]
                    try:
                        found = api.scene_search(DEFAULT_DATASET, (min(lats), min(lons), max(lats), max(lons)),
                                                 max_results=300)
                        others = [o for o in (scene_from_result(r) for r in found) if o]
                        s.reference_layers = pick_reference_layers(s, others)
                    except M2MError as e:
                        log.info("   %s: no CORONA reference for the orientation check: %s", s.display_id, e)
                scene_dir = raw_root / s.entity_id
                tiffs = _cached_tiffs(scene_dir)
                if not tiffs:
                    url = urls.get(s.entity_id)
                    if not url:
                        log.warning("   %s: no download URL; skipping", s.display_id)
                        continue
                    archive = http_download(url, scene_dir / "archive")
                    tiffs = extract_tiffs(archive, scene_dir / "tiff")
                    if not args.keep_archives and archive.suffix.lower() not in TIFF_SUFFIXES:
                        archive.unlink()
                log.info("   %s: %d TIFF part(s)", s.display_id, len(tiffs))
                mosaic = assemble_scan(tiffs)
                stem = f"{_slug(p.name)}_{_slug(s.display_id)}"
                margin = camera_of(s).margin_m if args.margin_m is None else args.margin_m
                crops = crop_scene(
                    s, mosaic, p.lat, p.lon, args.size_m, margin, point_dir, stem,
                    orientation=args.orientation, rectify=args.rectify, context_m=args.context_m,
                    pan_deg=args.pan_angle,
                )
                if s.browse_url and args.browse:
                    try:
                        b = http_download(s.browse_url, point_dir / "browse" / s.entity_id)
                        log.info("   browse image: %s", b)
                    except requests.RequestException as e:
                        log.debug("browse download failed: %s", e)
                meta = {
                    "point": p.__dict__,
                    "size_m": args.size_m,
                    "margin_m": margin,
                    "camera": camera_of(s).__dict__,
                    "scene": {"dataset": s.dataset, "entityId": s.entity_id, "displayId": s.display_id,
                              "acquired": s.acquired, "corners": s.corners, "metadata": s.metadata},
                    "product": {k: prod.get(k) for k in ("id", "productName", "filesize")},
                    "source_tiffs": [str(t) for t in tiffs],
                    "mosaic_size_px": [mosaic.width, mosaic.height],
                    "crops": [c.__dict__ for c in crops],
                    "note": "Geolocation is approximate (footprint corners + film model); "
                            "expect errors of about margin_m. Check the context preview.",
                }
                (point_dir / f"{stem}.json").write_text(json.dumps(meta, indent=2, default=str))
                entry["crops"].extend(
                    {"scene": s.display_id, **c.__dict__, "dir": str(point_dir)} for c in crops
                )
                del mosaic
                if args.delete_scenes:
                    shutil.rmtree(scene_dir, ignore_errors=True)

    (out_root / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    n_crops = sum(len(e["crops"]) for e in summary)
    log.info("done: %d point(s), %d crop(s); summary in %s", len(points), n_crops, out_root / "summary.json")
    return 0


def _cached_tiffs(scene_dir: Path) -> list[Path]:
    d = scene_dir / "tiff"
    if not d.is_dir():
        return []
    return sorted(p for p in d.rglob("*") if p.suffix.lower() in TIFF_SUFFIXES)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Download CORONA imagery around coordinates via the USGS M2M API.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    g = p.add_argument_group("points")
    g.add_argument("--point", action="append", metavar="LAT,LON", help="target point (repeatable)")
    g.add_argument("--points-file", help="CSV of points: name,lat,lon (header optional)")
    g = p.add_argument_group("search")
    g.add_argument("--dataset", default=DEFAULT_DATASETS,
                   help="M2M dataset(s), comma-separated in order of preference "
                        "(declassiii = KH-9 HEXAGON, corona2 = CORONA/ARGON)")
    g.add_argument("--start", help="earliest acquisition date YYYY-MM-DD")
    g.add_argument("--end", help="latest acquisition date YYYY-MM-DD")
    g.add_argument("--sort", choices=("newest", "oldest"), default="newest",
                   help="frame preference (newest favours the sharper KH-4B missions)")
    g.add_argument("--max-search", type=int, default=200, help="max search results per point")
    g.add_argument("--max-scenes", type=int, default=1, help="frames to download per point")
    g.add_argument("--product", help="substring of the product name to prefer")
    g.add_argument("--dry-run", action="store_true", help="search only; download nothing")
    g = p.add_argument_group("output")
    g.add_argument("--out", default="corona_output", help="output directory")
    g.add_argument("--size-m", type=float, default=1000.0, help="side of the square window, metres")
    g.add_argument("--margin-m", type=float, default=None,
                   help="extra margin on each side to absorb geolocation error, metres "
                        "(default: per camera, 1 km for ARGON, 5 km for KH-4, 10 km for KH-9)")
    g.add_argument("--pan-angle", type=float, default=None,
                   help="override the panoramic half scan angle in degrees (0 = frame camera)")
    g.add_argument("--context-m", type=float, default=30000.0,
                   help="side of the context preview, metres (0 disables)")
    g.add_argument("--orientation", default="auto",
                   help="how the scan sits relative to north: 'auto' (north-up, unmirrored), 'all' "
                        "(write every plausible orientation), or a name like 'TL=SE,TR=SW'")
    g.add_argument("--rectify", action="store_true",
                   help="resample the window to a north-up square instead of a raw pixel cut-out")
    g.add_argument("--browse", action="store_true", help="also save the USGS browse image")
    g.add_argument("--keep-archives", action="store_true", help="keep downloaded .tgz files")
    g.add_argument("--delete-scenes", action="store_true",
                   help="delete full scans after cropping (they are ~0.5-2 GB each)")
    g = p.add_argument_group("API")
    g.add_argument("--username", help="ERS username (default: $M2M_USERNAME)")
    g.add_argument("--api-url", default=os.environ.get("M2M_API_URL", API_URL))
    g.add_argument("--poll-s", type=float, default=15.0, help="download staging poll interval")
    g.add_argument("--wait-s", type=float, default=1800.0, help="max wait for staging")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    try:
        return run(args)
    except M2MError as e:
        log.error("M2M API error: %s", e)
        return 2


if __name__ == "__main__":
    sys.exit(main())
