import sys
from pathlib import Path

import numpy as np
import pytest
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import corona_fetch as cf  # noqa: E402

TARGET = (37.2343, -115.8067)


# ----------------------------------------------------------------- geometry --


def test_box_around_is_size_m_square():
    lat, lon = TARGET
    b = cf.box_around(lat, lon, 1000)
    proj = cf.LocalProjection(lat, lon)
    x0, y0 = proj.forward(b[0], b[1])
    x1, y1 = proj.forward(b[2], b[3])
    assert x1 - x0 == pytest.approx(1000, rel=1e-6)
    assert y1 - y0 == pytest.approx(1000, rel=1e-6)


def test_projection_roundtrip_and_antimeridian():
    proj = cf.LocalProjection(-16.5, 179.9)
    x, y = proj.forward(-16.4, -179.95)
    assert 0 < x < 20_000  # 0.15 deg east across the antimeridian
    lat, lon = proj.inverse(x, y)
    assert lat == pytest.approx(-16.4) and lon == pytest.approx(-179.95)


def test_homography_roundtrip():
    rng = np.random.default_rng(0)
    src = np.array([[0, 0], [100, 5], [110, 40], [-5, 35]], float)
    dst = np.array([[0, 0], [9000, 0], [9000, 800], [0, 800]], float)
    H = cf.fit_homography(src, dst)
    u, v = cf.apply_homography(H, src[:, 0], src[:, 1])
    np.testing.assert_allclose(np.column_stack([u, v]), dst, atol=1e-6)
    pts = rng.uniform(0, 100, (20, 2))
    u, v = cf.apply_homography(H, pts[:, 0], pts[:, 1])
    x, y = cf.apply_homography(np.linalg.inv(H), u, v)
    np.testing.assert_allclose(np.column_stack([x, y]), pts, atol=1e-6)


@pytest.mark.parametrize("value,expected", [
    ("37.25", 37.25),
    (-115.5, -115.5),
    ("37°15'00\"N", 37.25),
    ("115°30'00\"W", -115.5),
    ("371500N", 37.25),
    ("1153000W", -115.5),
    ("-12 30 00", -12.5),
    ("37&deg;48'39.60\"N", 37.811),  # as the M2M API sends it
    ("116&deg;14'20.40\"W", -116.239),
    ("33&deg;30'00\"S", -33.5),
    ("", None),
])
def test_parse_coordinate(value, expected):
    got = cf.parse_coordinate(value)
    assert got == (pytest.approx(expected) if expected is not None else None)


def test_scene_from_real_m2m_metadata():
    # Field names and values as returned by scene-search for corona2,
    # including USGS's "Cormer" typo and the boolean availability flag.
    md = {
        "Entity ID": "DS1111-1010AA025", "Acquisition Date": "1970/07/23",
        "Down Load Available": False,
        "NW Corner Lat": "37&deg;48'39.60\"N", "NW Corner Long": "116&deg;14'20.40\"W",
        "NW Cormer Lat dec": "37.811", "NW Corner Long dec": "-116.239",
        "NE Corner Lat dec": "35.899", "NE Corner Long dec": "-113.233",
        "SE Corner Lat dec": "35.67", "SE Corner Long dec": "-113.387",
        "SW Corner Lat dec": "37.639", "SW Corner Long dec": "-116.489",
    }
    result = {"entityId": "DS1111-1010AA025", "options": {"download": False},
              "metadata": [{"fieldName": k, "value": v} for k, v in md.items()]}
    s = cf.scene_from_result(result)
    assert s.corners["NW"] == (pytest.approx(37.811), pytest.approx(-116.239))
    assert s.download_available() is False
    md["Down Load Available"] = True
    result["metadata"] = [{"fieldName": k, "value": v} for k, v in md.items()]
    assert cf.scene_from_result(result).download_available() is True
    result["metadata"] = [m for m in result["metadata"] if m["fieldName"] != "Down Load Available"]
    assert cf.scene_from_result(result).download_available() is False  # from options.download


def test_orientation_candidates_square_frames_keep_all_eight():
    square = np.array([[0, 100], [100, 100], [100, 0], [0, 0]], float)
    assert len(cf.orientation_candidates(square, 1000, 1030)) == 8
    strip = np.array([[0, 15], [200, 15], [200, 0], [0, 0]], float)
    names = [n for n, _ in cf.orientation_candidates(strip, 9000, 800)]
    assert len(names) == 4 and names[0] == "TL=NW,TR=NE"


def test_pick_product_skips_browse_and_unavailable():
    opts = [
        {"id": "b", "productName": "Browse", "available": True, "filesize": 10},
        {"id": "f", "productName": "Film Scan", "available": False, "filesize": 9e8},
    ]
    assert cf.pick_product(opts) == opts[0]  # nothing better is available
    opts[1]["available"] = True
    assert cf.pick_product(opts)["id"] == "f"
    assert cf.pick_product([]) is None


# ------------------------------------------------------------------- mosaic --


def test_strip_mosaic_reads_across_parts(tmp_path):
    rng = np.random.default_rng(1)
    full = rng.integers(0, 255, (50, 300), dtype=np.uint8)
    paths = []
    for i, (a, b) in enumerate([(0, 120), (120, 210), (210, 300)]):
        p = tmp_path / f"x_{'abc'[i]}.tif"
        tifffile.imwrite(p, full[:, a:b])
        paths.append(p)
    m = cf.StripMosaic(paths)
    assert (m.width, m.height, m.axis) == (300, 50, 1)
    np.testing.assert_array_equal(m.read(100, 10, 250, 40), full[10:40, 100:250])
    np.testing.assert_array_equal(m.read(101, 3, 299, 47, step=7), full[3:47:7, 101:299:7])
    # Partly outside: zero-padded
    out = m.read(-10, -5, 20, 10)
    assert out.shape == (15, 30)
    np.testing.assert_array_equal(out[5:, 10:], full[0:10, 0:20])
    assert not out[:5].any() and not out[:, :10].any()


def test_extract_rejects_path_traversal(tmp_path):
    import io
    import tarfile
    evil = tmp_path / "evil.tgz"
    with tarfile.open(evil, "w:gz") as tf:
        data = b"x"
        info = tarfile.TarInfo("../../escape.tif")
        info.size = len(data)
        tf.addfile(info, io.BytesIO(data))
    with pytest.raises(ValueError):
        cf.extract_tiffs(evil, tmp_path / "out")


def _scan_with_border(tmp_path, photo):
    """Embed ``photo`` in a USGS-style scan: black film border, grey data strip
    on the left, white leader on the right."""
    h, w = photo.shape
    scan = np.zeros((h + 200, w + 400), np.uint8)
    scan[:, :60] = 100  # data strip
    scan[:, -40:] = 250  # leader
    scan[100 : 100 + h, 150 : 150 + w] = photo
    path = tmp_path / "scan_a.tif"
    tifffile.imwrite(path, scan)
    return cf.StripMosaic([path]), (150, 100, 150 + w, 100 + h)


def test_detect_frame_finds_photo_inside_border(tmp_path):
    rng = np.random.default_rng(2)
    photo = rng.integers(40, 230, (900, 1200), dtype=np.uint8)
    photo[300:400, 500:900] = 30  # dark shadow inside the photo
    m, expected = _scan_with_border(tmp_path, photo)
    got = cf.detect_frame(m)
    cell = max(m.width, m.height) / 400
    np.testing.assert_allclose(got, expected, atol=2 * cell)


def test_match_orientation_against_overlay(tmp_path):
    # A ground scene with structure, photographed rotated 180 degrees.
    corners = {"NW": (37.3, -115.9), "NE": (37.3, -115.7), "SE": (37.15, -115.7), "SW": (37.15, -115.9)}
    scene = cf.Scene("e", "d", "", corners)
    proj = scene.projection()
    ground = scene.ground_xy(proj)
    yy, xx = np.mgrid[0:300, 0:400]
    texture = (128 + 60 * np.sin(xx / 23.0) * np.cos(yy / 37.0) + 0.3 * xx - 0.2 * yy).astype(np.uint8)
    photo = texture[::-1, ::-1].copy()  # TL=SE,TR=SW
    m, frame = _scan_with_border(tmp_path, photo)
    lon_g, lat_g = np.meshgrid(np.linspace(-115.9, -115.7, 400), np.linspace(37.3, 37.15, 300))
    overlay = (texture.astype(np.float32), np.ones(texture.shape, bool), lat_g, lon_g)
    cands = cf.orientation_candidates(ground, frame[2] - frame[0], frame[3] - frame[1])
    best, scores = cf.match_orientation(scene, m, frame, cands, proj, ground, overlay)
    assert best == "TL=SE,TR=SW", scores
