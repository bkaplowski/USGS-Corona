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
    ("", None),
])
def test_parse_coordinate(value, expected):
    got = cf.parse_coordinate(value)
    assert got == (pytest.approx(expected) if expected is not None else None)


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
