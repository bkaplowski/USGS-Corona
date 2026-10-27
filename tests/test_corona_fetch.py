import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import corona_fetch as cf  # noqa: E402
from fake_m2m import FakeM2M, SyntheticScene  # noqa: E402

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


def test_corners_prefer_decimal_fields_and_polygon_fallback():
    s = SyntheticScene("DS1", TARGET, TARGET)
    res = s.search_result()
    scene = cf.scene_from_result(res)
    for k in cf.CORNERS:
        assert scene.corners[k] == pytest.approx(s.corners[k], abs=1e-5)
    res["metadata"] = []  # force the spatialCoverage fallback
    scene2 = cf.scene_from_result(res)
    for k in cf.CORNERS:
        assert scene2.corners[k] == pytest.approx(s.corners[k], abs=1e-9)
    assert scene.contains(*TARGET)
    assert not scene.contains(TARGET[0] + 1, TARGET[1])


def test_orientation_candidates_respect_long_axis():
    s = SyntheticScene("DS1", TARGET, TARGET)
    sc = cf.scene_from_result(s.search_result())
    g = sc.ground_xy(sc.projection())
    cands = cf.orientation_candidates(g, 12000, 1200)
    assert len(cands) == 4
    assert cands[0][0] == "TL=NW,TR=NE"
    assert "TL=SE,TR=SW" in [c[0] for c in cands]
    # A tall image would need the footprint's short axis across the top.
    tall = cf.orientation_candidates(g, 1200, 12000)
    assert len(tall) == 4 and all(c[0] not in [x[0] for x in cands] for c in tall)


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


# --------------------------------------------------------------- end to end --


def _run(fake, tmp_path, *extra, monkeypatch, points=("--point", f"{TARGET[0]},{TARGET[1]}")):
    monkeypatch.setenv("M2M_USERNAME", "user")
    monkeypatch.setenv("M2M_TOKEN", "tok")
    out = tmp_path / "out"
    argv = [*points, "--out", str(out), "--api-url", fake.api_url, "--poll-s", "0.01", *extra]
    assert cf.main(argv) == 0
    return out


def _crop_tif(out: Path, name="pt1"):
    tifs = sorted((out / name).glob("*.tif"))
    return tifs


def test_end_to_end_north_up(tmp_path, monkeypatch):
    # Target sits 25 km east of the footprint centre, inside part "b".
    center = cf.LocalProjection(*TARGET).inverse(-25_000, 1_000)
    scene = SyntheticScene("DS1107-2088DA070", TARGET, (float(center[0]), float(center[1])))
    off = SyntheticScene("DS1107-2088DA071", TARGET, (TARGET[0] + 0.5, TARGET[1]))  # MBR miss
    with FakeM2M([scene, off], tmp_path) as fake:
        out = _run(fake, tmp_path, monkeypatch=monkeypatch)
        eps = fake.endpoints_called()
    assert eps[0] == "login-token" and eps[-1] == "logout"
    assert eps.count("download-retrieve") == 2  # polled until staged
    tifs = _crop_tif(out)
    assert [t.name for t in tifs] == ["pt1_DS1107-2088DA070.tif"]
    data = tifffile.imread(tifs[0])
    # 1 km at 10 m/px (120 km over 12000 px), slightly more for the footprint rotation
    assert 95 <= data.shape[0] <= 125 and 95 <= data.shape[1] <= 125
    # Target disc (r = 100 m) is centred in the cut-out
    a = data.astype(float)
    ys, xs = np.nonzero(a >= 250)
    assert abs(xs.mean() - a.shape[1] / 2) < 3 and abs(ys.mean() - a.shape[0] / 2) < 3
    # Geo tags: centre pixel maps back to the target within ~1 pixel
    with tifffile.TiffFile(tifs[0]) as tf:
        t = np.array(tf.pages[0].tags["ModelTransformationTag"].value).reshape(4, 4)
        keys = tf.pages[0].tags["GeoKeyDirectoryTag"].value
    assert 4326 in keys
    lon = t[0, 0] * a.shape[1] / 2 + t[0, 1] * a.shape[0] / 2 + t[0, 3]
    lat = t[1, 0] * a.shape[1] / 2 + t[1, 1] * a.shape[0] / 2 + t[1, 3]
    x, y = cf.LocalProjection(*TARGET).forward(lat, lon)
    assert math.hypot(x, y) < 15
    meta = json.loads((out / "pt1" / "pt1_DS1107-2088DA070.json").read_text())
    assert meta["crops"][0]["gsd_m"] == pytest.approx(10.0, rel=0.05)
    assert (out / "pt1" / "pt1_DS1107-2088DA070_context.png").exists()
    summary = json.loads((out / "summary.json").read_text())
    assert [c["entityId"] for c in summary[0]["candidates"]] == ["DS1107-2088DA070"]


def test_end_to_end_rectified_is_north_up(tmp_path, monkeypatch):
    scene = SyntheticScene("DS1", TARGET, TARGET, rotate_deg=25)
    with FakeM2M([scene], tmp_path) as fake:
        out = _run(fake, tmp_path, "--rectify", monkeypatch=monkeypatch)
    data = tifffile.imread(_crop_tif(out)[0]).astype(float)
    n = data.shape[0]
    assert data.shape[0] == data.shape[1] and 95 <= n <= 105
    ys, xs = np.nonzero(data >= 245)
    assert abs(xs.mean() - n / 2) < 2 and abs(ys.mean() - n / 2) < 2
    # The north-pointing bar must be above the disc in a north-up image
    bar = (data > 190) & (data < 215)
    bys, bxs = np.nonzero(bar)
    assert bys.mean() < n / 2 - 15 and abs(bxs.mean() - n / 2) < 3


def test_rotated_scan_needs_orientation_all(tmp_path, monkeypatch):
    """When the scan is rotated 180 deg vs north (not knowable from metadata), the
    default guess misses the target and --orientation all recovers it."""
    center = cf.LocalProjection(*TARGET).inverse(-25_000, 1_000)
    scene = SyntheticScene("DS2", TARGET, (float(center[0]), float(center[1])), image_rotation=2)
    with FakeM2M([scene], tmp_path) as fake:
        out = _run(fake, tmp_path, "--orientation", "all", monkeypatch=monkeypatch)
    tifs = {t.stem: tifffile.imread(t) for t in _crop_tif(out)}
    assert len(tifs) >= 2
    hits = {k for k, v in tifs.items() if (v >= 250).sum() > 200}
    assert hits == {"pt1_DS2_TLSE_TRSW"}


def test_unavailable_scene_skipped_and_dry_run(tmp_path, monkeypatch):
    a = SyntheticScene("DS_NEW", TARGET, TARGET, available=False, acquired="1971-01-01")
    b = SyntheticScene("DS_OLD", TARGET, TARGET, acquired="1965-01-01")
    with FakeM2M([a, b], tmp_path) as fake:
        out = _run(fake, tmp_path, "--dry-run", monkeypatch=monkeypatch)
        assert "download-request" not in fake.endpoints_called()
        summary = json.loads((out / "summary.json").read_text())
        # downloadable frames are ranked first even though older
        assert [c["entityId"] for c in summary[0]["candidates"]] == ["DS_OLD", "DS_NEW"]
        out = _run(fake, tmp_path, monkeypatch=monkeypatch)
    assert [t.name for t in _crop_tif(out)] == ["pt1_DS_OLD.tif"]


def test_points_file_and_scene_reuse(tmp_path, monkeypatch):
    proj = cf.LocalProjection(*TARGET)
    lat2, lon2 = proj.inverse(5_000, 500)
    csv_path = tmp_path / "pts.csv"
    csv_path.write_text(f"name,lat,lon\nsite A,{TARGET[0]},{TARGET[1]}\nsite B,{lat2},{lon2}\n")
    scene = SyntheticScene("DS3", TARGET, TARGET)
    with FakeM2M([scene], tmp_path) as fake:
        out = _run(fake, tmp_path, monkeypatch=monkeypatch, points=("--points-file", str(csv_path)))
        eps = fake.endpoints_called()
    assert eps.count("download-request") == 1  # second point reuses the cached scan
    assert _crop_tif(out, "site_A") and _crop_tif(out, "site_B")
    b = tifffile.imread(_crop_tif(out, "site_B")[0])
    assert (b >= 250).sum() == 0  # disc is 5 km away from site B


def test_bad_credentials(tmp_path, monkeypatch):
    monkeypatch.setenv("M2M_USERNAME", "user")
    monkeypatch.setenv("M2M_TOKEN", "wrong")
    with FakeM2M([], tmp_path) as fake:
        rc = cf.main(["--point", "1,1", "--out", str(tmp_path), "--api-url", fake.api_url])
    assert rc == 2
