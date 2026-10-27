"""Live tests against the real USGS M2M API.

Skipped unless M2M_USERNAME and M2M_TOKEN are set. Set CORONA_LIVE_DOWNLOAD=1 to
also download one full scan (0.5-2 GB) and crop it.

    CORONA_TEST_POINT=37.2343,-115.8067 python3 -m pytest -q tests/test_live.py -s
"""

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import corona_fetch as cf  # noqa: E402

pytestmark = pytest.mark.skipif(
    not (os.environ.get("M2M_USERNAME") and os.environ.get("M2M_TOKEN")),
    reason="set M2M_USERNAME and M2M_TOKEN to run live USGS tests",
)
POINT = os.environ.get("CORONA_TEST_POINT", "37.2343,-115.8067")


def test_live_search(tmp_path):
    assert cf.main(["--point", POINT, "--out", str(tmp_path), "--dry-run"]) == 0
    summary = json.loads((tmp_path / "summary.json").read_text())
    cands = summary[0]["candidates"]
    print(json.dumps(cands[:10], indent=2))
    assert cands, f"no CORONA frames found covering {POINT}"


@pytest.mark.skipif(not os.environ.get("CORONA_LIVE_DOWNLOAD"), reason="set CORONA_LIVE_DOWNLOAD=1")
def test_live_download_and_crop(tmp_path):
    out = Path(os.environ.get("CORONA_LIVE_OUT", tmp_path))
    rc = cf.main(["--point", POINT, "--out", str(out), "--orientation", "all", "--margin-m", "500"])
    assert rc == 0
    crops = list((out / "pt1").glob("*.tif"))
    print("\n".join(str(c) for c in crops))
    assert crops, "no crop written"
