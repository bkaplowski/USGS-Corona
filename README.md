# USGS CORONA fetch

`corona_fetch.py` downloads declassified CORONA (KH-1 to KH-4B) film scans from the
USGS EROS M2M API and cuts out a ~1 km x 1 km window around each input coordinate.

## Setup

```
pip install -r requirements.txt
export M2M_USERNAME=<ERS username>
export M2M_TOKEN=<M2M application token>   # ers.cr.usgs.gov -> Profile -> Application Tokens
```

## Usage

```
./corona_fetch.py --point 37.2343,-115.8067 --dry-run          # list covering frames only
./corona_fetch.py --point 37.2343,-115.8067                    # download + 1 km crop
./corona_fetch.py --points-file sites.csv --max-scenes 2       # CSV: name,lat,lon
./corona_fetch.py --point LAT,LON --orientation all --margin-m 500 --rectify
```

Outputs per point (in `corona_output/<name>/`): an approximately georeferenced
GeoTIFF (WGS84), a PNG preview, a ~10 km `_context.png` with the target box drawn,
and a JSON sidecar. Full scans are cached in `corona_output/_scenes/` (0.5-2 GB each;
`--delete-scenes` removes them after cropping).

## Caveats

- CORONA scans are not georeferenced. The crop is located with a homography fitted
  to the four footprint corners in the USGS metadata; panoramic distortion means it
  can be off by hundreds of metres. Use `--margin-m` and check the context preview.
- The scan includes film border, a data strip and a USGS banner around the photo.
  The photo frame is detected automatically and the footprint corners are mapped
  onto it, not onto the whole scan.
- The metadata does not say how the film is rotated/mirrored relative to north.
  `--orientation auto` correlates the scan with USGS's georeferenced browse overlay
  and picks the matching orientation; if the overlay is unavailable or the match is
  ambiguous it falls back to north-up and logs a warning. `--orientation all` writes
  every plausible orientation (best match first; scores are in the JSON sidecar).
- Frames not yet scanned by USGS have no downloadable product; they are skipped
  (order them through EarthExplorer).

## Tests

```
pip install pytest
python3 -m pytest -q tests                      # unit tests (geometry, parsing, TIFF reading)
M2M_USERNAME=... M2M_TOKEN=... python3 -m pytest -q tests/test_live.py -s   # live search
CORONA_LIVE_DOWNLOAD=1 ... python3 -m pytest -q tests/test_live.py -s       # live download + crop
```

The live tests call the real USGS M2M API and are skipped when no credentials are set.
