# Declassified imagery fetch (HEXAGON / CORONA / ARGON)

`corona_fetch.py` is a single-file tool that downloads declassified spy-satellite
film scans from the USGS EROS M2M API and cuts out a georeferenced window around
each input coordinate. To hand it to another project, copy `corona_fetch.py` and
`requirements.txt`; nothing else is needed at runtime.

| Dataset (`--dataset`) | System | Years | Ground resolution |
|---|---|---|---|
| `declassiii` (default, tried first) | KH-9 HEXAGON panoramic | 1971-84 | 0.6-1.2 m |
| `corona2` (default, fallback) | CORONA KH-4/4A/4B panoramic | 1960-72 | ~2 m |
| `corona2` | ARGON (KH-5) frame camera | 1962-64 | ~30 m (ranked last) |
| `declassii` (opt-in) | KH-7 GAMBIT strip camera | 1963-67 | ~0.5 m (0.48 m measured) |
| `declassii` (opt-in) | KH-9 mapping camera | 1973-80 | 6-9 m |

## Setup

```
pip install -r requirements.txt          # numpy, requests, tifffile, Pillow
export M2M_USERNAME=<ERS username>
export M2M_TOKEN=<M2M application token> # ers.cr.usgs.gov -> Profile -> Application Tokens
```

Disk: each scan is 0.5-7 GB and is kept in `<out>/_scenes/` so reruns are fast
(`--delete-scenes` removes it after cropping). Each output GeoTIFF is up to a
few hundred MB because of the error margins below.

## Usage

Points go in a CSV with columns `name,lat,lon` (see `examples/morocco_points.csv`)
or on the command line with `--point LAT,LON`.

```
# 1. See what exists for the 1970s; downloads nothing, writes <out>/summary.json
./corona_fetch.py --points-file examples/morocco_points.csv \
    --start 1970-01-01 --end 1979-12-31 --dry-run --out declass_morocco

# 2. Download and cut out the best frame per point
./corona_fetch.py --points-file examples/morocco_points.csv \
    --start 1970-01-01 --end 1979-12-31 --out declass_morocco

# Useful options
#   --max-scenes 2           frames per point (default 1)
#   --dataset corona2        CORONA only; or "corona2,declassiii" to prefer CORONA
#   --orientation all        write every plausible film orientation
#   --rectify                resample the window north-up instead of a raw film cut-out
#   --margin-m 2000          override the per-camera error margin
```

Ranking: downloadable frames first, then by dataset order, then ARGON last,
then newest (`--sort oldest` to reverse). Frames USGS has not scanned yet are
listed in `summary.json` with `downloadAvailable: false`; order them in
EarthExplorer and rerun once they are scanned.

## Outputs

Per point, in `<out>/<name>/`:

- `<name>_<frame>.tif`: the window as a GeoTIFF (WGS84, approximate affine
  georeferencing), full scan resolution
- `<name>_<frame>.png`: preview (max 4000 px)
- `<name>_<frame>_context.png`: ~30 km around the point with the window outlined
  in red and a north arrow in blue, in the film's own orientation
- `<name>_<frame>.json`: frame metadata, camera, orientation and how it was
  chosen (`usgs-overlay`, `camera-default` or `unverified`), pixel window,
  ground sample distance

`<out>/summary.json` lists every candidate frame per point and every crop made.

## Accuracy: read this before using the crops

The scans carry no georeferencing. The only geolocation USGS supplies is the
four corners of each frame's footprint, and the script maps a point onto the
film from those corners plus a panoramic camera model. Measured on real scans:

| Camera | Measured placement error | Default margin each side |
|---|---|---|
| ARGON | not measured (frame camera, ~30 m pixels) | 1 km |
| KH-4 (CORONA) | 1-5 km (Pentagon, two frames) | 5 km |
| KH-9 (HEXAGON) | ~7-10 km near the strip end (Ouarzazate) | 10 km |
| KH-7 (GAMBIT) | ~6 km (Rabat, 1966) | 6 km |
| KH-9 mapping camera | not measured | 2 km |

So the window is a 1 km box plus the margin: the target is inside it, **not at
its centre**. Find the exact spot in the preview by eye, or co-register the
GeoTIFF against modern imagery downstream. USGS's own browse overlays share
the same corner errors, so they cannot be used to tighten this.

Film orientation (which way is north) is not in the metadata either:

| Camera | Orientation used | Verified on |
|---|---|---|
| ARGON | TL=SE (180 deg) | DS09066A063MC071 vs USGS overlay (r = 0.74) |
| KH-4 forward | TL=NW (north-up) | Pentagon in DS1101-2157DF028 |
| KH-4 aft | TL=SE (180 deg) | Pentagon in DS1043-2157DA014 |
| KH-9 aft | TL=SE (180 deg) | El Mansour Eddahbi reservoir in D3C1212-100087A023 |
| KH-9 forward | **unknown**: all candidates are written | none yet |
| KH-7 GAMBIT | TL=SW | Rabat coast and Bou Regreg mouth in DZB00402700011H021001 |
| KH-9 mapping camera | **unknown**: all candidates are written | none yet |

USGS's own overlay of the KH-7 frame above is rotated 180 degrees, so for KH-7
the script ignores it and uses overlapping CORONA overlays instead.

If the overlay match is conclusive it overrides these defaults. Check the north
arrow in the context preview.

Other things the script handles: multi-part scans are joined in the order that
leaves no title-block gaps (KH-4 parts run `_d.._a` along the strip, KH-9 and
KH-7 parts in name order), and the photo frame is detected inside the film border.

KH-7 scans hold several film segments side by side, each with its own title
block, and a window can straddle two of them. The geometry treats the scan as
one strip, which is part of the ~6 km error. At 0.5 m/px a KH-7 window with the
6 km margin is ~38000 x 21000 px (a ~650 MB GeoTIFF); plan disk accordingly or
pass a smaller `--margin-m` once you know the offset.

## Earliest sources for Morocco

Declass 2 (`--dataset declassii`) is not searched by default. It holds the
earliest sharp imagery, KH-7 GAMBIT from 1966, downloadable now for Tangier
(1966-05-15, 1966-09-17), Rabat (1966-04-20) and Agadir (1966-04-20). It also
holds KH-9 mapping-camera frames (1976-79) for Ouarzazate, Errachidia and
Agadir. For accurately map-aligned imagery, Landsat MSS (60 m) covers every
example point from August 1972 (not handled by this script).

## Morocco coverage (1970s, example points)

From a dry run on 2026-10-27 (`examples/morocco_points.csv`):

| Point | 1970s frames | Downloadable | First pick |
|---|---|---|---|
| Tangier | 14 | 5 | KH-9 aft, 1972-07-12 |
| Rabat | 24 | 6 | KH-9 aft, 1979-05-13 |
| Casablanca | 8 | 0 | - |
| Fes | 16 | 3 | KH-4 aft, 1972-05-26 |
| Oujda | 12 | 0 | - |
| Marrakech | 9 | 4 | KH-9 aft, 1976-07-19 |
| Errachidia | 14 | 2 | KH-9 aft, 1976-01-24 |
| Ouarzazate | 13 | 7 | KH-9 aft, 1976-07-19 |
| Agadir | 5 | 3 | KH-9 aft, 1974-01-14 |
| Guelmim | 18 | 1 | KH-9 aft, 1974-01-14 |
| Laayoune | 21 | 0 | - |
| Dakhla | 8 | 0 | - |

Where nothing is downloadable, the frames exist but must be ordered for scanning
in EarthExplorer, or use Landsat MSS (60 m, `landsat_mss_c2_l1`, not handled by
this script).

## Tests

```
pip install pytest
python3 -m pytest -q tests                      # offline unit tests
M2M_USERNAME=... M2M_TOKEN=... python3 -m pytest -q tests/test_live.py -s   # live search
CORONA_LIVE_DOWNLOAD=1 ... python3 -m pytest -q tests/test_live.py -s       # live download + crop
```

The live tests call the real USGS M2M API and are skipped without credentials.
