#!/usr/bin/env python3
"""rbi_merger: merge RBI geodatabases from many zips into one GeoPackage.

Currently implemented (default run): unzip + manifest.

Step 1: each <name>.zip is extracted to .unzip/<name>/. A zip is skipped if
        .unzip/<name>/.extracted marker exists (written only after a full
        extraction, so an interrupted run is retried).
Step 2: every *.gdb found under .unzip/<name>/ is read with pyogrio; the
        manifest lists unique layer names and which zip/gdb each appears in.

Zips are read from ./rbi_zip next to this script; .unzip/ and .manifest.json
are written next to this script (rbi_zip/ is input only). A zip with no readable .gdb is skipped.

Typical workflow is 2 steps: --create-manifest first, then --merge.

Usage:
    ./rbi_merger.py                    # print .manifest.json (coloured); build it first if missing
    ./rbi_merger.py --create-manifest  # unzip (skipping done zips) + rebuild .manifest.json
    ./rbi_merger.py --force            # like --create-manifest, but re-extract every zip
    ./rbi_merger.py --clean            # delete .unzip/ and .manifest.json
    ./rbi_merger.py --merge            # merge into rbi_merged.gpkg (needs .unzip, .manifest.json,
                                       # clip_features.gpkg and a filled admin_boundary_map.json)
"""

import argparse
import json
import shutil
import subprocess
import sys
import warnings
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import geopandas
import pandas as pd
import pyogrio

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_ZIP_DIR = BASE_DIR / "rbi_zip"
UNZIP_DIRNAME = ".unzip"
MARKER_NAME = ".extracted"
MANIFEST_NAME = ".manifest.json"
CLIP_FEATURES_NAME = "clip_features.gpkg"
BOUNDARY_MAP_NAME = "admin_boundary_map.json"
OUTPUT_NAME = "rbi_merged.gpkg"
OUTPUT_CRS = "EPSG:4326"
# layers left out of the output (redundant with clip_features.gpkg)
SKIP_LAYER_PREFIXES = ("ADMINISTRASI_",)
# stale after clipping/exploding, so dropped
STALE_COLUMNS = ("shape_length", "shape_area")


def extract_zip(zip_path, dest_dir, force):
    """Extract zip_path into dest_dir. Returns 'extracted' or 'skipped'."""
    marker = dest_dir / MARKER_NAME
    if marker.exists() and not force:
        return "skipped"

    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        members = [m for m in zf.namelist() if not m.startswith("__MACOSX/")]
        # extract() sanitises member paths, so entries cannot escape dest_dir
        for member in members:
            zf.extract(member, dest_dir)
    marker.write_text(datetime.now(timezone.utc).isoformat() + "\n")
    return "extracted"


def find_geodatabases(root):
    """All *.gdb directories under root, sorted."""
    return sorted(p for p in root.rglob("*.gdb") if p.is_dir())


def read_layers(gdb_path):
    """Sorted layer names of one gdb."""
    return sorted(str(name) for name, _geom in pyogrio.list_layers(gdb_path))


def build_manifest(zip_dir, unzip_dir, force):
    zips = sorted(zip_dir.glob("*.zip"))
    if not zips:
        sys.exit("no .zip files found in " + str(zip_dir))

    per_zip = {}      # zip name -> list of {name, path, layers}
    layer_zips = {}   # layer name -> set of zip names
    warnings = []

    for zip_path in zips:
        zip_name = zip_path.name
        dest = unzip_dir / zip_path.stem

        try:
            status = extract_zip(zip_path, dest, force)
        except (zipfile.BadZipFile, OSError) as exc:
            msg = "%s: extraction failed: %s" % (zip_name, exc)
            print("  " + msg, file=sys.stderr)
            warnings.append(msg)
            continue
        print("[%s] %s" % (status, zip_name))

        gdbs = find_geodatabases(dest)
        if not gdbs:
            msg = "%s: no .gdb found, skipped" % zip_name
            print("  " + msg, file=sys.stderr)
            warnings.append(msg)
            continue

        gdb_entries = []
        for gdb in gdbs:
            gdb_rel = gdb.relative_to(unzip_dir).as_posix()
            try:
                names = read_layers(gdb)
            except Exception as exc:  # pyogrio raises DataSourceError etc.
                msg = "%s: cannot read %s: %s, skipped" % (zip_name, gdb_rel, exc)
                print("  " + msg, file=sys.stderr)
                warnings.append(msg)
                continue
            gdb_entries.append({"name": gdb.name, "path": gdb_rel, "layers": names})
            for name in names:
                layer_zips.setdefault(name, set()).add(zip_name)

        if gdb_entries:
            per_zip[zip_name] = gdb_entries

    # key order matters: print_manifest shows it as is, conclusion goes last
    return {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "geodatabases_by_zip": per_zip,
        "zips_by_layer": {n: sorted(z) for n, z in sorted(layer_zips.items())},
        "warnings": warnings,
        "unique_layers": sorted(layer_zips),
    }


WORKFLOW_HELP = """\
typical workflow (2 steps):
  1. ./rbi_merger.py --create-manifest
       unzips rbi_zip/*.zip into .unzip/ and writes .manifest.json
  2. ./rbi_merger.py --merge
       first run creates admin_boundary_map.json; fill in each gdb's NAMOBJ,
       then run --merge again to write rbi_merged.gpkg

other:
  ./rbi_merger.py            print .manifest.json (needs step 1)
  ./rbi_merger.py --clean    remove .unzip/ and .manifest.json
"""


def print_manifest(path):
    """Print the manifest; coloured through jq when available and on a tty."""
    jq = shutil.which("jq")
    if jq:
        color = "-C" if sys.stdout.isatty() else "-M"
        subprocess.run([jq, color, ".", str(path)], check=False)
    else:
        print("(jq not found, printing plain JSON; brew install jq for colours)",
              file=sys.stderr)
        print(path.read_text(), end="")


def read_namobj_values(clip_path):
    """Sorted unique NAMOBJ values of the clip features gpkg."""
    df = pyogrio.read_dataframe(clip_path, columns=["NAMOBJ"], read_geometry=False)
    return sorted(df["NAMOBJ"].dropna().unique())


def manifest_gdb_paths(manifest):
    """gdb paths (relative to .unzip/) listed in the manifest, in manifest order."""
    paths = []
    for entries in manifest["geodatabases_by_zip"].values():
        for entry in entries:
            paths.append(entry["path"])
    return paths


def load_boundary_map(map_path, gdb_paths, namobj_values):
    """Validate admin_boundary_map.json. Returns {gdb path: NAMOBJ} or exits."""
    try:
        data = json.loads(map_path.read_text())
        mapping = data["boundary_map"]
    except (ValueError, KeyError) as exc:
        sys.exit("%s is not valid (%s). Fix it, or delete it to regenerate the template."
                 % (map_path.name, exc))

    problems = []
    for path in gdb_paths:
        if path not in mapping:
            problems.append("missing key (gdb added since template was made): " + path)
        elif not str(mapping[path]).strip():
            problems.append("empty value: " + path)
        elif mapping[path] not in namobj_values:
            problems.append("unknown NAMOBJ %r for %s" % (mapping[path], path))
    if problems:
        sys.exit("refusing to merge, fix %s:\n  " % map_path.name + "\n  ".join(problems))

    for path in mapping:
        if path not in gdb_paths:
            print("warning: %s has stale key not in manifest: %s" % (map_path.name, path),
                  file=sys.stderr)
    used = [mapping[p] for p in gdb_paths]
    for name in sorted(set(used)):
        if used.count(name) > 1:
            print("warning: NAMOBJ %r is mapped by more than one gdb" % name, file=sys.stderr)
    return {p: mapping[p] for p in gdb_paths}


def load_clip_masks(clip_path):
    """{NAMOBJ: 2D valid geometry in OUTPUT_CRS}; rows with the same NAMOBJ are unioned."""
    gdf = geopandas.read_file(clip_path, engine="pyogrio")
    gdf = gdf.to_crs(OUTPUT_CRS)
    gdf["geometry"] = gdf.geometry.force_2d().make_valid()
    return {name: grp.geometry.union_all() for name, grp in gdf.groupby("NAMOBJ")}


def prepare_layer(gdf, mask, zip_name):
    """Reproject, drop Z, explode to simple geometries and clip to mask.

    Returns None if nothing is left.
    """
    gdf = gdf[~gdf.geometry.isna() & ~gdf.geometry.is_empty]
    if gdf.empty:
        return None
    if gdf.crs is None:
        raise ValueError("layer has no CRS")
    gdf = gdf.to_crs(OUTPUT_CRS)
    gdf["geometry"] = gdf.geometry.force_2d().make_valid()
    gdf = gdf.explode(ignore_index=True)
    # keep_geom_type drops slivers of another type created along the clip edge
    gdf = geopandas.clip(gdf, mask, keep_geom_type=True)
    gdf = gdf[~gdf.geometry.is_empty]
    if gdf.empty:
        return None
    gdf = gdf.explode(ignore_index=True)
    drop = [c for c in gdf.columns if c.lower() in STALE_COLUMNS]
    gdf = gdf.drop(columns=drop)
    gdf.insert(0, "source_zip", zip_name)
    return gdf


def unify_dtypes(frames):
    """Cast columns whose dtype differs between frames to str so concat/write is safe."""
    seen = {}
    for frame in frames:
        for col in frame.columns:
            if col != "geometry":
                seen.setdefault(col, set()).add(str(frame[col].dtype))
    mixed = [col for col, dtypes in seen.items() if len(dtypes) > 1]
    for frame in frames:
        for col in mixed:
            if col in frame.columns:
                frame[col] = frame[col].map(lambda v: None if pd.isna(v) else str(v))
    return mixed


def layer_sources(layer, manifest):
    """[(zip name, gdb relative path)] that contain this layer, sorted."""
    sources = []
    for zip_name in manifest["zips_by_layer"][layer]:
        for entry in manifest["geodatabases_by_zip"][zip_name]:
            if layer in entry["layers"]:
                sources.append((zip_name, entry["path"]))
    return sorted(sources)


def merge_layers(unzip_dir, manifest, boundary_map, clip_path, out_path):
    masks = load_clip_masks(clip_path)
    if out_path.exists():
        print("replacing existing " + out_path.name)
        out_path.unlink()

    warnings.filterwarnings("ignore", message="Measured \\(M\\) geometry types")
    layers = [n for n in manifest["unique_layers"]
              if not n.startswith(SKIP_LAYER_PREFIXES)]
    skipped_admin = len(manifest["unique_layers"]) - len(layers)
    print("merging %d layers (%d ADMINISTRASI_* skipped) -> %s"
          % (len(layers), skipped_admin, out_path.name))

    written = 0
    for i, layer in enumerate(layers, 1):
        frames = []
        for zip_name, gdb_rel in layer_sources(layer, manifest):
            mask = masks[boundary_map[gdb_rel]]
            try:
                gdf = pyogrio.read_dataframe(unzip_dir / gdb_rel, layer=layer)
                if not isinstance(gdf, geopandas.GeoDataFrame):
                    raise ValueError("no geometry column (table)")
                gdf = prepare_layer(gdf, mask, zip_name)
            except Exception as exc:
                print("  warning: %s / %s skipped: %s" % (layer, zip_name, exc),
                      file=sys.stderr)
                continue
            if gdf is not None:
                frames.append(gdf)

        if not frames:
            print("[%d/%d] %s: no features after clipping, not written" % (i, len(layers), layer))
            continue

        mixed = unify_dtypes(frames)
        merged = geopandas.GeoDataFrame(
            pd.concat(frames, ignore_index=True), geometry="geometry", crs=OUTPUT_CRS)
        if mixed:
            print("  note: %s: cast to text (dtype differs across zips): %s"
                  % (layer, ", ".join(mixed)))

        geom_types = sorted(merged.geom_type.unique())
        for gtype in geom_types:
            part = merged[merged.geom_type == gtype].reset_index(drop=True)
            name = layer if len(geom_types) == 1 else "%s_%s" % (layer, gtype.lower())
            part.to_file(out_path, layer=name, driver="GPKG", engine="pyogrio")
            written += 1
            print("[%d/%d] %s: %d features from %d zips"
                  % (i, len(layers), name, len(part), len(frames)))
    print("done: %d layers written to %s" % (written, out_path))


def run_merge():
    unzip_dir = BASE_DIR / UNZIP_DIRNAME
    manifest_path = BASE_DIR / MANIFEST_NAME
    clip_path = BASE_DIR / CLIP_FEATURES_NAME
    map_path = BASE_DIR / BOUNDARY_MAP_NAME

    missing = [str(p.name) for p in (unzip_dir, manifest_path, clip_path) if not p.exists()]
    if missing:
        sys.exit("--merge needs %s (run ./rbi_merger.py --create-manifest first for "
                 ".unzip and .manifest.json)" % ", ".join(missing))

    manifest = json.loads(manifest_path.read_text())
    gdb_paths = manifest_gdb_paths(manifest)
    namobj_values = read_namobj_values(clip_path)

    if not map_path.exists():
        template = {
            "available_namobj": namobj_values,
            "boundary_map": {path: "" for path in gdb_paths},
        }
        map_path.write_text(json.dumps(template, indent=2, ensure_ascii=False) + "\n")
        sys.exit("created %s\nOpen it in a text editor and set each value in "
                 "\"boundary_map\" to one of the NAMOBJ values listed under "
                 "\"available_namobj\", then rerun ./rbi_merger.py --merge" % map_path)

    boundary_map = load_boundary_map(map_path, gdb_paths, namobj_values)
    print("%s ok: %d gdbs mapped" % (map_path.name, len(boundary_map)))
    merge_layers(unzip_dir, manifest, boundary_map, clip_path, BASE_DIR / OUTPUT_NAME)


def clean():
    """Delete the extracted .unzip/ folder and .manifest.json (both regenerable)."""
    unzip_dir = BASE_DIR / UNZIP_DIRNAME
    manifest_path = BASE_DIR / MANIFEST_NAME
    removed = False
    if unzip_dir.is_symlink() or unzip_dir.is_file():
        unzip_dir.unlink()
        removed = True
    elif unzip_dir.is_dir():
        shutil.rmtree(unzip_dir)
        removed = True
        print("removed " + UNZIP_DIRNAME + "/")
    if manifest_path.exists():
        manifest_path.unlink()
        removed = True
        print("removed " + MANIFEST_NAME)
    if not removed:
        print("nothing to clean")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=WORKFLOW_HELP,
    )
    parser.add_argument("--zip-dir", type=Path, default=DEFAULT_ZIP_DIR,
                        help="folder with the .zip files (default: ./rbi_zip)")
    parser.add_argument("--create-manifest", action="store_true",
                        help="(re)create .manifest.json, unzipping zips not yet extracted")
    parser.add_argument("--clean", action="store_true",
                        help="delete .unzip/ and .manifest.json, then exit")
    parser.add_argument("--merge", action="store_true",
                        help="merge all gdb layers into rbi_merged.gpkg, deduped by clipping")
    parser.add_argument("--force", action="store_true",
                        help="re-extract zips even if already extracted (implies --create-manifest)")
    args = parser.parse_args()

    if args.clean:
        clean()
        return

    if args.merge:
        run_merge()
        return

    out_path = BASE_DIR / MANIFEST_NAME
    if out_path.exists() and not (args.create_manifest or args.force):
        print_manifest(out_path)
        return

    zip_dir = args.zip_dir.resolve()
    unzip_dir = BASE_DIR / UNZIP_DIRNAME
    unzip_dir.mkdir(exist_ok=True)

    manifest = build_manifest(zip_dir, unzip_dir, args.force)

    out_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print("wrote %s: %d zips, %d unique layers, %d warnings" % (
        out_path.name, len(manifest["geodatabases_by_zip"]),
        len(manifest["unique_layers"]), len(manifest["warnings"])))


if __name__ == "__main__":
    main()
