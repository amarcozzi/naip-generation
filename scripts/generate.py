"""
Unified NAIP dataset generation script.

This script consolidates the behavior of:
 - generate_coconino_chunk_dataset.py
 - generate_coconino_nf_dataset.py
 - generate_coconino_single_tile.py

into a single CLI using argparse. It calls the logic in naip.py to
search, download, and sample NAIP tiles into an image dataset.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Optional

import geopandas as gpd


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a NAIP image dataset using data/naip.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Dataset parameters
    parser.add_argument("dataset_name", type=str, help="Name of the dataset to create")
    parser.add_argument("--train", type=int, default=1000, help="Number of training patches")
    parser.add_argument("--test", type=int, default=100, help="Number of test patches")
    parser.add_argument(
        "--val", dest="validation", type=int, default=100, help="Number of validation patches"
    )
    parser.add_argument("--patch-size", type=int, default=512, help="Patch size in pixels")

    # ROI options
    roi = parser.add_argument_group("ROI options (provide one)")
    roi.add_argument(
        "--roi-file",
        type=str,
        default=None,
        required=True,
        help="Path to a GeoJSON/GeoPackage/Shapefile defining the region of interest",
    )
    roi.add_argument(
        "--roi-filter-field",
        type=str,
        default=None,
        help="Optional attribute field name to filter the ROI file (e.g., 'Name')",
    )
    roi.add_argument(
        "--roi-filter-value",
        type=str,
        default=None,
        help="Value to match in --roi-filter-field (e.g., 'Coconino National Forest')",
    )
    roi.add_argument(
        "--point",
        type=str,
        default=None,
        help="A single point ROI as 'lon,lat' (e.g., -111.465107,34.849059)",
    )

    # Filters
    parser.add_argument("--year", type=int, default=None, help="Filter NAIP tiles by year")
    parser.add_argument(
        "--gsd",
        type=float,
        default=None,
        help="Filter NAIP tiles by GSD (m), e.g., 0.6 or 0.3",
    )

    # Execution
    parser.add_argument("--num-workers", type=int, default=4, help="Parallel worker count")
    parser.add_argument(
        "--continue-existing",
        action="store_true",
        help="Continue adding samples to an existing dataset directory",
    )
    parser.add_argument(
        "--include-nir",
        action="store_true",
        help="Include NIR band when present (4-band patches). If omitted, RGB only.",
    )
    parser.add_argument(
        "--png",
        action="store_true",
        help="Save patches as PNG instead of GeoTIFF (default is GeoTIFF)",
    )

    # Paths
    parser.add_argument(
        "--data-path",
        type=str,
        default=str((Path.cwd().parent / "data").resolve()),
        help="Root path to store generated datasets (DATA_PATH)",
    )

    return parser.parse_args()


def _build_roi(args: argparse.Namespace) -> gpd.GeoDataFrame:
    # If a point string is provided, build a single-point GeoDataFrame
    if args.point:
        try:
            lon_str, lat_str = [s.strip() for s in args.point.split(",", 1)]
            lon, lat = float(lon_str), float(lat_str)
        except Exception as e:
            raise ValueError(
                f"--point must be in 'lon,lat' format (e.g., -111.465107,34.849059): {e}"
            )

        return gpd.GeoDataFrame.from_features(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {},
                        "geometry": {"type": "Point", "coordinates": [lon, lat]},
                    }
                ],
            }
        )

    # Else, require an ROI file
    if not args.roi_file:
        raise ValueError(
            "You must provide either --point lon,lat or --roi-file <path> to define the ROI"
        )

    roi_gdf = gpd.read_file(args.roi_file)
    if args.roi_filter_field and args.roi_filter_value is not None:
        if args.roi_filter_field not in roi_gdf.columns:
            raise ValueError(
                f"ROI filter field '{args.roi_filter_field}' not found in {args.roi_file}. Available: {list(roi_gdf.columns)}"
            )
        roi_gdf = roi_gdf[roi_gdf[args.roi_filter_field] == args.roi_filter_value]
        if roi_gdf.empty:
            raise ValueError(
                f"No ROI features matched {args.roi_filter_field} == {args.roi_filter_value}"
            )

    return roi_gdf


def main() -> None:
    args = _parse_args()

    # Ensure we can import naip.py from this directory when running as a script
    this_dir = Path(__file__).parent.resolve()
    if str(this_dir) not in sys.path:
        sys.path.append(str(this_dir))

    import naip  # noqa: WPS433 (import after sys.path mutation)
    from naip import ImageDataset

    # Configure paths expected by naip.py
    data_path = Path(args.data_path).resolve()
    data_path.mkdir(parents=True, exist_ok=True)

    # No global path injection; paths are passed explicitly to API

    roi = _build_roi(args)

    print(
        f"Generating dataset '{args.dataset_name}' -> train={args.train}, test={args.test}, val={args.validation}"
    )

    datasets = ImageDataset.generate_naip_image_dataset(
        dataset_name=args.dataset_name,
        data_path=data_path,
        roi=roi,
        num_train_images=args.train,
        num_test_images=args.test,
        num_validation_images=args.validation,
        year_filter=args.year,
        gsd_filter=args.gsd,
        num_workers=args.num_workers,
        continue_existing=args.continue_existing,
        patch_size=args.patch_size,
        include_nir_band=args.include_nir,
        as_tiff=not args.png,
    )

    # Brief summary
    created = {mode: len(ds.image_filenames) for mode, ds in datasets.items()}
    if created:
        print(f"Created patches: {created}")
    print("All done!")


if __name__ == "__main__":
    main()
