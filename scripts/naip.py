# Core imports
from __future__ import annotations
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import os
from pathlib import Path
import random
import shutil
import time
from typing import Dict, List

# External imports
import geopandas as gpd
import pandas as pd
import pandera.pandas as pa
import numpy as np
from pandera.typing import Series
from pandera.typing.geopandas import GeoSeries, GeoDataFrame
import planetary_computer
import pystac_client
import requests
import rasterio
from rasterio.errors import RasterioIOError
from rasterio.windows import Window
from tqdm import tqdm
import uuid
from PIL import Image
import rioxarray as rxr

PLANETARY_COMPUTER_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"


@dataclass
class Location:
    longitude: float
    latitude: float


class TileSourceSchema(pa.DataFrameModel):
    id: Series[str] = pa.Field()
    geometry: GeoSeries = pa.Field()
    year: Series[int] = pa.Field(gt=2000, coerce=True)
    state: Series[str] = pa.Field(str_length={"min_value": 2, "max_value": 2})
    gsd: Series[float] = pa.Field(gt=0)
    source_url: Series[str] = pa.Field()

class ImageDataset:
    def __init__(self, dataset_name: str, mode: str, data_root: Path):
        self.project_name = dataset_name
        self.mode = mode
        self.data_root = Path(data_root)
        self.image_filenames = []
        self.tile_sources = []

    @classmethod
    def generate_naip_image_dataset(
        cls,
        dataset_name: str,
        data_path: Path,
        roi: gpd.GeoDataFrame,
        num_train_images: int = 0,
        num_test_images: int = 0,
        num_validation_images: int = 0,
        year_filter: int = None,
        gsd_filter: float = None,
        num_workers: int = 1,
        continue_existing: bool = False,
        patch_size: int = 256,
        include_nir_band: bool = True,
        as_tiff: bool = True,
    ) -> Dict[str, ImageDataset]:
        """
        Generate a dataset from NAIP imagery.

        Parameters
        ----------
        dataset_name : str
            Name of the dataset to be created
        data_path : Path
            Root directory where the dataset will be written
        roi : gpd.GeoDataFrame
            Region of interest to search for NAIP tiles
        num_train_images : int, optional
            Number of training images to generate, by default 0
        num_test_images : int, optional
            Number of test images to generate, by default 0
        num_validation_images : int, optional
            Number of validation images to generate, by default 0
        year_filter : int, optional
            Filter NAIP tiles by year, by default None
        gsd_filter : float, optional
            Filter NAIP tiles by ground sample distance (GSD) in meters, by default None
        num_workers : int, optional
            Number of worker processes for parallel processing, by default 1
        continue_existing : bool, optional
            Whether to continue generating images for an existing dataset, by default False
        patch_size : int, optional
            Size of image patches in pixels, by default 256
        include_nir_band : bool, optional
            Whether to include the NIR band in the generated patches, by default True
        as_tiff : bool, optional
            Whether to save the patches as TIFF files, by default True. If False, saves as PNG.

        Returns
        -------
        Dict[str, ImageDataset]
            Dictionary of created datasets by mode (train, test, validation)
        """
        datasets = {}
        modes = {
            "train": num_train_images,
            "test": num_test_images,
            "validation": num_validation_images,
        }

        # Dataset-local tiles directory (no global cache)
        data_path = Path(data_path)
        dataset_dir = data_path / dataset_name
        tiles_dir = dataset_dir / "naip_tiles"
        tiles_dir.mkdir(parents=True, exist_ok=True)

        # Search for NAIP tiles intersecting with the ROI
        print(f"Searching for NAIP tiles intersecting with ROI...")
        tile_source = search(roi)
        print(f"Found {len(tile_source)} NAIP tiles intersecting with ROI.")

        # Apply year filter if specified
        if year_filter:
            print(f"Filtering tiles by year: {year_filter}")
            tile_source = tile_source[tile_source["year"] == year_filter]
            print(f"Remaining tiles after filtering by year: {len(tile_source)}")

            if len(tile_source) == 0:
                raise ValueError(f"No NAIP tiles found for year {year_filter}")

        # Apply GSD filter if specified
        if gsd_filter:
            print(f"Filtering tiles by GSD: {gsd_filter}m")
            tile_source = tile_source[tile_source["gsd"].round(1) == gsd_filter]
            print(f"Remaining tiles after filtering by GSD: {len(tile_source)}")

            if len(tile_source) == 0:
                raise ValueError(f"No NAIP tiles found with GSD == {gsd_filter}m")

        # Ensure we have tiles with consistent resolution (GSD)
        if len(tile_source["gsd"].round(1).unique()) > 1:
            print(f"Warning: Multiple resolutions found: {tile_source['gsd'].unique()}")
            # Use the most common resolution
            most_common_gsd = tile_source["gsd"].value_counts().idxmax()
            tile_source = tile_source[tile_source["gsd"] == most_common_gsd]
            print(f"Selected resolution: {most_common_gsd}m")
        else:
            print(f"NAIP tiles have resolution: {tile_source['gsd'].unique()[0]}m")

        # Check which tiles already exist in the dataset tiles directory and validate them
        existing_tiles = {}  # Maps tile_id to path
        corrupted_tiles = []

        for tile_path in tiles_dir.glob("*.tif"):
            tile_id = tile_path.stem
            if validate_tiff(tile_path):
                existing_tiles[tile_id] = tile_path
            else:
                corrupted_tiles.append(tile_path)

        # Remove corrupted tiles
        if corrupted_tiles:
            print(f"Found {len(corrupted_tiles)} corrupted tiles in dataset tiles dir. Removing...")
            for path in corrupted_tiles:
                try:
                    path.unlink()
                except OSError as e:
                    print(f"Error removing corrupted tile {path}: {e}")

        # Identify tiles that need to be downloaded
        tiles_to_download = []
        tile_paths = {}

        for _, row in tile_source.iterrows():
            tile_id = row["id"]
            tile_path = tiles_dir / f"{tile_id}.tif"
            tile_paths[tile_id] = tile_path

            if tile_id not in existing_tiles:
                tiles_to_download.append((row["source_url"], tile_path))

        print(
            f"Found {len(set(existing_tiles.keys()).intersection(set(tile_paths.keys())))} of {len(tile_source)} tiles already present for this dataset."
        )
        print(f"Need to download {len(tiles_to_download)} new tiles.")

        # Optionally save the complete tile information to dataset directory
        # This is not a cache; it records tiles discovered for this dataset
        tile_info_path = tiles_dir / "tile_info.geojson"
        if not tile_info_path.exists() or not continue_existing:
            tile_source.to_file(tile_info_path, driver="GeoJSON")
        else:
            existing_tile_info = gpd.read_file(tile_info_path)
            existing_ids = set(existing_tile_info["id"])
            new_tiles = tile_source[~tile_source["id"].isin(existing_ids)]
            if not new_tiles.empty:
                updated_tile_info = gpd.GeoDataFrame(pd.concat([existing_tile_info, new_tiles], ignore_index=True))
                updated_tile_info.to_file(tile_info_path, driver="GeoJSON")

        # Download the tiles with retry logic
        successful_downloads = []
        failed_downloads = []

        if tiles_to_download:
            print(
                f"Downloading {len(tiles_to_download)} NAIP tiles to {tiles_dir}..."
            )

            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = []

                for url, output_path in tiles_to_download:
                    futures.append(executor.submit(download_tile, url, output_path))

                with tqdm(total=len(futures), desc="Downloading tiles") as pbar:
                    for future in as_completed(futures):
                        try:
                            # download_tile returns a Path on success or raises an exception
                            output_path = future.result()
                            successful_downloads.append(output_path)
                            # Add to existing tiles for sampling
                            existing_tiles[output_path.stem] = output_path
                        except Exception as e:
                            failed_downloads.append(str(e))
                        pbar.update(1)

            print(f"Successfully downloaded {len(successful_downloads)} new tiles.")

            if failed_downloads:
                print(
                    f"Failed to download {len(failed_downloads)} tiles after multiple retries."
                )
                print("The dataset will be created with the available tiles.")

        # Create datasets for each mode
        used_tiles = set()

        for mode, num_images in modes.items():
            if num_images <= 0:
                continue

            dataset = cls(dataset_name, mode, data_root=data_path)
            datasets[mode] = dataset

            # Create output directory
            output_dir = data_path / dataset_name / mode
            output_dir.mkdir(parents=True, exist_ok=True)

            # Clear existing files if not continuing
            if not continue_existing:
                for filename in output_dir.glob("*.tif"):
                    filename.unlink()
                for filename in output_dir.glob("*.png"):
                    filename.unlink()

            print(f"Sampling {num_images} {mode} patches...")

            # Get all available valid tile files from cache
            tile_files = [
                path
                for tile_id, path in existing_tiles.items()
                if tile_id in tile_paths
            ]

            if not tile_files:
                raise ValueError(
                    f"No valid NAIP tiles available for sampling. "
                    f"Downloaded: {len(successful_downloads)}, "
                    f"Failed: {len(failed_downloads)}"
                )

            # Sample the patches
            sampled_files = []
            patch_bboxes = []  # Store bounding box info for each patch
            mode_used_tiles = set()

            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = []
                sampling_attempts = 0
                max_sampling_attempts = num_images * 2  # Allow for some failures

                while (
                    len(futures) < num_images
                    and sampling_attempts < max_sampling_attempts
                ):
                    tile_path = random.choice(tile_files)
                    tile_id = tile_path.stem
                    mode_used_tiles.add(tile_id)

                    futures.append(
                        executor.submit(
                            sample_and_save_patch,
                            tile_path,
                            patch_size,
                            output_dir,
                            include_nir_band=include_nir_band,
                            as_tiff=as_tiff,
                        )
                    )
                    sampling_attempts += 1

                with tqdm(
                    total=len(futures), desc=f"Sampling and saving {mode} patches"
                ) as pbar:
                    for future in as_completed(futures):
                        try:
                            filepath, bbox_info = future.result()
                            if filepath:
                                sampled_files.append(filepath)
                                if bbox_info:
                                    patch_bboxes.append(bbox_info)
                        except Exception as e:
                            print(f"Error during sampling: {e}")
                        pbar.update(1)

            dataset.image_filenames = [Path(f).name for f in sampled_files]
            used_tiles.update(mode_used_tiles)

            print(
                f"Generated {len(sampled_files)} {mode} patches using {len(mode_used_tiles)} unique tiles."
            )

            # If we didn't get enough samples, warn the user
            if len(sampled_files) < num_images:
                print(
                    f"Warning: Only generated {len(sampled_files)}/{num_images} requested samples "
                    f"for {mode}. This may be due to corrupted tiles or sampling failures."
                )

            # Save bounding boxes as GeoJSON for each mode
            if patch_bboxes:
                # Convert the bounding boxes to a GeoDataFrame
                features = []
                for bbox_info in patch_bboxes:
                    left, bottom, right, top = bbox_info["bounds"]
                    # Create a polygon from the bounds
                    polygon = {
                        "type": "Polygon",
                        "coordinates": [
                            [
                                [left, bottom],
                                [right, bottom],
                                [right, top],
                                [left, top],
                                [left, bottom],
                            ]
                        ],
                    }

                    feature = {
                        "type": "Feature",
                        "properties": {
                            "file_name": bbox_info["file_name"],
                            "tile_id": bbox_info["tile_id"],
                        },
                        "geometry": polygon,
                    }
                    features.append(feature)

                # Create a GeoDataFrame
                if features:
                    patches_gdf = gpd.GeoDataFrame.from_features(
                        {"type": "FeatureCollection", "features": features}
                    )

                    # Set the CRS for the GeoDataFrame (using the CRS from the first patch)
                    if patches_gdf.shape[0] > 0:
                        patches_gdf.crs = patch_bboxes[0]["crs"]

                        # Save the GeoDataFrame as a GeoJSON file
                        patches_geojson_path = (
                            data_path / dataset_name / mode / "patch_locations.geojson"
                        )
                        patches_gdf.to_file(str(patches_geojson_path), driver="GeoJSON")
                        print(
                            f"Saved patch locations for {mode} to {patches_geojson_path}"
                        )

        # Create dataset-specific tile info (subset of used tiles)
        if used_tiles:
            dataset_dir = data_path / dataset_name
            dataset_dir.mkdir(parents=True, exist_ok=True)

            used_tile_source = tile_source[tile_source["id"].isin(used_tiles)]
            used_tile_source.to_file(
                dataset_dir / "used_naip_tiles.geojson", driver="GeoJSON"
            )

            print(f"Dataset uses {len(used_tiles)} unique NAIP tiles.")

        return datasets

    def get_image_paths(self):
        """
        Get the full paths to all images in this dataset.

        Returns
        -------
        List[Path]
            List of file paths
        """
        image_dir = self.data_root / self.project_name / self.mode
        return [image_dir / filename for filename in self.image_filenames]


def validate_tiff(file_path: Path) -> bool:
    """
    Validate that a TIFF file can be opened with rasterio.

    Parameters
    ----------
    file_path : Path
        Path to the TIFF file

    Returns
    -------
    bool
        True if the file is valid, False otherwise
    """
    try:
        with rasterio.open(file_path) as src:
            # Try to read metadata to validate
            _ = src.meta
            # Try to read a small window to ensure data is valid
            _ = src.read(1, window=Window(0, 0, 10, 10))

        return True
    except (RasterioIOError, ValueError, KeyError, Exception) as e:
        print(f"Invalid TIFF file {file_path}: {str(e)}")
        return False


@pa.check_types
def search(gdf: GeoDataFrame) -> GeoDataFrame[TileSourceSchema]:
    """
    Search for NAIP tile IDs by intersection with a GeoJSON.

    Parameters
    ----------
    gdf : GeoDataFrame
        A GeoDataFrame containing the geometry to intersect with NAIP tiles.

    Returns
    -------
    GeoDataFrame[TileSourceSchema]
        A GeoDataFrame containing information about the intersecting NAIP tiles.

    Notes
    -----
    This function currently only uses the first geometry if multiple are present.
    TODO: Use gdf.dissolve to combine all geometries.
    """

    # Extract the geometry from the geodataframe. If there are multiple
    # geoemtries, we only use the first.
    # TODO: Use gdf.dissolve to combine all geometries
    geometry = gdf.__geo_interface__["features"][0]["geometry"]

    # Connect to the Planetary Computer STAC client
    catalog = pystac_client.Client.open(
        PLANETARY_COMPUTER_URL,
        modifier=planetary_computer.sign_inplace,
    )

    search = catalog.search(collections=["naip"], intersects=geometry)

    # Only the properties and geometry fields are retained when we create a
    # geodataframe. So, we need to copy the image href over to the properties
    # key.
    feature_collection = search.item_collection().to_dict()
    for feature in feature_collection["features"]:
        feature["properties"]["id"] = feature["id"]
        feature["properties"]["source_url"] = feature["assets"]["image"]["href"]

    # STAC geometries are lon/lat (WGS84). Ensure CRS is set so outputs carry projection info.
    tile_source = gpd.GeoDataFrame.from_features(
        feature_collection, crs="EPSG:4326"
    )

    tile_source.rename(
        columns={"naip:year": "year", "naip:state": "state"},
        inplace=True,
    )

    return GeoDataFrame[TileSourceSchema](tile_source)


# def download_tile(args: tuple[str, str]) -> tuple[str, bool]:
#     """
#     Download a single NAIP tile.
#
#     Parameters
#     ----------
#     args : tuple[str, str]
#         A tuple containing the URL of the tile to download and the output file path.
#
#     Returns
#     -------
#     tuple[str, bool]
#         A tuple containing the output file path and a boolean indicating success.
#     """
#     url, output_file = args
#     Path(output_file).parent.mkdir(parents=True, exist_ok=True)
#     if not os.path.exists(output_file):
#         try:
#             response = requests.get(url, stream=True)
#             response.raise_for_status()
#             with open(output_file, "wb") as f:
#                 for chunk in response.iter_content(chunk_size=8192):
#                     f.write(chunk)
#             return (output_file, True)
#         except requests.RequestException as e:
#             print(f"Error downloading {url}: {str(e)}. Retrying...")
#             os.unlink(output_file)
#             # return (output_file, False)
#             return download_tile(args)
#     return (output_file, True)


def download_tile(
    tile_url: str, output_path: Path, max_retries: int = 3, min_file_size: int = 10000
) -> Path:
    """
    Download a NAIP tile with retry logic and validation.

    Parameters
    ----------
    tile_url : str
        URL of the NAIP tile to download
    output_path : Path
        Path to save the downloaded tile
    max_retries : int, optional
        Maximum number of retry attempts, by default 3
    min_file_size : int, optional
        Minimum expected file size in bytes, by default 10KB

    Returns
    -------
    Tuple[Path, bool]
        Path to the downloaded file and success status
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for attempt in range(max_retries):
        try:
            # Remove any existing partial download
            if output_path.exists():
                if validate_tiff(output_path):
                    return output_path
                else:
                    output_path.unlink()

            # Create a temporary file for downloading
            temp_path = output_path.with_suffix(".tmp")
            if temp_path.exists():
                temp_path.unlink()

            # Download to temporary file
            with requests.get(tile_url, stream=True) as response:
                response.raise_for_status()
                with open(temp_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)

            # Check if file is too small (likely incomplete)
            if temp_path.stat().st_size < min_file_size:
                raise ValueError(
                    f"Downloaded file is too small: {temp_path.stat().st_size} bytes"
                )

            # Move to final location
            shutil.move(temp_path, output_path)

            # Validate the downloaded file
            if validate_tiff(output_path):
                return output_path
            else:
                output_path.unlink()
                raise ValueError(f"Downloaded file failed validation")

        except Exception as e:
            print(
                f"Error downloading {tile_url} (attempt {attempt + 1}/{max_retries}): {str(e)}"
            )
            if temp_path.exists():
                temp_path.unlink()
            if output_path.exists():
                output_path.unlink()

            # Wait before retrying (exponential backoff)
            if attempt < max_retries - 1:
                time.sleep(2**attempt)

    # print(f"Failed to download {tile_url} after {max_retries} attempts")
    # return output_path, False
    raise ValueError(f"Failed to download {tile_url} after {max_retries} attempts")


def sample_and_save_patch(
    tile_path: Path,
    patch_size: int,
    output_dir: Path,
    include_nir_band: bool = True,
    as_tiff: bool = True,
) -> tuple[Path | None, dict | None]:
    """
    Sample a random patch from a NAIP tile and save it.

    Parameters
    ----------
    tile_path: Path
        Path to the NAIP tile
    patch_size: int
        Size of the patch to sample in pixels
    output_dir: Path
        Directory to save the sampled patch
    include_nir_band: bool, optional
        Whether to include the NIR band in the generated patches, by default True
    as_tiff: bool, optional
        Whether to save the patches as TIFF files, by default True. If False, saves as PNG.

    Returns
    -------
    tuple[Path | None, dict | None]
        Path to the saved file and bounding box info if successful, None values otherwise
    """
    # Choose the appropriate function based on as_tiff parameter
    if as_tiff:
        return sample_and_save_patch_tiff(
            tile_path, patch_size, output_dir, include_nir_band
        )
    else:
        return sample_and_save_patch_png(
            tile_path, patch_size, output_dir, include_nir_band
        )


def sample_and_save_patch_tiff(
    tile_path: Path, patch_size: int, output_dir: Path, include_nir_band: bool = True
) -> tuple[Path | None, dict | None]:
    """
    Sample a random patch from a NAIP tile and save it as a georeferenced GeoTIFF.

    Parameters
    ----------
    tile_path: Path
        Path to the NAIP tile
    patch_size: int
        Size of the patch to sample in pixels
    output_dir: Path
        Directory to save the sampled patch
    include_nir_band: bool, optional
        Whether to include the NIR band in the generated patches, by default True

    Returns
    -------
    tuple[Path | None, dict | None]
        Path to the saved image and bounding box info if successful, None values otherwise
    """
    try:
        # Validate tile before sampling
        if not validate_tiff(tile_path):
            print(f"Skipping invalid tile: {tile_path}")
            return None, None

        with rasterio.open(tile_path) as src:
            height, width = src.height, src.width
            band_count = src.count

            if height < patch_size or width < patch_size:
                return None, None

            # Sample from valid area (not too close to the edge)
            row = random.randint(0, height - patch_size)
            col = random.randint(0, width - patch_size)

            window = Window(col, row, patch_size, patch_size)

            # Get the metadata needed for the georeferenced output
            # This is the critical step for correct bounding box calculation
            out_transform = src.window_transform(window)

            # Calculate the bounds directly from the source raster and window
            window_bounds = rasterio.windows.bounds(window, src.transform)

            # Read the data from the window - handle band selection
            if band_count >= 4 and not include_nir_band:
                # NAIP imagery typically has 4 bands (R, G, B, NIR)
                # If we don't want NIR, read only the first 3 bands
                patch = src.read([1, 2, 3], window=window)
                out_count = 3
            else:
                # Read all bands
                patch = src.read(window=window)
                out_count = band_count

            # Generate unique filename (use .tif extension for GeoTIFF)
            filename = f"{uuid.uuid4()}.tif"
            filepath = output_dir / filename

            # Create a new profile (metadata) for the output file
            out_profile = src.profile.copy()
            out_profile.update(
                {
                    "height": patch_size,
                    "width": patch_size,
                    "transform": out_transform,
                    "count": out_count,  # Update band count if we're excluding NIR
                }
            )

            # Write the GeoTIFF with rasterio
            with rasterio.open(filepath, "w", **out_profile) as dst:
                dst.write(patch)

            # Create the bounding box info
            bbox_info = {
                "file_name": filename,
                "tile_id": tile_path.stem,
                "bounds": window_bounds,  # (left, bottom, right, top)
                "crs": src.crs.to_string(),
            }

            return filepath, bbox_info

    except (RasterioIOError, ValueError, KeyError, Exception) as e:
        print(f"Error sampling from {tile_path}: {str(e)}")
        return None, None


def sample_and_save_patch_png(
    tile_path: Path, patch_size: int, output_dir: Path, include_nir_band: bool = True
) -> tuple[Path | None, dict | None]:
    """
    Sample a random patch from a NAIP tile and save it as an image.

    Parameters
    ----------
    tile_path: Path
        Path to the NAIP tile
    patch_size: int
        Size of the patch to sample in pixels
    output_dir: Path
        Directory to save the sampled patch
    include_nir_band: bool, optional
        Whether to include the NIR band in the generated patches, by default True

    Returns
    -------
    tuple[Path | None, dict | None]
        Path to the saved image and bounding box info if successful, None values otherwise
    """
    try:
        # Validate tile before sampling
        if not validate_tiff(tile_path):
            print(f"Skipping invalid tile: {tile_path}")
            return None, None

        with rasterio.open(tile_path) as src:
            height, width = src.height, src.width
            band_count = src.count

            if height < patch_size or width < patch_size:
                return None, None

            # Sample from valid area (not too close to the edge)
            row = random.randint(0, height - patch_size)
            col = random.randint(0, width - patch_size)

            window = Window(col, row, patch_size, patch_size)

            # Get the window transform
            out_transform = src.window_transform(window)

            # Calculate the bounds directly from the source raster and window
            window_bounds = rasterio.windows.bounds(window, src.transform)

            # Handle band selection
            if band_count >= 4:
                if include_nir_band:
                    # Read all 4 bands (R, G, B, NIR)
                    patch = src.read(window=window)
                else:
                    # Read only RGB bands
                    patch = src.read([1, 2, 3], window=window)
            else:
                # If there are fewer than 4 bands, read all available
                patch = src.read(window=window)

            # Transpose for PIL image format (bands last)
            patch = np.transpose(patch, (1, 2, 0))

        # Generate unique filename
        filename = f"{uuid.uuid4()}.png"
        filepath = output_dir / filename

        # Convert to appropriate PIL format based on number of bands
        if patch.shape[2] == 4 and include_nir_band:  # RGBA
            # For PNG with NIR, we use the NIR band as the alpha channel
            image = Image.fromarray(patch, "RGBA")
        elif patch.shape[2] == 3 or (
            patch.shape[2] == 4 and not include_nir_band
        ):  # RGB
            image = Image.fromarray(patch[:, :, 0:3], "RGB")
        elif patch.shape[2] == 1:  # Grayscale
            image = Image.fromarray(patch.squeeze(), "L")
        else:
            return None, None

        image.save(filepath)

        # Create the bounding box info
        bbox_info = {
            "file_name": filename,
            "tile_id": tile_path.stem,
            "bounds": window_bounds,  # (left, bottom, right, top)
            "crs": src.crs.to_string(),
        }

        return filepath, bbox_info

    except (RasterioIOError, ValueError, KeyError, Exception) as e:
        print(f"Error sampling from {tile_path}: {str(e)}")
        return None, None


if __name__ == "__main__":
    pass
    # gdf = gpd.read_file(DATA_PATH / "coconino/boundary.geojson")
    # gdf = gpd.GeoDataFrame.from_features(
    #     {
    #         "type": "FeatureCollection",
    #         "features": [
    #             {
    #                 "type": "Feature",
    #                 "properties": {},
    #                 "geometry": {
    #                     "coordinates": [-111.465107, 34.849059],
    #                     "type": "Point",
    #                 },
    #             }
    #         ],
    #     }
    # )

    # tile_set = search(coconino_gdf)
    # tile_set = tile_set[tile_set["gsd"] == 0.3]
    # tile_set.to_file(
    #     DATA_PATH / "coconino-single/naip/coconino_tiles.geojson", driver="GeoJSON"
    # )
    # download(tile_set, output_dir=DATA_PATH / "coconino-single/naip/", max_workers=14)
    #
    # samples = sample_patches_from_naip_tile(
    #     DATA_PATH / "coconino-single" / "naip",
    #     n_samples=512,
    #     dataset_name="coconino-single",
    #     mode="train",
    #     max_workers=14,
    # )
    #
    # samples = sample_patches_from_naip_tile(
    #     DATA_PATH / "coconino-single" / "naip",
    #     n_samples=64,
    #     dataset_name="coconino-single",
    #     mode="test",
    #     max_workers=14,
    # )
