"""
Data Download and Preprocessing for DeepOSWSRM

This script handles:
1. Downloading Sentinel-1 and Sentinel-2 data from Google Earth Engine
2. Using Landsat-8/9 as a free alternative to PlanetScope for reference data
3. Preprocessing and cloud simulation
"""

import ee
import numpy as np
import rasterio
from rasterio.transform import from_bounds
import os
from datetime import datetime, timedelta
import geemap
from tqdm import tqdm
import json


class SentinelDataDownloader:
    """Download Sentinel-1 and Sentinel-2 data from Google Earth Engine"""
    
    def __init__(self, project_id='remote-sensing-469118'):
        """
        Initialize GEE
        
        Args:
            project_id: Your Google Cloud project ID (required for GEE)
        """
        try:
            if project_id:
                ee.Initialize(project=project_id)
            else:
                ee.Initialize()
            print("Google Earth Engine initialized successfully!")
        except Exception as e:
            print(f"Error initializing GEE: {e}")
            print("Please run 'earthengine authenticate' in terminal first")
            raise
    
    def get_sentinel1_image(self, roi, start_date, end_date, orbit='DESCENDING'):
        """
        Get Sentinel-1 SAR image
        
        Args:
            roi: Region of interest as ee.Geometry
            start_date: Start date string 'YYYY-MM-DD'
            end_date: End date string 'YYYY-MM-DD'
            orbit: Orbit direction 'ASCENDING' or 'DESCENDING'
        
        Returns:
            ee.Image with VV and VH bands
        """
        s1_collection = (ee.ImageCollection('COPERNICUS/S1_GRD')
                        .filterBounds(roi)
                        .filterDate(start_date, end_date)
                        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
                        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
                        .filter(ee.Filter.eq('orbitProperties_pass', orbit))
                        .filter(ee.Filter.eq('instrumentMode', 'IW')))
        
        # Get median composite to reduce speckle
        s1_image = s1_collection.median().select(['VV', 'VH'])
        
        return s1_image
    
    def get_sentinel2_image(self, roi, start_date, end_date, cloud_cover=20):
        """
        Get Sentinel-2 optical image
        
        Args:
            roi: Region of interest as ee.Geometry
            start_date: Start date string 'YYYY-MM-DD'
            end_date: End date string 'YYYY-MM-DD'
            cloud_cover: Maximum cloud cover percentage
        
        Returns:
            ee.Image with B, G, R, NIR bands at 10m resolution
        """
        s2_collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                        .filterBounds(roi)
                        .filterDate(start_date, end_date)
                        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloud_cover)))
        
        # Get median composite and select 10m bands
        s2_image = s2_collection.median().select(['B2', 'B3', 'B4', 'B8'])
        
        return s2_image
    
    def get_landsat_reference(self, roi, start_date, end_date, cloud_cover=20):
        """
        Get Landsat-8/9 image as high-resolution reference (30m, free alternative to PlanetScope)
        
        For better resolution, we can also use:
        - Sentinel-2 at 10m (same as input but can use different dates)
        - Landsat pan-sharpened to 15m
        
        Args:
            roi: Region of interest
            start_date: Start date
            end_date: End date
            cloud_cover: Maximum cloud cover
        
        Returns:
            ee.Image with optical bands
        """
        # Try Landsat-9 first (launched 2021), then Landsat-8
        l9_collection = (ee.ImageCollection('LANDSAT/LC09/C02/T1_L2')
                        .filterBounds(roi)
                        .filterDate(start_date, end_date)
                        .filter(ee.Filter.lt('CLOUD_COVER', cloud_cover)))
        
        l8_collection = (ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
                        .filterBounds(roi)
                        .filterDate(start_date, end_date)
                        .filter(ee.Filter.lt('CLOUD_COVER', cloud_cover)))
        
        # Merge collections
        landsat = l9_collection.merge(l8_collection)
        
        if landsat.size().getInfo() == 0:
            print("Warning: No Landsat images found, trying Sentinel-2 as reference")
            return self.get_sentinel2_image(roi, start_date, end_date, cloud_cover)
        
        # Get median composite
        landsat_image = landsat.median()
        
        # Scale and select bands (SR_B2=Blue, SR_B3=Green, SR_B4=Red, SR_B5=NIR)
        landsat_image = landsat_image.select(['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5'])
        
        # Apply scaling factors
        def apply_scale_factors(image):
            optical_bands = image.select(['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5']).multiply(0.0000275).add(-0.2)
            return optical_bands
        
        landsat_image = apply_scale_factors(landsat_image)
        
        return landsat_image
    
    def download_image(self, image, roi, filename, scale=10, crs='EPSG:4326'):
        """
        Download an ee.Image to local file
        
        Args:
            image: ee.Image to download
            roi: Region of interest
            filename: Output filename
            scale: Resolution in meters
            crs: Coordinate reference system
        """
        # Create export task
        geemap.ee_export_image(
            image,
            filename=filename,
            scale=scale,
            region=roi,
            file_per_band=False,
            crs=crs
        )
        print(f"Downloaded: {filename}")
    
    def prepare_training_data(self, roi, start_date, end_date, output_dir, 
                             site_name, scale_factor=4):
        """
        Prepare a complete training sample
        
        Args:
            roi: Region of interest
            start_date: Start date
            end_date: End date
            output_dir: Output directory
            site_name: Name for this site
            scale_factor: Super-resolution scale factor
        
        Returns:
            Dictionary with file paths
        """
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\nPreparing data for site: {site_name}")
        print(f"Date range: {start_date} to {end_date}")
        
        # Download Sentinel-1 (10m)
        print("Downloading Sentinel-1...")
        s1_image = self.get_sentinel1_image(roi, start_date, end_date)
        s1_path = os.path.join(output_dir, f"{site_name}_S1.tif")
        self.download_image(s1_image, roi, s1_path, scale=10)

        # Download Sentinel-2 (10m)
        print("Downloading Sentinel-2...")
        s2_image = self.get_sentinel2_image(roi, start_date, end_date)
        s2_path = os.path.join(output_dir, f"{site_name}_S2.tif")
        self.download_image(s2_image, roi, s2_path, scale=10)

        # Download reference data at SAME 10m resolution
        print("Downloading reference data (Landsat/Sentinel-2)...")
        ref_image = self.get_landsat_reference(roi, start_date, end_date)
        ref_path = os.path.join(output_dir, f"{site_name}_reference.tif")
        self.download_image(ref_image, roi, ref_path, scale=10)  # SAME as Sentinel
        
        return {
            'sentinel1': s1_path,
            'sentinel2': s2_path,
            'reference': ref_path,
            'site_name': site_name,
            'roi': roi.getInfo(),
            'dates': {'start': start_date, 'end': end_date}
        }


class WaterIndexCalculator:
    """Calculate water indices to create reference water maps"""
    
    @staticmethod
    def ndwi(green, nir):
        """
        Normalized Difference Water Index
        NDWI = (Green - NIR) / (Green + NIR)
        """
        return (green - nir) / (green + nir + 1e-8)
    
    @staticmethod
    def mndwi(green, swir):
        """
        Modified NDWI
        MNDWI = (Green - SWIR) / (Green + SWIR)
        """
        return (green - swir) / (green + swir + 1e-8)
    
    @staticmethod
    def awei_nsh(blue, green, nir, swir1, swir2):
        """
        Automated Water Extraction Index (no shadow)
        AWEInsh = 4 * (Green - SWIR1) - (0.25 * NIR + 2.75 * SWIR2)
        """
        return 4 * (green - swir1) - (0.25 * nir + 2.75 * swir2)
    
    @staticmethod
    def create_water_mask(image_path, method='ndwi', threshold=0):
        """
        Create binary water mask from multispectral image
        
        Args:
            image_path: Path to image file
            method: Water index method ('ndwi', 'mndwi', 'awei')
            threshold: Threshold value
        
        Returns:
            Binary water mask as numpy array
        """
        with rasterio.open(image_path) as src:
            # Assuming bands: Blue, Green, Red, NIR
            bands = src.read()
            
            if method == 'ndwi' and bands.shape[0] >= 4:
                green = bands[1].astype(float)
                nir = bands[3].astype(float)
                index = WaterIndexCalculator.ndwi(green, nir)
            else:
                # Default: use NIR threshold
                nir = bands[-1].astype(float)
                index = -nir  # Water has low NIR
            
            # Create binary mask
            water_mask = (index > threshold).astype(np.uint8)
            
            return water_mask, src.transform, src.crs


class CloudSimulator:
    """Simulate cloud cover for training data augmentation"""
    
    @staticmethod
    def generate_cloud_mask(shape, coverage_rate=0.5, num_clouds=5):
        """
        Generate realistic cloud mask
        
        Args:
            shape: (height, width) of the image
            coverage_rate: Target cloud coverage (0-1)
            num_clouds: Number of cloud patches
        
        Returns:
            Binary cloud mask (1 = cloud, 0 = clear)
        """
        height, width = shape
        mask = np.zeros((height, width), dtype=np.uint8)
        
        for _ in range(num_clouds):
            # Random cloud center
            center_y = np.random.randint(0, height)
            center_x = np.random.randint(0, width)
            
            # Random cloud size
            size_y = np.random.randint(height // 10, height // 3)
            size_x = np.random.randint(width // 10, width // 3)
            
            # Create elliptical cloud
            y_grid, x_grid = np.ogrid[:height, :width]
            cloud = ((y_grid - center_y) ** 2 / size_y ** 2 + 
                    (x_grid - center_x) ** 2 / size_x ** 2) <= 1
            
            mask = np.logical_or(mask, cloud).astype(np.uint8)
            
            # Check coverage
            current_coverage = mask.sum() / mask.size
            if current_coverage >= coverage_rate:
                break
        
        return mask
    
    @staticmethod
    def apply_cloud_mask(image, cloud_mask, fill_value=0):
        """
        Apply cloud mask to image
        
        Args:
            image: Image array [C, H, W] or [H, W]
            cloud_mask: Binary mask [H, W]
            fill_value: Value to fill clouded areas
        
        Returns:
            Masked image
        """
        if image.ndim == 3:
            cloud_mask_3d = np.repeat(cloud_mask[np.newaxis, :, :], image.shape[0], axis=0)
            masked_image = image.copy()
            masked_image[cloud_mask_3d == 1] = fill_value
        else:
            masked_image = image.copy()
            masked_image[cloud_mask == 1] = fill_value
        
        return masked_image


def prepare_sample_training_sites():
    """
    Prepare sample training sites
    
    Returns list of dictionaries with ROI and date information
    """
    # Example sites - you can modify these
    sites = [
        {
            'name': 'Rabindra_Sarobar',
            'roi': ee.Geometry.Rectangle([88.3445, 22.5085, 88.3555, 22.5185]),
            'start_date': '2023-06-01',
            'end_date': '2024-08-31'
        },
        {
            'name': 'East_Kolkata_Wetlands',
            'roi': ee.Geometry.Rectangle([88.4400, 22.5150, 88.4500, 22.5250]),
            'start_date': '2023-06-01',
            'end_date': '2024-08-31'
        },
        {
            # bad quality data - keep for testing
            'name': 'Nalban_Lake',
            'roi': ee.Geometry.Rectangle([88.4160, 22.5740, 88.4280, 22.5840]),
            'start_date': '2023-06-01',
            'end_date': '2024-08-31'
        },
        {
            'name': 'Santragachi_Jheel',
            'roi': ee.Geometry.Rectangle([88.2850, 22.5700, 88.2950, 22.5800]),
            'start_date': '2023-06-01',
            'end_date': '2024-08-31'
        },
        {
            'name': 'Subhas_Sarobar',
            'roi': ee.Geometry.Rectangle([88.3900, 22.5570, 88.4000, 22.5670]),
            'start_date': '2023-06-01',
            'end_date': '2024-08-31'
        },
        
        # West Bengal - Other regions
        {
            'name': 'Digha_Beach',
            'roi': ee.Geometry.Rectangle([87.5200, 21.6200, 87.5400, 21.6400]),
            'start_date': '2023-06-01',
            'end_date': '2024-08-31'
        },
        {
            'name': 'Sundarbans_Creek',
            'roi': ee.Geometry.Rectangle([88.9500, 22.0500, 88.9700, 22.0700]),
            'start_date': '2023-06-01',
            'end_date': '2024-08-31'
        },
        
        # Major Indian Water Bodies
        {
            'name': 'Dal_Lake_Kashmir',
            'roi': ee.Geometry.Rectangle([74.8600, 34.0800, 74.8900, 34.1100]),
            'start_date': '2023-06-01',
            'end_date': '2024-08-31'
        },
        {
            'name': 'Chilika_Lake_Odisha',
            'roi': ee.Geometry.Rectangle([85.3200, 19.6800, 85.3600, 19.7200]),
            'start_date': '2023-06-01',
            'end_date': '2024-08-31'
        },
        {
            'name': 'Vembanad_Lake_Kerala',
            'roi': ee.Geometry.Rectangle([76.3500, 9.5800, 76.3900, 9.6200]),
            'start_date': '2023-06-01',
            'end_date': '2024-08-31'
        },
        {
            'name': 'Loktak_Lake_Manipur',
            'roi': ee.Geometry.Rectangle([93.7600, 24.5200, 93.8000, 24.5600]),
            'start_date': '2023-06-01',
            'end_date': '2024-08-31'
        },
        {
            'name': 'Pulicat_Lake_TN',
            'roi': ee.Geometry.Rectangle([80.3000, 13.4200, 80.3400, 13.4600]),
            'start_date': '2023-06-01',
            'end_date': '2024-08-31'
        },
        
        # Reservoirs
        {
            'name': 'Hirakud_Reservoir_Odisha',
            'roi': ee.Geometry.Rectangle([83.8500, 21.5200, 83.9000, 21.5700]),
            'start_date': '2023-06-01',
            'end_date': '2024-08-31'
        },
        {
            'name': 'Nagarjuna_Sagar_AP',
            'roi': ee.Geometry.Rectangle([79.3000, 16.5500, 79.3500, 16.6000]),
            'start_date': '2023-06-01',
            'end_date': '2024-08-31'
        },
        {
            'name': 'Gobind_Sagar_HP',
            'roi': ee.Geometry.Rectangle([76.4200, 31.4000, 76.4700, 31.4500]),
            'start_date': '2023-06-01',
            'end_date': '2024-08-31'
        },
        
        # Rivers
        {
            'name': 'Hooghly_River_Kolkata',
            'roi': ee.Geometry.Rectangle([88.3200, 22.5500, 88.3500, 22.5800]),
            'start_date': '2023-06-01',
            'end_date': '2024-08-31'
        },
        {
            'name': 'Brahmaputra_Assam',
            'roi': ee.Geometry.Rectangle([91.7000, 26.1500, 91.7500, 26.2000]),
            'start_date': '2023-06-01',
            'end_date': '2024-08-31'
        },
        {
            # bad quality data - keep for testing
            'name': 'Narmada_Gujarat',
            'roi': ee.Geometry.Rectangle([73.0000, 21.8000, 73.0500, 21.8500]),
            'start_date': '2023-06-01',
            'end_date': '2024-08-31'
        },
        
        # Coastal areas
        {
            'name': 'Mumbai_Harbor',
            'roi': ee.Geometry.Rectangle([72.8300, 18.9000, 72.8700, 18.9400]),
            'start_date': '2023-06-01',
            'end_date': '2024-08-31'
        },
        {
            'name': 'Chennai_Marina',
            'roi': ee.Geometry.Rectangle([80.2700, 13.0400, 80.3100, 13.0800]),
            'start_date': '2023-06-01',
            'end_date': '2024-08-31'
        }

    ]
    
    return sites


def main():
    """Main function to download and prepare data"""
    
    # Initialize downloader
    # If you get an error, you need to authenticate first:
    # Run in terminal: earthengine authenticate
    print("Initializing Google Earth Engine...")
    try:
        downloader = SentinelDataDownloader()
    except:
        print("\nPlease authenticate with Google Earth Engine:")
        print("1. Run: pip install earthengine-api")
        print("2. Run: earthengine authenticate")
        print("3. Follow the authentication process")
        return
    
    # Output directory
    output_base_dir = './deeposwsrm_data'
    os.makedirs(output_base_dir, exist_ok=True)
    
    # Get sample sites
    sites = prepare_sample_training_sites()
    
    # Download data for each site
    all_samples = []
    for site in sites:
        try:
            sample_data = downloader.prepare_training_data(
                roi=site['roi'],
                start_date=site['start_date'],
                end_date=site['end_date'],
                output_dir=os.path.join(output_base_dir, site['name']),
                site_name=site['name'],
                scale_factor=4
            )
            all_samples.append(sample_data)
        except Exception as e:
            print(f"Error processing site {site['name']}: {e}")
            continue
    
    # Save metadata
    metadata_path = os.path.join(output_base_dir, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(all_samples, f, indent=2)
    
    print(f"\nData preparation complete!")
    print(f"Downloaded {len(all_samples)} training samples")
    print(f"Metadata saved to: {metadata_path}")
    
    # Create water masks from reference images
    print("\nCreating water masks from reference images...")
    for sample in all_samples:
        try:
            ref_path = sample['reference']
            water_mask, transform, crs = WaterIndexCalculator.create_water_mask(
                ref_path, method='ndwi', threshold=0
            )
            
            # IMPORTANT: Upsample water mask to fine resolution
            # For scale_factor=4, if input is 64×64, mask should be 256×256
            import cv2
            scale_factor = 4  # Match your training scale factor
            h_current, w_current = water_mask.shape
            h_fine = h_current * scale_factor
            w_fine = w_current * scale_factor
            
            water_mask_fine = cv2.resize(
                water_mask.astype(np.uint8), 
                (w_fine, h_fine), 
                interpolation=cv2.INTER_NEAREST  # Use nearest neighbor for binary masks
            )
            
            print(f"Upsampled water mask from {h_current}×{w_current} to {h_fine}×{w_fine}")
            
            # Save water mask at fine resolution
            mask_path = ref_path.replace('_reference.tif', '_water_mask.tif')
            with rasterio.open(ref_path) as src:
                profile = src.profile.copy()
                profile.update(
                    count=1, 
                    dtype=rasterio.uint8,
                    height=h_fine,
                    width=w_fine,
                    transform=src.transform * src.transform.scale(
                        src.width / w_fine,
                        src.height / h_fine
                    )
                )
                
                with rasterio.open(mask_path, 'w', **profile) as dst:
                    dst.write(water_mask_fine, 1)
            
            print(f"Created water mask: {mask_path}")
            sample['water_mask'] = mask_path
        except Exception as e:
            print(f"Error creating water mask for {sample['site_name']}: {e}")
    
    # Update metadata with water masks
    with open(metadata_path, 'w') as f:
        json.dump(all_samples, f, indent=2)
    
    print("\nAll done! Your data is ready for training.")
    print(f"Check the '{output_base_dir}' directory for your data.")


if __name__ == "__main__":
    main()
