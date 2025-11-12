"""
Inference Script for DeepOSWSRM

Apply trained model to generate super-resolution water maps
"""

import torch
import numpy as np
import rasterio
from rasterio.transform import from_bounds
from pathlib import Path
import argparse
import json
from tqdm import tqdm
import matplotlib.pyplot as plt

from deeposwsrm_model import DeepOSWSRM


class WaterMapper:
    """Inference class for water body mapping"""
    
    def __init__(self, checkpoint_path, device=None):
        """
        Initialize mapper
        
        Args:
            checkpoint_path: Path to model checkpoint
            device: Device to use (cuda/cpu)
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.config = checkpoint['config']
        
        # Create model
        self.model = DeepOSWSRM(
            sentinel1_channels=self.config['sentinel1_channels'],
            sentinel2_channels=self.config['sentinel2_channels'],
            scale_factor=self.config['scale_factor'],
            base_channels=self.config['base_channels']
        ).to(self.device)
        
        # Load weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"Loaded model from epoch {checkpoint['epoch']}")
        print(f"Scale factor: {self.config['scale_factor']}")
    
    def _normalize_sentinel1(self, s1_data):
        """Normalize Sentinel-1 data"""
        s1_data = 10 * np.log10(np.abs(s1_data) + 1e-8)
        s1_data = np.clip(s1_data, -25, 5)
        s1_data = (s1_data + 25) / 30
        return s1_data
    
    def _normalize_sentinel2(self, s2_data):
        """Normalize Sentinel-2 data"""
        return np.clip(s2_data, 0, 1)
    
    def predict_from_files(self, sentinel1_path, sentinel2_path, output_path=None):
        """
        Generate water map from Sentinel-1 and Sentinel-2 files
        
        Args:
            sentinel1_path: Path to Sentinel-1 image
            sentinel2_path: Path to Sentinel-2 image
            output_path: Path to save output (optional)
        
        Returns:
            water_fraction: Coarse resolution water fraction
            water_map: Fine resolution water map
        """
        print(f"Processing:")
        print(f"  Sentinel-1: {sentinel1_path}")
        print(f"  Sentinel-2: {sentinel2_path}")
        
        # Load images
        with rasterio.open(sentinel1_path) as src:
            s1_data = src.read().astype(np.float32)
            s1_profile = src.profile
        
        with rasterio.open(sentinel2_path) as src:
            s2_data = src.read().astype(np.float32)
            s2_profile = src.profile
        
        # Check shapes match
        if s1_data.shape[1:] != s2_data.shape[1:]:
            raise ValueError("Sentinel-1 and Sentinel-2 images must have the same spatial dimensions")
        
        # Normalize
        s1_data = self._normalize_sentinel1(s1_data)
        s2_data = self._normalize_sentinel2(s2_data)
        
        # Process in patches to handle large images
        patch_size = 256  # Process in 256x256 patches
        overlap = 32  # Overlap to avoid edge artifacts
        
        h, w = s1_data.shape[1:]
        scale_factor = self.config['scale_factor']
        
        # Initialize output arrays
        water_fraction = np.zeros((1, h, w), dtype=np.float32)
        water_map = np.zeros((1, h * scale_factor, w * scale_factor), dtype=np.float32)
        count_map = np.zeros((1, h, w), dtype=np.float32)
        
        print(f"Processing {h}x{w} image in patches...")
        
        # Process patches with overlap
        with torch.no_grad():
            for i in tqdm(range(0, h, patch_size - overlap)):
                for j in range(0, w, patch_size - overlap):
                    # Extract patch
                    i_end = min(i + patch_size, h)
                    j_end = min(j + patch_size, w)
                    
                    s1_patch = s1_data[:, i:i_end, j:j_end]
                    s2_patch = s2_data[:, i:i_end, j:j_end]
                    
                    # Pad if needed
                    pad_h = patch_size - s1_patch.shape[1]
                    pad_w = patch_size - s1_patch.shape[2]
                    
                    if pad_h > 0 or pad_w > 0:
                        s1_patch = np.pad(s1_patch, ((0, 0), (0, pad_h), (0, pad_w)), mode='reflect')
                        s2_patch = np.pad(s2_patch, ((0, 0), (0, pad_h), (0, pad_w)), mode='reflect')
                    
                    # Convert to tensor
                    s1_tensor = torch.from_numpy(s1_patch[np.newaxis, :, :, :]).float().to(self.device)
                    s2_tensor = torch.from_numpy(s2_patch[np.newaxis, :, :, :]).float().to(self.device)
                    
                    # Predict
                    pred_fraction, pred_map_probs, pred_map = self.model.predict(s1_tensor, s2_tensor)
                    
                    # Convert to numpy
                    pred_fraction = pred_fraction.cpu().numpy()[0]
                    pred_map_probs = pred_map_probs.cpu().numpy()[0]
                    
                    # Remove padding
                    pred_fraction = pred_fraction[:, :i_end-i, :j_end-j]
                    pred_map_probs = pred_map_probs[:, :(i_end-i)*scale_factor, :(j_end-j)*scale_factor]
                    
                    # Add to output with overlap handling
                    water_fraction[:, i:i_end, j:j_end] += pred_fraction
                    water_map[:, i*scale_factor:i_end*scale_factor, 
                             j*scale_factor:j_end*scale_factor] += pred_map_probs
                    count_map[:, i:i_end, j:j_end] += 1
        
        # Average overlapping regions
        water_fraction = water_fraction / np.maximum(count_map, 1)
        
        # For fine map, upsample count map and average
        count_map_fine = np.repeat(np.repeat(count_map, scale_factor, axis=1), 
                                   scale_factor, axis=2)
        water_map = water_map / np.maximum(count_map_fine, 1)
        
        # Create binary map
        water_map_binary = (water_map > 0.5).astype(np.uint8)
        
        # Save if output path provided
        if output_path:
            self._save_results(
                water_fraction, water_map, water_map_binary,
                output_path, s1_profile, scale_factor
            )
        
        return water_fraction[0], water_map[0], water_map_binary[0]
    
    def _save_results(self, water_fraction, water_map_prob, water_map_binary,
                     output_path, profile, scale_factor):
        """Save prediction results"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Update profile for fraction
        frac_profile = profile.copy()
        frac_profile.update(count=1, dtype=rasterio.float32)
        
        # Save fraction
        frac_path = output_path.parent / f"{output_path.stem}_fraction.tif"
        with rasterio.open(frac_path, 'w', **frac_profile) as dst:
            dst.write(water_fraction)
        print(f"Saved fraction map: {frac_path}")
        
        # Update profile for fine resolution map
        fine_profile = profile.copy()
        fine_profile.update(
            height=profile['height'] * scale_factor,
            width=profile['width'] * scale_factor,
            transform=profile['transform'] * profile['transform'].scale(
                1/scale_factor, 1/scale_factor
            ),
            count=1
        )
        
        # Save probability map
        fine_profile.update(dtype=rasterio.float32)
        prob_path = output_path.parent / f"{output_path.stem}_probability.tif"
        with rasterio.open(prob_path, 'w', **fine_profile) as dst:
            dst.write(water_map_prob)
        print(f"Saved probability map: {prob_path}")
        
        # Save binary map
        fine_profile.update(dtype=rasterio.uint8)
        binary_path = output_path
        with rasterio.open(binary_path, 'w', **fine_profile) as dst:
            dst.write(water_map_binary)
        print(f"Saved binary map: {binary_path}")
    
    def visualize_results(self, sentinel2_path, water_fraction, water_map, 
                         output_path=None):
        """
        Visualize prediction results
        
        Args:
            sentinel2_path: Path to Sentinel-2 image for RGB visualization
            water_fraction: Coarse resolution water fraction
            water_map: Fine resolution water map
            output_path: Path to save visualization
        """
        # Load Sentinel-2 for RGB
        with rasterio.open(sentinel2_path) as src:
            # Assuming bands are B, G, R, NIR
            s2_data = src.read([3, 2, 1])  # RGB
        
        # Normalize for display
        s2_rgb = np.transpose(s2_data, (1, 2, 0))
        s2_rgb = np.clip(s2_rgb * 3, 0, 1)  # Enhance brightness
        
        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Plot RGB
        axes[0].imshow(s2_rgb)
        axes[0].set_title('Sentinel-2 RGB')
        axes[0].axis('off')
        
        # Plot fraction
        im1 = axes[1].imshow(water_fraction, cmap='Blues', vmin=0, vmax=1)
        axes[1].set_title('Water Fraction (Coarse)')
        axes[1].axis('off')
        plt.colorbar(im1, ax=axes[1], fraction=0.046)
        
        # Plot water map
        axes[2].imshow(water_map, cmap='Blues')
        axes[2].set_title(f'Water Map (Fine, Scale {self.config["scale_factor"]}x)')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Saved visualization: {output_path}")
        else:
            plt.show()
        
        plt.close()


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Generate water maps with DeepOSWSRM')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--sentinel1', type=str, required=True,
                       help='Path to Sentinel-1 image')
    parser.add_argument('--sentinel2', type=str, required=True,
                       help='Path to Sentinel-2 image')
    parser.add_argument('--output', type=str, required=True,
                       help='Path to save output water map')
    parser.add_argument('--visualize', action='store_true',
                       help='Generate visualization')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda/cpu)')
    
    return parser.parse_args()


def main():
    """Main function"""
    args = parse_args()
    
    # Set device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device: {device}")
    
    # Create mapper
    mapper = WaterMapper(args.checkpoint, device=device)
    
    # Generate predictions
    water_fraction, water_map_prob, water_map_binary = mapper.predict_from_files(
        sentinel1_path=args.sentinel1,
        sentinel2_path=args.sentinel2,
        output_path=args.output
    )
    
    # Visualize if requested
    if args.visualize:
        vis_path = Path(args.output).parent / f"{Path(args.output).stem}_visualization.png"
        mapper.visualize_results(
            sentinel2_path=args.sentinel2,
            water_fraction=water_fraction,
            water_map=water_map_binary,
            output_path=vis_path
        )
    
    print("\nInference completed successfully!")


if __name__ == "__main__":
    main()
