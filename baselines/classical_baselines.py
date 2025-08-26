import argparse
import os
import numpy as np
from scipy.ndimage import rotate
from skimage.transform import iradon, iradon_sart
import torch

def reconstruct_3d(projections, angles, recon_type='fbp'):
    """Performs 3D reconstruction using either filtered backprojection or SART.
    
    Args:
        projections: numpy array of shape (num_angles, height, width) containing 2D projections
        angles: numpy array of projection angles in degrees
        recon_type: string specifying reconstruction type ('fbp' or 'sart') 
        
    Returns:
        3D numpy array containing the reconstructed volume
    """
    if isinstance(projections, torch.Tensor):
        projections = projections.detach().cpu().numpy()
    if isinstance(angles, torch.Tensor):
        angles = angles.detach().cpu().numpy()
        
    num_angles, width, height = projections.shape
        
    # Initialize volume
    volume = np.zeros((width, width, height))
    
    # Reconstruct each vertical slice
    for i in range(height):
        sinogram = projections[:, :, i].T

        if recon_type == 'fbp':
            recon = iradon(sinogram, 
                          theta=angles,
                          filter_name='ramp',
                          interpolation='linear', 
                          circle=False,
                          output_size=width)
        elif recon_type == 'sart':
            recon = iradon_sart(sinogram, theta=angles)
        else:
            raise ValueError("recon_type must be either 'fbp' or 'sart'")
            
        volume[:, :, i] = recon
        
    return volume

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--proj_path', type=str, default='../data/projections/',
                       help='Path to projection files')
    args = parser.parse_args()
    proj_path = os.path.join(args.proj_path)
    proj_files = [f for f in os.listdir(proj_path) if f.endswith('_proj.npy')]
    
    # Create output directories if they don't exist
    fbp_dir = 'fbp_recons'
    sart_dir = 'sart_recons'
    if not os.path.exists(fbp_dir):
        os.makedirs(fbp_dir)
    if not os.path.exists(sart_dir):
        os.makedirs(sart_dir)
        
    for proj_file in proj_files:
        if not(proj_file.replace('_proj.npy', '_recon.npy') in os.listdir(out_dir) or 'ZrTe' in proj_file):
            # Get corresponding angles file
            angles_file = proj_file.replace('_proj.npy', '_angles.npy')

            # Load projections and angles
            projections = np.load(os.path.join(proj_path, proj_file))
            angles = np.load(os.path.join(proj_path, angles_file))
            
            # Run SART reconstruction
            recon = reconstruct_3d(projections, angles, recon_type='sart')
            
            # Save reconstruction
            out_file = os.path.join(out_dir, proj_file.replace('_proj.npy', '_recon.npy'))
            np.save(out_file, recon)