import abtem
import argparse
import ase
import numpy as np
import os
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

def make_tomo_projections(filepath, angles):
    atoms_init = ase.io.read(filepath)
    potential_init = abtem.potentials.Potential(
        atoms_init, 
        sampling=0.25,
        slice_thickness=0.25,
    )

    projection_shape = potential_init.project().shape

    projections = np.empty((len(angles), projection_shape[1], projection_shape[0]))
    for i, angle in tqdm(enumerate(angles)):
        atoms_init_rotated = atoms_init.copy()
        atoms_init_rotated.rotate(angle, 'x', center='COU')
        potential_init_rotated = abtem.potentials.Potential(
            atoms_init_rotated,
            sampling=0.25,
            slice_thickness=0.25,
        )

        # Rotate to get the tube horizontal
        # For others, orientation doesn't matter; use the same convention        
        projections[i] = np.rot90(potential_init_rotated.project().array)

        # Gaussian blur to simulate limited microscope resolution
        sigma_blur = 2.0 # 0.5 Angstroms
        projections[i] = gaussian_filter(projections[i], sigma=sigma_blur)
        
        # Add Gaussian noise to simulate detector noise
        noise_level = 50
        noise = np.random.normal(0, noise_level, projections[i].shape)
        projections[i] += noise

    return projections

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--filenames', nargs='+', type=str,
                       default=['aPd_NP.xyz',
                               'Au_NP.xyz', 
                               'FePt_NP.xyz',
                               'HEA_NP.xyz',
                               'ZrTe_model_full.vasp'],
                       help='List of structure files to process')
    args = parser.parse_args()
    filenames = args.filenames

    angles = np.linspace(-180, 180, 361)

    for filename in tqdm(filenames):
        filepath = os.path.join('structures', filename)
        projections = make_tomo_projections(filepath, angles)
        filestr = filename.split('.')[0]

        save_dir = 'projections'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        np.save(f'{save_dir}/sim_{filestr}_projections.npy', projections)
