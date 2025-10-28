import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import json
import yaml
import numpy as np
import torch
from matplotlib import pyplot as plt
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.models.splatfacto import *
from nerfstudio.cameras.cameras import Cameras, CameraType
from PIL.Image import open as popen
from pathlib import Path
import ase
import abtem
import plotly.graph_objs as go
import numpy as np
from PIL import Image
import io

def load_splatfacto_model(model_path, extent=None, ckpt=None):
    """
    Load a Splatfacto model from a checkpoint file.
    
    Parameters
    ----------
    model_path : str
        Path to the directory containing the model checkpoint and config files.
    extent : tuple of float, optional
        Bounding box extent as (x, y, z) dimensions. If None, uses default [-1, 1] cube.
    ckpt : int, optional
        Checkpoint step number. If None, uses the default checkpoint (step 9999).
        
    Returns
    -------
    SplatfactoModel
        Loaded and configured Splatfacto model ready for inference on CUDA device.
    """
    if ckpt is None:
        ckpt_path = os.path.join(model_path, 'nerfstudio_models', 'step-000009999.ckpt')
    else:
        ckpt_path = os.path.join(model_path, 'nerfstudio_models', f'step-0000{ckpt:04d}.ckpt')
    splatfacto_dict = torch.load(ckpt_path)['pipeline']

    splatfacto_dict_rename = {}
    for key in splatfacto_dict.keys():
        splatfacto_dict_rename[key[7:]] = splatfacto_dict[key]

    config_path = os.path.join(model_path, 'config.yml')
    config_dict = yaml.load(Path(config_path).read_text(), Loader=yaml.Loader)
    
    config = SplatfactoModelConfig()
    config.force_isotropic = config_dict.pipeline.model.force_isotropic
    config.cluster_gaussians = config_dict.pipeline.model.cluster_gaussians
    if extent is None:
        splatfacto = SplatfactoModel(config,SceneBox(aabb=torch.tensor([[-1,-1,-1], [1,1,1]])),1)
    else:
        splatfacto = SplatfactoModel(config,SceneBox(aabb=torch.tensor([[-extent[0]/2,-extent[1]/2,-extent[1]/2], [extent[0]/2,extent[1]/2,extent[1]/2]])),1)
    splatfacto.load_state_dict(splatfacto_dict_rename)
    splatfacto.training = False
    splatfacto.to('cuda:0')

    return splatfacto


def get_recon(model, ns_data_path, n=90):
    """
    Generate projections and 3D reconstruction from a Splatfacto model.
    
    Parameters
    ----------
    model : SplatfactoModel
        The loaded Splatfacto model to render from.
    ns_data_path : str
        Path to the NeRF Studio data directory containing transforms.json and images.
    n : int, default=90
        Frame index to use for camera parameters from transforms.json. Should correspond to a 0 degree projection.
        
    Returns
    -------
    tuple
        A tuple containing:
        - projections : torch.Tensor
            2D projection image from the specified camera viewpoint
        - recon : numpy.ndarray
            3D reconstruction array with shape (height, width, height)
            
    """
    camera_json = json.load(open(os.path.join(ns_data_path,'transforms.json')))

    camera_to_worlds = []
    view_i = np.asarray(camera_json['frames'][n]['transform_matrix'])
    trans_i = torch.Tensor(view_i)
    camera_to_worlds.append(trans_i)

    camera_to_worlds = torch.stack(camera_to_worlds,axis=0).to('cuda:0')

    camera = Cameras(camera_to_worlds = camera_to_worlds,
                        fx = float(camera_json['fl_x']),
                        fy = float(camera_json['fl_y']),
                        cx = float(camera_json['cx']),
                        cy = float(camera_json['cy']),
                        width = camera_json['w'],
                        height = camera_json['h'],
                        camera_type = CameraType.ORTHOPHOTO
                    )
    measurements_png = os.path.join(ns_data_path,'images','frame_00000.png')
    h, w, _ = np.asarray(popen(measurements_png)).shape

    recon = np.zeros((h,w,h))
    recon = np.transpose(recon,(0,1,2))
    recon = np.flip(recon,axis=0)
    for i, x in enumerate(np.linspace(0.5,1.5,h,endpoint=False)):
        # Image lies in [-0.5,0.5] along this axis
        # Camera is at 1 looking in the negative direction
        # TODO: Instead of rendering slices, compute a true volumetric render
        c = 0.5
        crop_outputs = model.get_outputs_for_camera(camera,near_plane=c+i/h,far_plane=c+(i+1)/h)
        recon[:,:,i] = crop_outputs['rgb'].cpu()[:,:,0]

    return recon


def load_data(model_path, atoms_path, ckpt=None, threshold=True, opacity_threshold=1, scale_threshold=-6, flip_yz=False, load_potential=False):
    """
    Load and process Splatfacto model and atomic structure data for visualization.
    
    Parameters
    ----------
    model_path : str
        Path to the directory containing the Splatfacto model checkpoint and config.
    atoms_path : str
        Path to the atomic structure file (e.g., .xyz, .cif, etc.).
    ckpt : int, optional
        Checkpoint step number for the model. If None, uses default checkpoint.
    threshold : bool, default=True
        Whether to apply opacity and scale thresholds to filter Gaussians.
    opacity_threshold : float, default=1
        Minimum opacity value for Gaussian filtering.
    scale_threshold : float, default=-6
        Minimum scale value (log scale) for Gaussian filtering.
    flip_yz : bool, default=False
        Whether to flip the y and z coordinates of the atomic positions.
    load_potential : bool, default=False
        Whether to load and return the electrostatic potential data.
        
    Returns
    -------
    list
        A list containing:
        - true_positions : numpy.ndarray
            Atomic positions after coordinate transformations
        - species : list
            Chemical symbols of the atomic species
        - gauss_means : numpy.ndarray
            Gaussian center positions from the model
        - opacities : torch.Tensor
            Opacity values of the filtered Gaussians
        - scales : torch.Tensor
            Scale values of the filtered Gaussians
        - extent : tuple
            Bounding box extent of the atomic structure
        - potential : numpy.ndarray, optional
            Electrostatic potential array (only if load_potential=True)
    """
    if(load_potential):
        true_positions, extent, species, potential = get_true_positions(atoms_path, return_species=True, return_potential=load_potential)
        potential = potential.build(pbar=False).array
    else:
        true_positions, extent, species = get_true_positions(atoms_path, return_species=True)

    model = load_splatfacto_model(model_path, extent=extent, ckpt=ckpt)

    gauss_means = model.means.detach().cpu().numpy()

    if(threshold):
        opacity_mask = model.opacities[:,0].detach().cpu().numpy()>opacity_threshold
        scale_mask = model.scales[:,0].detach().cpu().numpy()>scale_threshold
    else:
        opacity_mask = np.ones(model.opacities.shape[0], dtype=bool)
        scale_mask = np.ones(model.scales.shape[0], dtype=bool)

    # In NerfStudio coordinates, the particles lie between -0.5 and 0.5
    loc_mask = np.all((gauss_means > -0.5) & (gauss_means < 0.5), axis=1)
    gauss_means = gauss_means[opacity_mask & loc_mask & scale_mask]
    
    opacities = model.opacities[opacity_mask & loc_mask & scale_mask]
    scales = model.scales[opacity_mask & loc_mask & scale_mask]
    scales = scales[:,0]
    
    gauss_means[:,0] = gauss_means[:,0]*(extent[0])
    gauss_means[:,1] = -gauss_means[:,1]*(extent[1])
    gauss_means[:,2] = gauss_means[:,2]*(extent[1])

    if(flip_yz):
        tmp = gauss_means[:,1].copy()
        gauss_means[:,1] = gauss_means[:,2].copy()
        gauss_means[:,2] = tmp
        gauss_means[:,1] = -gauss_means[:,1]

        tmp = true_positions[:,0].copy()
        true_positions[:,0] = true_positions[:,1].copy()
        true_positions[:,1] = tmp

    true_positions[:,0] = true_positions[:,0]-extent[0]/2
    true_positions[:,1] = true_positions[:,1]-extent[1]/2
    true_positions[:,2] = true_positions[:,2]-extent[1]/2

    to_return = [true_positions, species, gauss_means, opacities, scales, extent]
    if(load_potential):
        to_return.append(potential)

    return to_return



def get_coordinates(atoms, plane='xy'):
    """
    Read in-plane and slice coordinates from atomic structure object.
    
    Parameters
    ----------
    atoms : ase.Atoms
        Atomic structure object containing positions and unit cell information.
    plane : str, default='xy'
        Projection plane. Options: 'xy', 'xz', 'yz'.
        
    Returns
    -------
    dict
        Dictionary containing:
        - positions : numpy.ndarray
            Transformed atomic positions
        - cell : numpy.ndarray
            Transformed unit cell vectors
        - xy_coords : numpy.ndarray
            2D projection coordinates (first two dimensions)
        - z_coords : numpy.ndarray
            Slice positions (third dimension)
        - species : list
            Chemical symbols of the atoms
    """
    atoms = atoms.copy()
    atoms = abtem.structures.orthogonalize_cell(atoms)
    atoms = abtem.structures.rotate_atoms_to_plane(atoms, plane)
    positions = atoms.positions
    cell = atoms.cell
    
    return {
        'positions': positions,
        'cell': cell,
        'xy_coords': positions[:, :2],  # Projection coordinates
        'z_coords': positions[:, 2],     # Slice position
        'species': atoms.get_chemical_symbols()
    }

def get_true_positions(file_path, return_species=False, return_potential=False):
    """
    Extract atomic positions and structure information from an atomic structure file.
    
    This function reads an atomic structure file and extracts the atomic positions,
    species, and optionally the electrostatic potential with resolution 0.25 Å.
    
    Parameters
    ----------
    file_path : str
        Path to the atomic structure file (supports formats readable by ASE).
    return_species : bool, default=False
        Whether to return the chemical species of the atoms.
    return_potential : bool, default=False
        Whether to return the electrostatic potential object.
        
    Returns
    -------
    list
        A list containing:
        - atom_positions : numpy.ndarray
            3D atomic positions with shape (n_atoms, 3)
        - extent : tuple
            Bounding box extent of the atomic structure
        - species : list, optional
            Chemical symbols of the atoms (if return_species=True)
        - potential : abtem.potentials.Potential, optional
            Electrostatic potential object (if return_potential=True)
    """
    atoms_init = ase.io.read(file_path)
    
    results = get_coordinates(atoms_init)
    species = results['species']
    atom_positions = np.concatenate([results['xy_coords'], np.expand_dims(results['z_coords'], axis=1)], axis=1)

    potential_init = abtem.potentials.Potential(
        atoms_init, 
        sampling=0.25,
        slice_thickness=0.25,
    )
    extent = potential_init.extent

    to_return = [atom_positions, extent]
    if return_species:
        to_return.append(species)
    if return_potential:
        to_return.append(potential_init)

    return to_return

def get_gt_img(file_path, slice_start, slice_end, plane='xy'):
    """
    Generate a ground truth image by projecting atomic structure through a slice range.
    
    Parameters
    ----------
    file_path : str
        Path to the atomic structure file (supports formats readable by ASE).
    slice_start : int
        Starting slice index for the projection range.
    slice_end : int
        Ending slice index for the projection range (exclusive).
    plane : str, default='xy'
        Projection plane. Options: 'xy', 'xz', 'yz'.
        
    Returns
    -------
    numpy.ndarray
        2D projection image as a numpy array with shape (height, width).
        The image represents the summed electrostatic potential over the
        specified slice range.
    """
    atoms_init = ase.io.read(file_path)
    atoms = atoms_init.copy()

    # Make cell orthogonal
    atoms = abtem.structures.orthogonalize_cell(atoms)
    atoms = abtem.structures.rotate_atoms_to_plane(atoms, plane)    

    # Rotate to projection plane
    potential_init = abtem.potentials.Potential(
        atoms, 
        sampling=0.25,
        slice_thickness=0.25,
    )   
    single_slice = np.sum(potential_init[slice_start:slice_end].array, axis=0)
    return single_slice


def plot_img_no_axis(img, ax=None, **kwargs):
    """
    Display an image without axis ticks or labels.
    
    Parameters
    ----------
    img : array-like
        Image data to display. Can be any array-like object that matplotlib
        can handle (numpy arrays, PIL images, etc.).
    ax : matplotlib.axes.Axes, optional
        Matplotlib axes object to plot on. If None, creates a new figure and axes.
    **kwargs
        Additional keyword arguments passed to matplotlib's imshow function.
            
    Returns
    -------
    matplotlib.image.AxesImage
        The image object returned by imshow.
    """
    if ax is None:
        plt.figure()
        ax = plt.gca()
    ax.imshow(img,cmap='gray',**kwargs)
    ax.set_xticks([])
    ax.set_yticks([])


def plot_projections(projections, angles=None, n_cols=5, figsize=None):
    """
    Create a grid of projection images, all using the same color scale.
    
    Parameters
    ----------
    projections : numpy.ndarray
        Array of projection images with shape (n_projections, height, width).
    angles : numpy.ndarray or list, optional
        Array of angles corresponding to each projection. If provided, labels are added to each panel.
    n_cols : int, default=5
        Number of columns in the grid layout. Grid is limited to 20 images (5 cols x 4 rows).
    figsize : tuple of int, optional
        Figure size as (width, height) in inches. If None, auto-calculates based on grid size.
    
    Returns
    -------
    matplotlib.figure.Figure
        The figure object containing all the subplots.
    numpy.ndarray
        Array of matplotlib axes objects for each subplot.
    """
    n_projections_total = projections.shape[0]
    
    # Limit to 20 projections if more are present, evenly spaced
    if n_projections_total > 20:
        indices = np.linspace(0, n_projections_total - 1, 20, dtype=int)
        projections = projections[indices]
        if angles is not None:
            angles = np.array(angles)[indices]
    
    n_projections = projections.shape[0]
    n_rows = int(np.ceil(n_projections / n_cols))
    
    # Ensure we don't exceed 4 rows (20 total images max)
    if n_rows > 4:
        n_rows = 4
        n_projections = min(n_projections, n_cols * n_rows)
        projections = projections[:n_projections]
        if angles is not None:
            angles = np.array(angles)[:n_projections]
    
    # Calculate figure size if not provided
    if figsize is None:
        aspect_ratio = projections.shape[2] / projections.shape[1]  # width / height
        figsize = (n_cols * aspect_ratio * 1.5, n_rows * 1.5)
    
    # Create figure and axes
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_projections > 1:
        axes = axes.flatten()
    else:
        axes = [axes]
    
    # Calculate global vmin and vmax for consistent color scale
    vmin = np.min(projections)
    vmax = np.max(projections)
    
    # Plot each projection with angle label
    for i in range(n_projections):
        plot_img_no_axis(projections[i], ax=axes[i], vmin=vmin, vmax=vmax)
        
        # Add angle label if provided
        if angles is not None:
            angle_label = f'{angles[i]:.1f}°'
            axes[i].text(0.02, 0.98, angle_label, transform=axes[i].transAxes,
                        fontsize=8, verticalalignment='top', horizontalalignment='left',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    # Hide extra axes if any
    for i in range(n_projections, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    return fig, axes

