# Gaussian Atoms

A Gaussian Parameterization for Direct Atomic Structure Identification in Electron Tomography

keywords: atomic electron tomography, Gaussian splatting

## Dependencies

All dependencies required to run this code are specified in `environment.yml`. To create an anaconda environment with those dependencies installed, run `conda env create --name <env> --file environment.yml`. 

This command will automatically install the appropriate nerfstudio and gsplat dependencies, which are notably _not_ the original nerfstudio implementations. Our version is specific to the parallel beam tomography forward model in AET and includes configurable options to enforce physical constraints (atomic isotropy and minimum interatomic distances). If you have a previously existing nerfstudio/gsplat installation, you may need to resolve the resulting conflict.

You will also need to add this repo to your python path (if you're using conda, `conda-develop /path/to/gaussian-atoms/`).

## Workflow

The complete workflow from atomic structure files to reconstructed volumes is demonstrated in `analysis/example/example.ipynb`. This notebook shows how to:
1. Simulate AET projections from atomic structure files
2. Train a Gaussian splatting model to fit the projections
3. Load and visualize the reconstruction
4. Compare results with classical baseline methods (FBP and SART)

## Components
The workflow consists of several components that can be used independently:

**Data Preparation:**
- `data/make_tomo_projections.py`: Simulates electron microscopy projections from atomic structure files (.xyz format) with configurable noise and blur parameters
- `data/make_nerfstudio_format.py`: Converts projection arrays into nerfstudio format, creating the necessary directory structure and `transforms.json` file

**Training:**
- Training is performed using nerfstudio's `splatfacto` method. 

Example training command:
```bash
ns-train splatfacto --data /path/to/projections \
    --vis wandb --experiment-name $job_name$ --project-name $project_name$
```

**Baseline Comparisons:**
- `baselines/classical_baselines.py`: Implements classical reconstruction baselines (Filtered Backprojection and SART) for comparison against the Gaussian Splatting approach

## Paper

If you use the ideas or implementation in this repository, please cite our paper:

```
@inproceedings{singh2025gaussian,
  title={A Gaussian Parameterization for Direct Atomic Structure Identification in Electron Tomography},
  author={Singh, Nalini M and Chien, Tiffany and McCray, Arthur RC and Ophus, Colin and Waller, Laura},
  booktitle={2025 IEEE International Conference on Computational Photography (ICCP)},
  pages={1--10},
  year={2025},
  organization={IEEE}
}
```