from nerfstudio.models import splatfacto
import copy
import os
import numpy as np

run_limited_ablation = True
run_ablation = False
run_allparticles_full = True
run_allparticles_missingwedge = True

#################################################################################################
# Ablation experiments comparing:
# - Our full method (ours)
# - Without anisotropic gaussians (noiso)
# - Without gaussian clustering (nocluster)
# - Without both anisotropic gaussians and clustering (puresplat)
#################################################################################################

if(run_limited_ablation):
    label = 'rebuttal_iccp_limited'
    names = ['Au_NP']
    for range_str in ['-90_90_3','-70_70_1']:    
        for name in names:
            print(name)
            cmd = f"ns-train splatfacto --data data/nerfstudio_format/sim_{name}_projections_{range_str}_sigma_2 \
                --vis wandb --experiment-name sim_{name}_{range_str}_ours_{label} --project-name atomic-gaussians"
            os.system(cmd)

#################################################################################################
# Ablation experiments comparing:
# - Our full method (ours)
# - Without anisotropic gaussians (noiso)
# - Without gaussian clustering (nocluster)
# - Without both anisotropic gaussians and clustering (puresplat)
#################################################################################################

if(run_ablation):
    # Map config names to their specific arguments
    config_args = {
        'ours': '',
        'noiso': '--pipeline.model.force-isotropic=False',
        'nocluster': '--pipeline.model.cluster-gaussians=False', 
        'puresplat': '--pipeline.model.force-isotropic=False --pipeline.model.cluster-gaussians=False'
    }

    names = ['ours', 'noiso', 'nocluster', 'puresplat']

    
    for name in names:
        print(name)
        config = config_args[name]
        cmd = f"ns-train splatfacto --data data/nerfstudio_format/nonoise_ablation/ZrTe_model_full \
            --vis wandb --experiment-name sim_tube_{name}_{label} --project-name atomic-gaussians {config}"
        os.system(cmd)

#################################################################################################
# Run experiments with default settings (anisotropic gaussians + clustering) on:
# - Full dataset with all particles
# - Missing wedge dataset with all particles 
#################################################################################################

names = ['Au_NP','FePt_NP','aPd_NP','HEA_NP']
label = 'final_iccp'
if(run_allparticles_full):
    for name in names:
        if(os.path.exists('outputs/sim_{name}_full_ours_{label}')):
            print(f"Skipping {name} because it already exists")
            continue
        cmd = f"ns-train splatfacto --data data/nerfstudio_format/sim_{name}_projections_-90_90_1_sigma_2 \
            --vis wandb --experiment-name sim_{name}_full_ours_{label} --project-name atomic-gaussians"
        os.system(cmd)

if(run_allparticles_missingwedge):
    for name in names:
        if(os.path.exists('outputs/sim_{name}_mw_ours_{label}')):
            print(f"Skipping {name} because it already exists")
            continue
        cmd = f"ns-train splatfacto --data data/nerfstudio_format/sim_{name}_projections_-70_68_3_sigma_2 \
            --vis wandb --experiment-name sim_{name}_mw_ours_{label} --project-name atomic-gaussians"
        os.system(cmd)
