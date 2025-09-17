import argparse
import json
import os

import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from scipy.spatial.transform import Rotation as R


def get_rot_mat(yaw, pitch, roll):
    r = R.from_euler("zyx", (yaw, pitch, roll), degrees=True)
    return(r.as_matrix())


def write_nerfstudio_format_from_proj_array(measurements, angles, write_path):
    img_path = os.path.join(write_path, 'images')

    if not os.path.exists(img_path):
        os.makedirs(img_path)

    json_path = os.path.join(write_path, 'transforms.json')

    frames_list = []

    for i,angle in enumerate(angles):
        # Scale to 0-255 for nerfstudio compatibility
        im = Image.fromarray(256*measurements[i,...]/np.max(measurements)).convert('RGB')

        frame_str = 'frame_'+str(i).zfill(5)+'.png'
        frame_path = os.path.join(img_path,frame_str)
        im.save(frame_path)

        rot_matrix = get_rot_mat(0, 0, angle-90)
        matrix = np.append(rot_matrix,[[0],[np.cos((angle)/360.0*2*np.pi)],[np.sin((angle)/360.0*2*np.pi)]],axis=1)
        matrix = np.append(matrix,[[0,0,0,1]],axis=0)

        frames_list.append({
            "file_path": os.path.join('./images',frame_str),
            "transform_matrix": matrix.tolist()
        })

        camera_model = {
            "camera_model": "ORTHOPHOTO",
            "fl_x": measurements.shape[2], # // focal length x
            "fl_y": measurements.shape[1], # // focal length y
            "cx": measurements.shape[2]/2, # // principal point x
            "cy": measurements.shape[1]/2, # // principal point y
            "w": measurements.shape[2], # // image width
            "h": measurements.shape[1], # // image height
            "frames": frames_list # // ... per-frame intrinsics and extrinsics parameters
        }

        j = json.dumps(camera_model, indent=4)
        with open(json_path, 'w') as f:
            print(j, file=f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--projection_data_dir', type=str, default='projections',
                       help='Directory containing projection files')
    parser.add_argument('--save_dir', type=str, default='nerfstudio_format',
                       help='Directory to save nerfstudio format files')
    args = parser.parse_args()
    projection_data_dir = args.projection_data_dir
    save_dir = args.save_dir
    
    for filename in os.listdir(projection_data_dir):
        if filename.endswith('_proj.npy'):
            filepath = os.path.join(projection_data_dir, filename)
            measurements = np.load(filepath)

            angles_path = os.path.join(projection_data_dir, filename.replace('_proj.npy', '_angles.npy'))
            angles = np.load(angles_path)
            
            write_path = os.path.join(save_dir,filename.split('_proj.npy')[0])
            write_nerfstudio_format_from_proj_array(measurements, angles, write_path)