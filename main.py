
from types import SimpleNamespace
import sys
import os
import math
import cv2
import torch

#init device
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Using device:', DEVICE)
device = DEVICE  # At least one of the modules expects this name..

settings = {
    'path': os.getcwd(),
    'vr_mode': False,
    'vr_eye_angle': 0.5,
    'vr_ipd': 5.0,
    # 'wh':[512, 512],
    # 'fps':60,
    'input_video': "sample-1080-short.mp4"
}

project_path = settings['path']
model_path = project_path + '/models'
frames_path = project_path+'/data/frames'
output_path = project_path+'/data/output'

midas_depth_model = "dpt_large"  # @param {type:"string"}
midas_weight = 0.3  # @param {type:"number"}
near_plane = 200  # @param {type:"number"}
far_plane = 10000  # @param {type:"number"}
fov = 40  # @param {type:"number"}
padding_mode = 'border'  # @param {type:"string"}
sampling_mode = 'bicubic'  # @param {type:"string"}
trans_scale = 1/200.0
vr_eye_angle = settings['vr_eye_angle']
vr_ipd = settings['vr_ipd']

args = {
    'midas_depth_model': midas_depth_model,
    'midas_weight': midas_weight,
    'near_plane': near_plane,
    'far_plane': far_plane,
    'fov': fov,
    'padding_mode': padding_mode,
    'sampling_mode': sampling_mode,
}

args = SimpleNamespace(**args)

# Default MiDaS depth models.
default_models = {
    "midas_v21_small": f"{model_path}/midas_v21_small-70d6b9c8.pt",
    "midas_v21": f"{model_path}/midas_v21-f6b98070.pt",
    "dpt_large": f"{model_path}/dpt_large-midas-2f21e586.pt",
    "dpt_hybrid": f"{model_path}/dpt_hybrid-midas-501f0c75.pt",
    "dpt_hybrid_nyu": f"{model_path}/dpt_hybrid_nyu-2ce69ec7.pt", }

import setup_utils as sutils
sutils.configure_env(project_path, model_path)

import model_utils as mutils
import image_transforms as transforms
import py3d_tools as p3dT

midas_model, midas_transform, midas_net_w, midas_net_h, midas_resize_mode, midas_normalization = mutils.init_midas_depth_model(
    DEVICE, default_models, args.midas_depth_model)

def process(file_path):
    print("Extracting frames...")
    os.system( "rm " + frames_path + '/*.jpg')

    vcap = cv2.VideoCapture(file_path)
    # #width = vcap.get(cv2.CAP_PROP_FRAME_WIDTH )
    # #height = vcap.get(cv2.CAP_PROP_FRAME_HEIGHT )
    fps =  vcap.get(cv2.CAP_PROP_FPS)

    os.system( "ffmpeg -y -i " + file_path + ' -r ' + str(fps) + '/1 ' +  frames_path + '/frame_%04d.jpg')

    files = sorted(glob.glob(frames_path + '/frame_*.jpg'))
    for f in files:
        filename = f.replace(frames_path+'/','')
        print("Processing ",filename)
        generate_eye_views(vr_eye_angle,vr_ipd,trans_scale,frames_path,filename,files.index(f),midas_model, midas_transform)

    os.system("ffmpeg -y -framerate " + str(fps) + " -i " + frames_path + "/eye_%4d_l.jpg " + output_path + "/l.mp4")
    os.system("ffmpeg -y -framerate " + str(fps) + " -i " + frames_path + "/eye_%4d_r.jpg " + output_path + "/r.mp4")

    if (fps!=60):
        os.system("ffmpeg -y -i " + output_path + "/l.mp4 -crf 10 -vf \"minterpolate=fps=60:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1\" " + output_path + "/l-60fps.mp4")
        os.system("ffmpeg -y -i " + output_path + "/r.mp4 -crf 10 -vf \"minterpolate=fps=60:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1\" " + output_path + "/r-60fps.mp4")

    print("Finished.")


def generate_eye_views(vr_eye_angle, vr_ipd, trans_scale, folder_path, filename, frame_num, midas_model, midas_transform):
    for i in range(2):
        theta = vr_eye_angle * (math.pi/180)  # x * 2 * pi ­ pi
        #   phi = pi / 2 ­ y * pi
        ipd = vr_ipd
        ray_origin = math.cos(theta) * ipd / 2 * (-1.0 if i == 0 else 1.0)
        ray_rotation = (theta if i == 0 else -theta)
        # translate_xyz = [-(translation_x+ray_origin)*trans_scale, translation_y*trans_scale, -translation_z*trans_scale]
        translate_xyz = [-(ray_origin)*trans_scale, 0, 0]
        rotate_xyz = [0, (ray_rotation), 0]
        rot_mat = p3dT.euler_angles_to_matrix(torch.tensor(
            rotate_xyz, device=device), "XYZ").unsqueeze(0)
        transformed_image = transforms.transform_image_3d(f'{folder_path}/{filename}', midas_model, midas_transform, DEVICE,
                                                          rot_mat, translate_xyz, args.near_plane, args.far_plane,
                                                          args.fov, padding_mode=args.padding_mode,
                                                          sampling_mode=args.sampling_mode, midas_weight=args.midas_weight, spherical=True)
        eye_file_path = folder_path + \
            f"/eye_{frame_num:04}" + ('_l' if i == 0 else '_r')+'.jpg'
        transformed_image.save(eye_file_path)
        #print("saved turbo eye version")


def main(argv):
    process(project_path+'/data/input/'+settings['input_video'])

if __name__ == "__main__":
    main(sys.argv[1:])
