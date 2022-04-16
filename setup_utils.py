
import os,sys
import subprocess


def gitclone(url):
  res = subprocess.run(['git', 'clone', url], stdout=subprocess.PIPE).stdout.decode('utf-8')
  print(res)

def wget(url, outputdir):
  res = subprocess.run(['wget', url, '-P', f'{outputdir}'], stdout=subprocess.PIPE).stdout.decode('utf-8')
  print(res)

def createPath(filepath):
    os.makedirs(filepath, exist_ok=True)

def configure_env(project_path,model_path):
    print("Configuring environment...")
    sys.path.append(f'{project_path}/AdaBins')

    # try:
    #     import py3d_tools
    # except:
    #     if os.path.exists('pytorch3d-lite') is not True:
    #         gitclone("https://github.com/MSFTserver/pytorch3d-lite.git")
    sys.path.append(f'{project_path}/pytorch3d-lite')


    # try:
    #     from midas.dpt_depth import DPTDepthModel
    # except:
    #     if os.path.exists('MiDaS') is not True:
    #         gitclone("https://github.com/isl-org/MiDaS.git")
        #if os.path.exists('MiDaS/midas_utils.py') is not True:
            #shutil.move('MiDaS/utils.py', 'MiDaS/midas_utils.py')
    
    if not os.path.exists(f'{model_path}/dpt_large-midas-2f21e586.pt'):
        wget("https://github.com/intel-isl/DPT/releases/download/1_0/dpt_large-midas-2f21e586.pt", model_path)
    sys.path.append(f'{project_path}/MiDaS')


    createPath(f'{project_path}/data/frames')
    createPath(f'{project_path}/data/output')

