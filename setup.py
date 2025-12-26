from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, include_paths, library_paths
import os.path as osp
import os, glob
from pathlib import Path


asmk = Path(__file__).parent / "thirdparty" / "mast3r" / "asmk"
pyimgui_path = Path(__file__).parent / "thirdparty"/ "in3d" / "thirdparty" / "pyimgui"

ROOT = osp.dirname(osp.abspath(__file__))
torch_include_dirs = include_paths()
torch_library_dirs = library_paths()
conda_prefix = os.environ.get("PREFIX", os.environ.get("CONDA_PREFIX", ""))
eigen_path = osp.join(conda_prefix, 'include', 'eigen3')

setup(
    name='vslamlab-mast3rslam-mono',
    version='0.1',
    description='MASt3R-SLAM mono mode',
    packages=find_packages(where='.') +
        find_packages(where='thirdparty/mast3r', include=["mast3r", "mast3r.*","dust3r", "dust3r.*"]) +
        find_packages(where='thirdparty/in3d', include=["in3d", "in3d.*"]),  
    package_dir={
        'mast3r': 'thirdparty/mast3r/mast3r',
        'dust3r': 'thirdparty/mast3r/dust3r',
        'in3d': 'thirdparty/in3d/in3d',
    },    
    include_package_data=True,
    package_data={
        'mast3r_slam': ['resources/programs/*.glsl'],
        'in3d': ['resources/programs/*.glsl'],
        'mast3r_slam.configs': ['*.yaml'],
    },
    entry_points={
        'console_scripts': [
            'vslamlab_mast3rslam_mono = mast3r_slam.vslamlab_mast3rslam_mono:main',
        ]
    },
    install_requires=[f"asmk @ {asmk.as_uri()}", f"imgui @ {pyimgui_path.as_uri()}"],
    ext_modules=[
        CUDAExtension(
            name='mast3r_slam_backends',
            include_dirs=torch_include_dirs + [
                osp.join(ROOT, 'mast3r_slam/backend/include'),
                eigen_path
                ],
            library_dirs=torch_library_dirs,
            sources=[
                'mast3r_slam/backend/src/gn.cpp',
                'mast3r_slam/backend/src/gn_kernels.cu',
                'mast3r_slam/backend/src/matching_kernels.cu',
            ],
            extra_compile_args={
                'cores': ["j8"],
                'cxx': ['-O3', '-D_GLIBCXX_USE_CXX11_ABI=1'],
                'nvcc': [
                    '-O3',
                    '-gencode=arch=compute_86,code=sm_86',
                ]
            }
        ),
    ],
    cmdclass={'build_ext': BuildExtension.with_options(no_python_abi_suffix=True)},
)
