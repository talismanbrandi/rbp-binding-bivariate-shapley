import numpy as np
import os
import sys

os.chdir('/work/jdy/davin/rbp-binding-bivariate-shapley')
sys.path.append('./')
from shared_utils import *

def mkdir_p(dir):
    '''make a directory (dir) if it doesn't exist'''
    if not os.path.exists(dir):
        os.mkdir(dir)
    

job_directory = '/work/jdy/davin/tmp'
script_path = '/work/jdy/davin/rbp-binding-bivariate-shapley/scripts'
# Make top level directories
mkdir_p(job_directory)

seed = 0
num_samples = 1000
min_idx_list = np.arange(0,100000, num_samples).tolist()


for i, min_idx in enumerate(min_idx_list):

    job_name = '%s.job' % str(min_idx)
    job_file = os.path.join(job_directory, job_name)
    python_script = 'bivshap_iterate.py'

    cmd = os.path.join(script_path, python_script)
    args = [
        '--dataset_min_index %s' % str(min_idx),
        '--dataset_samples %s' % str(num_samples),
        '--seed %s' % str(seed),
    ]
    cmd = cmd + ' ' + ' '.join(args)

    utils_slurm.submit_slurm(
        cmd,
        job_file,
        conda_env = 'a100',
        partition = 'ai-jumpstart',
        mem = 32,
        n_cpu =  2,
        time_hrs = 10,
        job_name = job_name,
        n_gpu = 0,
    )
