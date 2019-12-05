import pandas as pd
import numpy as np
import subprocess
import os

CURRENT_DIRECTORY = os.path.dirname(__file__)
CPP_DIRECTORY = f"{CURRENT_DIRECTORY}/cpp"
TMP_DIRECTORY = f"{CURRENT_DIRECTORY}/tmp"
MATRIX_IN = f"{TMP_DIRECTORY}/matrix.in"
MATRIX_OUT = f"{TMP_DIRECTORY}/matrix.out"

def run_cmd(cmd):
    completed_process = subprocess.run(cmd,
                                       stderr=subprocess.PIPE,
                                       check=True)
    if completed_process.stderr:
        raise subprocess.CalledProcessError(cmd = cmd,
                    returncode = completed_process.returncode)

def write_matrix(M, fmt):
    np.savetxt(MATRIX_IN,
               M,
               header=' '.join(map(str,M.shape)),
               fmt=fmt)


def GDV_similarity(M):
    if type(M) == pd.DataFrame:
        M = M.values
    if   M.dtype == int:
        write_matrix(M, fmt='%d')
        cmd = [f"{CPP_DIRECTORY}/int_GDV-similarity", MATRIX_IN, MATRIX_OUT]
        run_cmd(cmd)
        return np.loadtxt(MATRIX_OUT)
    else:
        raise Exception("Datatype not understood.")


def normalized1_lp(M, p=1):
    if type(M) == pd.DataFrame:
        M = M.values
    if  M.dtype == int:
        write_matrix(M, fmt='%d')
        if p == np.inf: p = 0
        cmd = [f"{CPP_DIRECTORY}/int_normalized1_lp", str(p), MATRIX_IN, MATRIX_OUT]
        run_cmd(cmd)
        return np.loadtxt(MATRIX_OUT)
    elif M.dtype == float:
        write_matrix(M, fmt='%.7f')
        if p == np.inf: p = 0
        cmd = [f"{CPP_DIRECTORY}/float_normalized1_lp", str(p), MATRIX_IN, MATRIX_OUT]
        run_cmd(cmd)
        return np.loadtxt(MATRIX_OUT)
    else:
        raise Exception(f"Datatype not understood. {M.dtype}")

def distance_matrix(M, distance):
    if   distance == 'normalized1_l1':
        return normalized1_lp(M, p=1)
    elif distance == 'normalized1_l2':
        return normalized1_lp(M, p=2)
    elif distance == 'normalized1_linf':
        return normalized1_lp(M, p=inf)
    else:
        return cdist(M, M, distance)
