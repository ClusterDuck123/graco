from scipy.spatial.distance import pdist, squareform
import pandas as pd
import numpy as np
import subprocess
import time
import graco
import os

GRACO_PATH = os.path.dirname(graco.__file__)
CPP_PATH = f"{GRACO_PATH}/cpp"
TMP_PATH = f"{GRACO_PATH}/tmp"

def run_cmd(cmd):
    completed_process = subprocess.run(cmd,
                                       stderr=subprocess.PIPE,
                                       check=True)
    if completed_process.stderr:
        raise subprocess.CalledProcessError(cmd = cmd,
                    returncode = completed_process.returncode)

def write_matrix(filename, M, fmt):
    np.savetxt(filename,
               M,
               header=' '.join(map(str,M.shape)),
               fmt=fmt)


def GDV_similarity(M):
    if type(M) == pd.DataFrame:
        M = M.values
    else:
        raise Exception(f"Datatype not understood. {type(M)}")
    if M.dtype == int:
        timestamp = time.time()
        matrix_in  = f"{TMP_PATH}/{timestamp}.in"
        matrix_out = f"{TMP_PATH}/{timestamp}.out"
        write_matrix(matrix_in, M, fmt='%d')
        cmd = [f"{CPP_PATH}/int_GDV-similarity", matrix_in, matrix_out]
        run_cmd(cmd)
        return np.loadtxt(matrix_out)
    else:
        raise Exception(f"Datatype not integer. {M.dtype}")


def normalized1_lp(M, p=1):
    if type(M) == pd.DataFrame:
        M = M.values
    if  M.dtype == int:
        timestamp = time.time()
        matrix_in  = f"{TMP_PATH}/{timestamp}.in"
        matrix_out = f"{TMP_PATH}/{timestamp}.out"
        write_matrix(matrix_in, M, fmt='%d')
        if p == np.inf: p = 0
        cmd = [f"{CPP_PATH}/int_normalized1_lp", str(p), matrix_in, matrix_out]
        run_cmd(cmd)
        return np.loadtxt(matrix_out)
    elif M.dtype == float:
        if p == np.inf: p = 0
        timestamp = time.time()
        matrix_in  = f"{TMP_PATH}/n1{p}_{timestamp}.in"
        matrix_out = f"{TMP_PATH}/n1{p}_{timestamp}.out"
        write_matrix(matrix_in, M, fmt='%.7f')
        cmd = [f"{CPP_PATH}/float_normalized1_lp", str(p), matrix_in, matrix_out]
        run_cmd(cmd)
        return np.loadtxt(matrix_out)
    else:
        raise Exception(f"Datatype not understood. {M.dtype}")

def normalized2_lp(M, p=1):
    if type(M) == pd.DataFrame:
        M = M.values
    if  M.dtype == int:
        timestamp = time.time()
        matrix_in  = f"{TMP_PATH}/n2{p}_{timestamp}.in"
        matrix_out = f"{TMP_PATH}/n2{p}_{timestamp}.out"
        write_matrix(matrix_in, M, fmt='%d')
        if p == np.inf: p = 0
        cmd = [f"{CPP_PATH}/int_normalized2_lp", str(p), matrix_in, matrix_out]
        run_cmd(cmd)
        return np.loadtxt(matrix_out)
    elif M.dtype == float:
        if p == np.inf: p = 0
        timestamp = time.time()
        matrix_in  = f"{TMP_PATH}/n2{timestamp}.in"
        matrix_out = f"{TMP_PATH}/n2{timestamp}.out"
        write_matrix(matrix_in, M, fmt='%.7f')
        cmd = [f"{CPP_PATH}/float_normalized2_lp", str(p), matrix_in, matrix_out]
        run_cmd(cmd)
        return np.loadtxt(matrix_out)
    else:
        raise Exception(f"Datatype not understood. {M.dtype}")

def hellinger(M):
    if type(M) == pd.DataFrame:
        M = M.values

    timestamp = time.time()
    matrix_in  = f"{TMP_PATH}/hell{timestamp}.in"
    matrix_out = f"{TMP_PATH}/hell{timestamp}.out"
    write_matrix(matrix_in, M, fmt='%.7f')
    cmd = [f"{CPP_PATH}/hellinger", matrix_in, matrix_out]
    run_cmd(cmd)
    return np.loadtxt(matrix_out)
