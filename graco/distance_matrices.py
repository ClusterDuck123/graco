from scipy.spatial.distance import pdist, squareform
import pandas as pd
import numpy as np
import subprocess
import random
import time
import graco
import os

GRACO_PATH = os.path.dirname(graco.__file__)
CPP_PATH = f"{GRACO_PATH}/cpp"
TMP_PATH = f"{GRACO_PATH}/tmp"

def _get_timestamp():
    return time.time()*random.random()

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
    if M.dtype == int:
        timestamp = _get_timestamp()
        matrix_in  = f"{TMP_PATH}/{timestamp}.in"
        matrix_out = f"{TMP_PATH}/{timestamp}.out"
        write_matrix(matrix_in, M, fmt='%d')
        cmd = [f"{CPP_PATH}/int_GDV-similarity", matrix_in, matrix_out]
        run_cmd(cmd)
        D_arr = np.loadtxt(matrix_out)

        os.remove(matrix_in)
        os.remove(matrix_out)

        return D_arr
    else:
        raise Exception(f"Datatype not integer. {M.dtype}")


def hellinger(M):
    if type(M) == pd.DataFrame:
        M = M.values
    M = (M.T / M.sum(axis=1)).T  #transposing back and forth for broadcasting

    timestamp = _get_timestamp()
    matrix_in  = f"{TMP_PATH}/hell{timestamp}.in"
    matrix_out = f"{TMP_PATH}/hell{timestamp}.out"
    write_matrix(matrix_in, M, fmt='%.7f')
    cmd = [f"{CPP_PATH}/hellinger", matrix_in, matrix_out]
    run_cmd(cmd)

    D_arr = np.loadtxt(matrix_out)

    os.remove(matrix_in)
    os.remove(matrix_out)

    return D_arr


def js_divergence(M):
    if type(M) == pd.DataFrame:
        M = M.values
    M = (M.T / M.sum(axis=1)).T  #transposing back and forth for broadcasting

    timestamp = _get_timestamp()
    matrix_in  = f"{TMP_PATH}/js{timestamp}.in"
    matrix_out = f"{TMP_PATH}/js{timestamp}.out"
    write_matrix(matrix_in, M, fmt='%.7f')
    cmd = [f"{CPP_PATH}/js_divergence", matrix_in, matrix_out]
    run_cmd(cmd)

    D_arr = np.loadtxt(matrix_out)

    os.remove(matrix_in)
    os.remove(matrix_out)

    return D_arr
