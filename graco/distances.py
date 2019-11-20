#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 17:22:20 2019
@author: clusterduck
"""

import networkx as nx
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


class Write:
    @staticmethod
    def GDV(M):
        np.savetxt(MATRIX_IN,
                   M,
                   header=' '.join(map(str,M.shape)),
                   fmt='%d')

class Calculate:
    @staticmethod
    def GDV_similarity(M):
        Write.GDV(M)
        cmd = [f"{CPP_DIRECTORY}/tijana", MATRIX_IN, MATRIX_OUT]
        run_cmd(cmd)
        return np.loadtxt(MATRIX_OUT)
