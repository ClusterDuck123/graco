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

def tvd(P,Q):
   if   np.isnan(P).all() and np.isnan(Q).all():
       return 0.
   elif np.isnan(P).all()  or np.isnan(Q).all():
       return 1.
   else:
       return np.sum(np.abs(P-Q))/2


def get_orca_path():
    return f"{CURRENT_DIRECTORY}/orca/orca"
def get_edgelist_path():
    return f"{CURRENT_DIRECTORY}/tmp/edgelist.tmp"
def get_orbits_path():
    return f"{CURRENT_DIRECTORY}/tmp/orbits.tmp"
def get_coefficients_path():
    return f"{CURRENT_DIRECTORY}/tmp/clusterings.tmp"


def run_cmd(cmd):
    completed_process = subprocess.run(cmd,
                                       stderr=subprocess.PIPE,
                                       check=True)
    if completed_process.stderr:
        raise subprocess.CalledProcessError(cmd = cmd,
                    returncode = completed_process.returncode)


gcc_header = {'c_02' :r'$c_{0\rightarrow2}$' , 'c_03' :r'$c_{0\rightarrow3}$' ,
               'c_15' :r'$c_{1\rightarrow5}$' , 'c_18' :r'$c_{1\rightarrow8}$' ,
               'c_110':r'$c_{1\rightarrow10}$', 'c_112':r'$c_{1\rightarrow12}$',
               'c_27' :r'$c_{2\rightarrow7}$' , 'c_211':r'$c_{2\rightarrow11}$',
               'c_213':r'$c_{2\rightarrow13}$',
               'c_311':r'$c_{3\rightarrow11}$', 'c_313':r'$c_{3\rightarrow13}$',
               'c_314':r'$c_{3\rightarrow14}$'}
class Write:
    @staticmethod
    def edgelist(G):
        file_in  = get_edgelist_path()
        nx.write_edgelist(G, file_in, data=False)
        N, M = G.number_of_nodes(), G.number_of_edges()
        with open(file_in, 'r+') as f:
            content = f.read()
            f.seek(0)
            f.write(f"{N} {M}\n" + content)

    @staticmethod
    def orbits(G, file_out=None, graphlet_nodes=4):
        Write.edgelist(G)

        file_in  = get_edgelist_path()

        if file_out is None:
            file_out = get_orbits_path()

        orca     = get_orca_path()
        orca_cmd = [orca, str(graphlet_nodes), file_in, file_out]
        run_cmd(orca_cmd)

    @staticmethod
    def coefficients(G):
        Write.orbits(G)

        file_in  = get_orbits_path()
        file_out = get_coefficients_path()

        Pau_script = get_Pau_script()
        Pau_cmd    = [Pau_script, file_in, file_out]
        run_cmd(Pau_cmd)


class Calculate:
    @staticmethod
    def orbits(G, dtype=pd.DataFrame):

        label_mapping   = {name:n for n,name in enumerate(G)}
        reverse_mapping = {value:key for key,value in label_mapping.items()}

        G = nx.relabel_nodes(G, label_mapping)

        Write.orbits(G=G)
        file_in = get_orbits_path()

        if   dtype == pd.DataFrame:
            columnn_names = [f'o{i}' for i in range(15)]
            df = pd.read_csv(file_in, delimiter=' ', names=columnn_names)
            return df.rename(index=reverse_mapping)

        elif dtype == np.ndarray:
            return np.genfromtxt(file_in)
        else:
            raise TypeError('Please provide an appropriate type.')

    @staticmethod
    def coefficients(G, dtype=pd.DataFrame):

        label_mapping   = {name:n for n,name in enumerate(G)}
        reverse_mapping = {value:key for key,value in label_mapping.items()}

        G = nx.relabel_nodes(G, label_mapping)

        Write.coefficients(G=G)
        file_in = get_coefficients_path()

        if   dtype == pd.DataFrame:
            columnn_names = ['c_02' , 'c_03',
                             'c_15' , 'c_18' , 'c_110', 'c_112',
                             'c_27' , 'c_211', 'c_213',
                             'c_311', 'c_313', 'c_314']

            df = pd.read_csv(file_in, delimiter=' ', names=columnn_names)
            return df.rename(index=reverse_mapping)

        elif dtype == np.ndarray:
            return np.genfromtxt(file_in)
        else:
            raise TypeError('Please provide an appropriate type.')
