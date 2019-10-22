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

def run_cmd(cmd):
    completed_process = subprocess.run(cmd,
                                       stderr=subprocess.PIPE,
                                       check=True)
    if completed_process.stderr:
        raise subprocess.CalledProcessError(cmd = cmd,
                    returncode = completed_process.returncode)


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
        if   type(G) == pd.DataFrame:
            GDV=G
        elif type(G) == nx.Graph:
            GDV = Calculate.orbits(G)
        else:
            raise TypeError('Please provide an appropriate type.')

        GCV = pd.DataFrame({
            'c_0-2' : 2*GDV['o2']  / (GDV['o0'] * (GDV['o0']-1)),
            'c_0-3' : 2*GDV['o3']  / (GDV['o0'] * (GDV['o0']-1)),

            'c_1-5' : 1*GDV['o5' ] / (GDV['o1'] * (GDV['o0']-1)),
            'c_1-8' : 2*GDV['o8' ] / (GDV['o1'] * (GDV['o0']-1)),
            'c_1-10': 1*GDV['o10'] / (GDV['o1'] * (GDV['o0']-1)),
            'c_1-12': 2*GDV['o12'] / (GDV['o1'] * (GDV['o0']-1)),

            'c_2-7' : 3*GDV['o7']  / (GDV['o2'] * (GDV['o0']-2)),
            'c_2-11': 2*GDV['o11'] / (GDV['o2'] * (GDV['o0']-2)),
            'c_2-13': 1*GDV['o13'] / (GDV['o2'] * (GDV['o0']-2)),

            'c_3-11': 1*GDV['o11'] / (GDV['o3'] * (GDV['o0']-2)),
            'c_3-13': 2*GDV['o13'] / (GDV['o3'] * (GDV['o0']-2)),
            'c_3-14': 3*GDV['o14'] / (GDV['o3'] * (GDV['o0']-2))
        })
        if   dtype == pd.DataFrame:
            return GCV
        elif dtype == np.ndarray:
            return np.genfromtxt(file_in)
        else:
            raise TypeError('Please provide an appropriate type.')
