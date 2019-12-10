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
            columnn_names = [f'{i}' for i in range(15)]
            df = pd.read_csv(file_in, delimiter=' ', names=columnn_names)
            df.columns.name = 'Orbit'
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
            ('D','0','2') : 2*GDV['2']  / (GDV['0'] * (GDV['0']-1)),
            ('D','0','3') : 2*GDV['3']  / (GDV['0'] * (GDV['0']-1)),

            ('D','1','5') : 1*GDV['5' ] / (GDV['1'] * (GDV['0']-1)),
            ('D','1','8') : 2*GDV['8' ] / (GDV['1'] * (GDV['0']-1)),
            ('D','1','10'): 1*GDV['10'] / (GDV['1'] * (GDV['0']-1)),
            ('D','1','12'): 2*GDV['12'] / (GDV['1'] * (GDV['0']-1)),

            ('D','2','7') : 3*GDV['7']  / (GDV['2'] * (GDV['0']-2)),
            ('D','2','11'): 2*GDV['11'] / (GDV['2'] * (GDV['0']-2)),
            ('D','2','13'): 1*GDV['13'] / (GDV['2'] * (GDV['0']-2)),

            ('D','3','11'): 1*GDV['11'] / (GDV['3'] * (GDV['0']-2)),
            ('D','3','13'): 2*GDV['13'] / (GDV['3'] * (GDV['0']-2)),
            ('D','3','14'): 3*GDV['14'] / (GDV['3'] * (GDV['0']-2)),


            ('A', '0-0', '1') : 1*GDV['1'] / (GDV['1'] + 2*GDV['3']),
            ('A', '0-0', '3') : 2*GDV['3'] / (GDV['1'] + 2*GDV['3']),

            ('A', '2-1', '4')  : 1*GDV['4']  / (1*GDV['4'] + 2*GDV['8'] + 1*GDV['10'] + 2*GDV['13']),
            ('A', '2-1', '8')  : 1*GDV['8']  / (1*GDV['4'] + 2*GDV['8'] + 1*GDV['10'] + 2*GDV['13']),
            ('A', '2-1', '10') : 1*GDV['10'] / (1*GDV['4'] + 2*GDV['8'] + 1*GDV['10'] + 2*GDV['13']),
            ('A', '2-1', '13') : 1*GDV['13'] / (1*GDV['4'] + 2*GDV['8'] + 1*GDV['10'] + 2*GDV['13']),

            ('A', '1-2', '6')  : 1*GDV['6']  / (1*GDV['6'] + 1*GDV['10'] + 1*GDV['13']),
            ('A', '1-2', '10') : 1*GDV['10'] / (1*GDV['6'] + 1*GDV['10'] + 1*GDV['13']),
            ('A', '1-2', '13') : 1*GDV['13'] / (1*GDV['6'] + 1*GDV['10'] + 1*GDV['13']),

            ('A', '3-3', '9')  : 1*GDV['9']  / (1*GDV['9'] + 2*GDV['12'] + 3*GDV['14']),
            ('A', '3-3', '12') : 1*GDV['12'] / (1*GDV['9'] + 2*GDV['12'] + 3*GDV['14']),
            ('A', '3-3', '14') : 1*GDV['14'] / (1*GDV['9'] + 2*GDV['12'] + 3*GDV['14']),


            ('G', '0-0', '1') : 1*GDV['1'] / (GDV['1'] + 2*GDV['3']),
            ('G', '0-0', '3') : 2*GDV['3'] / (GDV['1'] + 2*GDV['3']),

            ('G', '1-1', '4')  : 1*GDV['4']  / (1*GDV['4'] + 2*GDV['8'] + 2*GDV['9'] + 2*GDV['12']),
            ('G', '1-1', '8')  : 2*GDV['8']  / (1*GDV['4'] + 2*GDV['8'] + 2*GDV['9'] + 2*GDV['12']),
            ('G', '1-1', '9')  : 2*GDV['9']  / (1*GDV['4'] + 2*GDV['8'] + 2*GDV['9'] + 2*GDV['12']),
            ('G', '1-1', '12') : 2*GDV['12'] / (1*GDV['4'] + 2*GDV['8'] + 2*GDV['9'] + 2*GDV['12']),

            ('G', '1-2', '6')  : 2*GDV['6']  / (2*GDV['6']  + 1*GDV['10'] + 2*GDV['9']  + 2*GDV['12']),
            ('G', '1-2', '9')  : 2*GDV['9']  / (2*GDV['6']  + 1*GDV['10'] + 2*GDV['9']  + 2*GDV['12']),
            ('G', '1-2', '10') : 1*GDV['10'] / (2*GDV['6']  + 1*GDV['10'] + 2*GDV['9']  + 2*GDV['12']),
            ('G', '1-2', '12') : 2*GDV['12'] / (2*GDV['6']  + 1*GDV['10'] + 2*GDV['9']  + 2*GDV['12']),

            ('G', '2-1', '5')  : 1*GDV['5']  / (1*GDV['5']  + 2*GDV['11'] + 2*GDV['8']  + 2*GDV['13']),
            ('G', '2-1', '8')  : 1*GDV['8']  / (1*GDV['5']  + 2*GDV['11'] + 2*GDV['8']  + 2*GDV['13']),
            ('G', '2-1', '11') : 1*GDV['11'] / (1*GDV['5']  + 2*GDV['11'] + 2*GDV['8']  + 2*GDV['13']),
            ('G', '2-1', '13') : 1*GDV['13'] / (1*GDV['5']  + 2*GDV['11'] + 2*GDV['8']  + 2*GDV['13']),

            ('G', '3-3', '10') : 1*GDV['10'] / (1*GDV['10'] + 2*GDV['12'] + 6*GDV['14'] + 2*GDV['13']),
            ('G', '3-3', '12') : 2*GDV['12'] / (1*GDV['10'] + 2*GDV['12'] + 6*GDV['14'] + 2*GDV['13']),
            ('G', '3-3', '13') : 2*GDV['13'] / (1*GDV['10'] + 2*GDV['12'] + 6*GDV['14'] + 2*GDV['13']),
            ('G', '3-3', '14') : 6*GDV['14'] / (1*GDV['10'] + 2*GDV['12'] + 6*GDV['14'] + 2*GDV['13'])
        })
        GCV.columns.name  = 'Coefficient'
        GCV.columns.names = ['Order', 'Source', 'Target']
        if   dtype == pd.DataFrame:
            return GCV
        elif dtype == np.ndarray:
            return np.genfromtxt(file_in)
        else:
            raise TypeError('Please provide an appropriate type.')
