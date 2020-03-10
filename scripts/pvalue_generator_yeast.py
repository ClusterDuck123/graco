from goatools import obo_parser
from scipy.stats import hypergeom

import os
import sys
import numpy as np
import pandas as pd
import networkx as nx

"""
Takes network, feature and metric as input and calculates distance matrix.
"""
# =============================================================================
#  ---------------------------- GLOBAL PARAMETERS ----------------------------
# =============================================================================

RUN = sys.argv[1]
MIN_CLUSTERS = 40
MAX_CLUSTERS = 50

# =============================================================================
#  -------------------------------- FUNCTIONS --------------------------------
# =============================================================================

def cluster2GO(cluster):
    return set.union(*(gene2GO.get(gene, set()) for gene in cluster))

def is_annotated_in(gene, GO_set):
    return not gene2GO.get(gene,set()).isdisjoint(GO_set)

# =============================================================================
#  ---------------------------------- MAIN -----------------------------------
# =============================================================================

def main(network, feature, metric, method):
    CLUSTER_DIRECTORY = f"{YEAST_DIRECTORY}/clusterings/{network}/{feature}/{metric}/{method}"
    PVALUE_DIRECTORY  = f"{YEAST_DIRECTORY}/pvalues/{network}/{feature}/{metric}/{method}/{aspect}"

    G_nx = nx.read_edgelist(f"{YEAST_DIRECTORY}/networks/{network}.txt")
    annotation_df = pd.read_csv(f"{ANNOTATION_DIRECTORY}/GO_{aspect}_systematic_SGD.csv")
    annotation_df = annotation_df[annotation_df.Systematic_ID.isin(G_nx)]
    go_dag = obo_parser.GODag(f"{RAW_DATA_DIRECTORY}/go-basic.obo")

    GO_population = set(annotation_df.GO_ID)

    # Conversion dictionaries
    GO2genes = pd.Series({go_id: set(genes.Systematic_ID)
                            for go_id, genes in annotation_df.groupby('GO_ID')},
                         name='nb_genes')

    gene2GO  = {gene : set(go_ids.GO_ID) for gene, go_ids in annotation_df.groupby('Systematic_ID')}
    global_GO_counter = GO2genes.apply(len)

    for n_clusters in range(MIN_CLUSTERS, MAX_CLUSTERS+1):
        with open(f"{CLUSTER_DIRECTORY}/{RUN}_{n_clusters}.txt", 'r') as f:
             cluster_list = [set(line.split()) for line in f]
        cluster_df = pd.Series({gene:cluster_idx
                                    for cluster_idx,cluster in enumerate(cluster_list)
                                    for gene in cluster})

        n_annotated_genes_in_cluster = pd.DataFrame(np.array(
                [ [len(go_genes & cluster) for cluster in cluster_list] for go_genes in GO2genes]),
                                                   index   = GO2genes.index,
                                                   columns = range(n_clusters))


        k = n_annotated_genes_in_cluster

        K = pd.concat([global_GO_counter[GO2genes.index]]*n_clusters, axis=1)
        K.columns = k.columns

        n = pd.concat([pd.DataFrame(map(len, cluster_list)).T]*len(GO2genes))
        n.index = k.index

        N = pd.DataFrame(len(G_nx), columns=k.columns, index=k.index)

        assert K.eq(k.sum(axis=1), axis=0).all().all()
        assert N.eq(n.sum(axis=1), axis=0).all().all()

        # scipy has a really messed up nomeclature...
        p_values = pd.DataFrame(1-hypergeom.cdf(k=k.values-1,
                                                M=N.values,
                                                N=n.values,
                                                n=K.values),
                                index=GO2genes.index)
        p_values.to_csv(f"{PVALUE_DIRECTORY}/{RUN}_{n_clusters}.txt")

if __name__ == '__main__':
    from itertools import product
    from multiprocessing import Pool

    print('Number of available cores:', os.cpu_count())

 # Global constants

    DATA_DIRECTORY = "/media/clusterduck123/joe/data"
    RAW_DATA_DIRECTORY = f"{DATA_DIRECTORY}/raw_data"
    YEAST_DIRECTORY = f"{DATA_DIRECTORY}/processed_data/yeast"
    ANNOTATION_DIRECTORY = f"{YEAST_DIRECTORY}/annotations"

 # Input parameters
    with open("input_parameters.py") as f:
        for line in f:
            exec(line.strip())


 # Define necessary directories
    for network, feature, metric, method, aspect in product(networks, features, metrics, methods, aspects):
        PVALUE_DIRECTORY  = f"{YEAST_DIRECTORY}/pvalues/{network}/{feature}/{metric}/{method}/{aspect}"
        if not os.path.exists(PVALUE_DIRECTORY):
            os.makedirs(PVALUE_DIRECTORY)

    with Pool() as p:
        p.starmap(main,product(networks, features, metrics, methods))
