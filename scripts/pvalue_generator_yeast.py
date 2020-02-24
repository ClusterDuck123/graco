from pyclustering.cluster.kmedoids import kmedoids

import os
import random
import numpy as np
import pandas as pd

"""
Takes network, feature and metric as input and calculates distance matrix.
"""

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

    PPI = nx.read_edgelist(f"{NETWORK_DIRECTORY}/{network}.txt")
    annotation_df = pd.read_csv(f"{ANNOTATION_DIRECTORY}/GO_{aspect}_BioGRID-SGD.csv")
    go_dag = obo_parser.GODag(f"{RAW_DATA_DIRECTORY}/go-basic.obo")

    gene_population = set(PPI.nodes())
    GO_population = set(annotation_df.GO_ID)

    # Conversion dictionaries
    GO2genes = pd.Series({go_id: set(genes.Systematic_ID)
                            for go_id, genes in annotation_df.groupby('GO_ID')},
                         name='nb_genes')

    gene2GO  = {gene : set(go_ids.GO_ID) for gene, go_ids in annotation_df.groupby('Systematic_ID')}
    global_GO_counter = GO2genes.apply(len)

    for nb_clusters in range(MIN_CLUSTERS, MAX_CLUSTERS):
        with open(f"{CLUSTER_DIRECTORY}/{run}_{nb_clusters}.txt", 'r') as f:
             cluster_list = [set(line.split()) for line in f]
        cluster_df = pd.Series({gene:cluster_idx
                                    for cluster_idx,cluster in enumerate(cluster_list)
                                    for gene in cluster})

        nb_annotated_genes_in_cluster = pd.DataFrame(np.array(
                [ [len(go_genes & cluster) for cluster in cluster_list] for go_genes in GO2genes]),
                                                   index   = GO2genes.index,
                                                   columns = range(nb_clusters))


        k = nb_annotated_genes_in_cluster

        K = pd.concat([global_GO_counter[GO2genes.index]]*nb_clusters, axis=1)
        K.columns = k.columns

        n = pd.concat([pd.DataFrame(map(len, cluster_list)).T]*len(GO2genes))
        n.index = k.index

        N = pd.DataFrame(len(PPI), columns=k.columns, index=k.index)

        assert K.eq(k.sum(axis=1), axis=0).all().all()
        assert N.eq(n.sum(axis=1), axis=0).all().all()

        # scipy has a really messed up nomeclature...
        p_values = pd.DataFrame(1-hypergeom.cdf(k=k-1, M=N, N=n, n=K), index=GO2genes.index)
        p_values.to_csv(f"{PVALUE_DIRECTORY}/{run}_{nb_clusters}.txt")
        t2 = time.time()
        print(f'{run}_{nb_clusters}: {t2-t1:.2f}sec', end='\r')

if __name__ == '__main__':
    from itertools import product
    from multiprocessing import Pool

    print('Number of available cores:', os.cpu_count())

 # Global constants
    N_CORES = None

    DATA_DIRECTORY = "/media/clusterduck123/joe/data"
    YEAST_DIRECTORY = f"{DATA_DIRECTORY}/processed_data/yeast"
    MATRIX_DIRECTORY  = f"{YEAST_DIRECTORY}/distance_matrices"

 # Input parameters
    networks = {'systematic_PPI_BioGRID', 'systematic_CoEx_COEXPRESdb'}
    features = {'GCV-O+', 'GDV'}
    metrics  = {'canberra'}
    methods   = {'kmedoid'}


 # Define necessary directories
    for network, feature, metric, method, aspect in product(networks, features, metrics, methods, aspects):
        PVALUE_DIRECTORY  = f"{YEAST_DIRECTORY}/pvalues/{network}/{feature}/{distance}/{method}/{aspect}"
        if not os.path.exists(PVALUE_DIRECTORY):
            os.makedirs(PVALUE_DIRECTORY)

    with Pool(N_CORES) as p:
        p.starmap(main,product(networks, features, metrics, methods))
