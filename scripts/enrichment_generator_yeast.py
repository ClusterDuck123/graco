from goatools import obo_parser
from collections import defaultdict

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

ALPHA = 0.05
LB_GO = 5
UB_GO = 500
MIN_LVL = 0
MAX_LVL = 100

# =============================================================================
#  -------------------------------- FUNCTIONS --------------------------------
# =============================================================================

def get_enrichments(alpha, p_values, cluster_list, correction, gene2GO):
    relevant_p_values = [p_values[str(cluster_idx)][cluster2GO(cluster, gene2GO)]
                             for cluster_idx,cluster in enumerate(cluster_list)]

    sorted_p_values = sorted(p for p_cluster in relevant_p_values
                               for p in p_cluster)
    m = len(sorted_p_values)
    if   correction == 'BY':
        c = np.log(m) + np.euler_gamma + 1/(2*m)
    elif correction == 'BH':
        c = 1
    else:
        print("Correction not known!")
        raise Exception
    for k,P_k in enumerate(sorted_p_values,1):
        if P_k > k/(m*c) * alpha:
            break
    threshold = sorted_p_values[k-2]
    return p_values < threshold


def cluster2GO(cluster, gene2GO):
    return set.union(*(gene2GO.get(gene, set()) for gene in cluster))

def is_annotated_in(gene, GO_set, gene2GO):
    return not gene2GO.get(gene,set()).isdisjoint(GO_set)

# =============================================================================
#  ---------------------------------- MAIN -----------------------------------
# =============================================================================

def main(network, feature, metric, method):
    CLUSTER_DIRECTORY    = f"{YEAST_DIRECTORY}/clusterings/" \
                    f"{network}/{feature}/{metric}/{method}"
    PVALUE_DIRECTORY     = f"{YEAST_DIRECTORY}/pvalues/"     \
                    f"{network}/{feature}/{metric}/{method}/{aspect}"
    ENRICHMENT_DIRECTORY = f"{YEAST_DIRECTORY}/enrichments/" \
                    f"{network}/{feature}/{metric}/{method}/{aspect}/{correction}"

    G_nx = nx.read_edgelist(f"{YEAST_DIRECTORY}/networks/{network}.txt")
    annotation_df = pd.read_csv(f"{ANNOTATION_DIRECTORY}/GO_{aspect}_systematic_SGD.csv")
    annotation_df = annotation_df[annotation_df.Systematic_ID.isin(G_nx)]
    go_dag = obo_parser.GODag(f"{RAW_DATA_DIRECTORY}/go-basic.obo")

    GO_population = {go_id for go_id in set(annotation_df.GO_ID)
                           if (LB_GO <= len(annotation_df[annotation_df.GO_ID == go_id]) <= UB_GO and
                               MIN_LVL <= go_dag[go_id].level <= MAX_LVL)}

    annotation_df = annotation_df[annotation_df.GO_ID.isin(GO_population)]

    # Conversion dictionaries
    GO2genes = pd.Series({go_id: set(genes.Systematic_ID) for go_id, genes in annotation_df.groupby('GO_ID')},
                     name='nb_genes')
    gene2GO = defaultdict(set)
    gene2GO = {gene : set(go_ids.GO_ID) for gene, go_ids in annotation_df.groupby('Systematic_ID')}
    global_GO_counter = GO2genes.apply(len)

    cluster_coverages = np.zeros(MAX_CLUSTERS-MIN_CLUSTERS+1)
    GO_coverages      = np.zeros(MAX_CLUSTERS-MIN_CLUSTERS+1)
    gene_coverages    = np.zeros(MAX_CLUSTERS-MIN_CLUSTERS+1)

    for nr, n_clusters in enumerate(range(MIN_CLUSTERS, MAX_CLUSTERS+1)):
        with open(f"{CLUSTER_DIRECTORY}/{RUN}_{n_clusters}.txt", 'r') as f:
             cluster_list = [set(line.split()) for line in f]
        cluster_df = pd.Series({gene:cluster_idx
                                    for cluster_idx,cluster in enumerate(cluster_list)
                                    for gene in cluster})

        p_values = pd.read_csv(f"{PVALUE_DIRECTORY}/{RUN}_{n_clusters}.txt", index_col=0)

        enrichments = get_enrichments(ALPHA, p_values, cluster_list, correction, gene2GO)
        enrichmet_list = [set(enrichments[i][enrichments[i]].index) for i in enrichments.columns]

        cluster_coverages[nr] = sum(enrichments.any())       / n_clusters
        GO_coverages[nr]      = sum(enrichments.any(axis=1)) / len(GO_population)
        gene_coverages[nr]    = sum(is_annotated_in(gene,
                                                enrichmet_list[cluster_idx],
                                                gene2GO)
                                    for gene, cluster_idx in cluster_df.items()) / len(G_nx)

    specs = f"{RUN}_{MIN_CLUSTERS}-{MAX_CLUSTERS}"
    np.savetxt(f"{ENRICHMENT_DIRECTORY}/{specs}_clusters.csv", cluster_coverages)
    np.savetxt(f"{ENRICHMENT_DIRECTORY}/{specs}_GO-terms.csv", GO_coverages)
    np.savetxt(f"{ENRICHMENT_DIRECTORY}/{specs}_genes.csv"   , gene_coverages)


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
    for network, feature, metric, method, aspect, correction in    \
            product(networks, features, metrics, methods, aspects, corrections):
        ENRICHMENT_DIRECTORY = f"{YEAST_DIRECTORY}/enrichments/"   \
                  f"{network}/{feature}/{metric}/{method}/{aspect}/{correction}"
        if not os.path.exists(ENRICHMENT_DIRECTORY):
            os.makedirs(ENRICHMENT_DIRECTORY)

    with Pool() as p:
        p.starmap(main,product(networks, features, metrics, methods))
