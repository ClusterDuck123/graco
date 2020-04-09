from itertools import product

import os
import sys
import numpy as np
import pandas as pd
import networkx as nx

# =============================================================================
#  ---------------------------------  CLASSES ---------------------------------
# =============================================================================

class InputParameters():
    RUN   = sys.argv[1]
    RANGE = 10

    ALPHA = 0.05
    MIN_GO = 5
    MAX_GO = 500
    MIN_LVL = 0
    MAX_LVL = np.inf
    CORRECTION = 'BY'

    def __init__(self, network_name, feature, metric, method, aspect):
        self.network_name = network_name
        self.feature = feature
        self.metric  = metric
        self.method  = method
        self.aspect  = aspect

class Paths():
    DATA_DIRECTORY = "/Users/markusyoussef/Desktop/git/supplements/data"
    RAW_DATA_DIRECTORY = f"{DATA_DIRECTORY}/raw_data"
    YEAST_DIRECTORY = f"{DATA_DIRECTORY}/processed_data/yeast"
    NETWORK_DIRECTORY = f"{YEAST_DIRECTORY}/networks"
    ANNOTATION_DIRECTORY = f"{YEAST_DIRECTORY}/annotations"

    def __init__(self, in_parms):
        self.NETWORK_FILE    = f"{self.NETWORK_DIRECTORY}/{in_parms.network_name}.txt"
        self.ANNOTATION_FILE = f"{self.ANNOTATION_DIRECTORY}/GO_{in_parms.aspect}_systematic_SGD.csv"

        network_to_method = f"{in_parms.network_name}/{in_parms.feature}/{in_parms.metric}/{in_parms.method}"
        self.CLUSTER_DIRECTORY    = f"{self.YEAST_DIRECTORY}/clusterings/"   \
                                    f"{network_to_method}"
        self.PVALUE_DIRECTORY     = f"{self.YEAST_DIRECTORY}/pvalues/"       \
                                    f"{network_to_method}/{in_parms.aspect}"
        self.ENRICHMENT_DIRECTORY = f"{self.YEAST_DIRECTORY}/enrichments/"   \
                                    f"{network_to_method}/{in_parms.aspect}/{in_parms.CORRECTION}"

        if not os.path.exists(self.ENRICHMENT_DIRECTORY):
            os.makedirs(self.ENRICHMENT_DIRECTORY)


# =============================================================================
#  -------------------------------- FUNCTIONS --------------------------------
# =============================================================================

def filter_GOterms(annotation_df, all_genes):
    def lvl_filter(min_lvl = InputParameters.MIN_LVL,
                   max_lvl = InputParameters.MAX_LVL):
        return annotation_df.Level.between(min_lvl, max_lvl)
    def GO_filter(geneset,
                  min_GO = InputParameters.MIN_GO,
                  max_GO = InputParameters.MAX_GO):
        return min_GO <= len(geneset) <= max_GO

    annotation_df = annotation_df[annotation_df.Systematic_ID.isin(all_genes)]
    annotation_df = annotation_df[lvl_filter()]
    annotation_df = annotation_df.groupby('GO_ID').filter(GO_filter)

    return annotation_df


def nb_genes_enriched_in_cluster(cluster_enrichment, cluster, gene2GOset):
        enriched_GOterms = set(cluster_enrichment[cluster_enrichment].index)

        def annotated_in_enriched_term(gene):
            return not enriched_GOterms.isdisjoint(gene2GOset[gene])

        return sum(map(annotated_in_enriched_term, cluster))


def get_qvalue_threshold(pvalues, cluster_list, gene2GOset):
    correction = InputParameters.CORRECTION
    alpha      = InputParameters.ALPHA
    # union of gene2GOset across all genes in a cluster
    def cluster_to_GOset(cluster):
        if len(cluster) == 0:
            return list()
        else:
            return set.union(*map(gene2GOset.get, cluster))

    relevant_pvalues = (list(pvalues[idx_cluster][cluster_to_GOset(cluster)])
                             for idx_cluster,cluster in enumerate(cluster_list))

    sorted_pvalues = sorted(p for pvalues_s in relevant_pvalues for p in pvalues_s)

    m = len(sorted_pvalues) # number of tests

    # (pvalue == 1) <==> (gene is not annotated)
    assert (pvalues < 1).values.sum() ==  m

    if   correction == 'Bonferroni': return alpha/m
    elif correction == 'BH': c = 1
    elif correction == 'BY': c = np.log(m) + np.euler_gamma + 1/(2*m)
    else: raise Exception("Correction not known!")

    for k,P_k in enumerate(sorted_pvalues, 1):
        if P_k > k/(m*c) * alpha:
            # one index shift for starting numeration with 1
            # and another one for because of overshoot in the loop
            return sorted_pvalues[k-2]


# =============================================================================
#  ---------------------------------- MAIN -----------------------------------
# =============================================================================

def main(in_parms):
    RUN   = InputParameters.RUN
    RANGE = InputParameters.RANGE

    network_nx = nx.read_edgelist(Paths(in_parms).NETWORK_FILE)
    annotation_df = pd.read_csv(Paths(in_parms).ANNOTATION_FILE)
    annotation_df = filter_GOterms(annotation_df, network_nx.nodes)

    annotated_genes = set(annotation_df.Systematic_ID)
    filtered_GOset  = set(annotation_df.GO_ID)

    gene2GOset = {gene : set(go_ids.GO_ID)        for gene, go_ids in annotation_df.groupby('Systematic_ID')}
    GO2geneset = {go_id: set(genes.Systematic_ID) for go_id, genes in annotation_df.groupby('GO_ID')}

    GO2geneset_s = pd.Series(GO2geneset).sort_values()

    cluster_coverages = np.zeros(2*RANGE+1)
    GO_coverages      = np.zeros(2*RANGE+1)
    gene_coverages    = np.zeros(2*RANGE+1)

    # ----------------------- this is where the fun starts ------------------------
    N = len(network_nx)
    M = int(np.sqrt(N/2))

    for count, n_clusters in enumerate(range(M-RANGE, M+RANGE+1)):
        with open(f"{Paths(in_parms).CLUSTER_DIRECTORY}/{RUN}_{n_clusters}.txt", 'r') as f:
                     cluster_list = [set(line.split()) for line in f]

    # Keep only annotated genes in cluster
        annotated_cluster_list = [geneset & annotated_genes for geneset in cluster_list]

        pvalues = pd.read_csv(f"{Paths(in_parms).PVALUE_DIRECTORY}/{RUN}_{n_clusters}.txt",
                              index_col = 0).loc[filtered_GOset]
        pvalues.columns = map(int, pvalues.columns)

        qvalue_th   = get_qvalue_threshold(pvalues, annotated_cluster_list, gene2GOset)
        enrichments = pvalues < qvalue_th

        cluster_coverages[count] = sum(enrichments.any(axis=0)) / n_clusters
        GO_coverages[     count] = sum(enrichments.any(axis=1)) / len(filtered_GOset)
        gene_coverages[   count] = sum(nb_genes_enriched_in_cluster(enrichment_s,
                                                                    annotated_cluster_list[idx],
                                                                    gene2GOset)
              for idx, enrichment_s in enrichments.iteritems()) / len(annotated_genes)

    specs = f"{RUN}_{M-RANGE}-{M+RANGE}"
    np.savetxt(f"{Paths(in_parms).ENRICHMENT_DIRECTORY}/{specs}_clusters.csv", cluster_coverages)
    np.savetxt(f"{Paths(in_parms).ENRICHMENT_DIRECTORY}/{specs}_GOterms.csv", GO_coverages)
    np.savetxt(f"{Paths(in_parms).ENRICHMENT_DIRECTORY}/{specs}_genes.csv"   , gene_coverages)


# =============================================================================
#  ----------------------------------- INIT -----------------------------------
# =============================================================================


network_names = {'systematic_PPI_BioGRID', 'GI_Constanzo2016',
                 'systematic_CoEx_COEXPRESdb'}
features = {'GDV'}
metrics  = {'mahalanobis', 'GDV_similarity', 'seuclidean', 'hellinger',
            'cityblock', 'euclidean', 'chebyshev', 'canberra', 'cosine',
            'correlation', 'braycurtis', 'sqeuclidean'}
methods  = {'kmedoid'}
aspects  = {'CC'}

loop_product = product(network_names, features, metrics, methods, aspects)
for network_name, feature, metric, method, aspect in loop_product:
    print(network_name, feature, metric, aspect)
    main(InputParameters(network_name, feature, metric, method, aspect))
