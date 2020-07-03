from itertools import product
from scipy.stats import hypergeom

import os
import sys
import numpy as np
import pandas as pd
import networkx as nx

# =============================================================================
#  ---------------------------------  CLASSES ---------------------------------
# =============================================================================

class InputParameters():
    RUN = sys.argv[1]
    RANGE = 10

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
        self.CLUSTER_DIRECTORY = f"{self.YEAST_DIRECTORY}/clusterings/"  \
                                 f"{network_to_method}"
        self.PVALUE_DIRECTORY  = f"{self.YEAST_DIRECTORY}/pvalues/"      \
                                 f"{network_to_method}/{in_parms.aspect}"

        if not os.path.exists(self.PVALUE_DIRECTORY):
            os.makedirs(self.PVALUE_DIRECTORY)


# =============================================================================
#  -------------------------------- FUNCTIONS ---------------------------------
# =============================================================================

def get_pvalues(cluster_list, annotation, gene_population):
    """
    Takes a liks of clusters and an annotation file and returns
    a dataframe of p-values for each cluster and each annotation term
    """

    n_clusters = len(cluster_list)

# ---------------------------- population size, M -----------------------------
    nb_of_annoteted_genes = pd.DataFrame(len(gene_population),
                                         index   = annotation.index,
                                         columns = range(n_clusters))

# ---------- number of draws (i.e. quantity drawn in each trial), N -----------
    n_GOterm_copies_of_cluster_sizes = iter([pd.Series(map(len, cluster_list))]*len(annotation))
    size_of_clusters = pd.concat(n_GOterm_copies_of_cluster_sizes, axis=1).T
    size_of_clusters.index = annotation.index

    # sum of |(annotated) genes in cluster| across all clusters
    # == |overall (annotated) genes|
    assert (size_of_clusters.sum(axis=1) == len(gene_population)).all()

# -------------- number of success states in the population, n ----------------
    n_cluster_copies_of_annotation_counts = iter([annotation.apply(len)]*n_clusters)
    nb_annotated_genes = pd.concat(n_cluster_copies_of_annotation_counts, axis=1)
    nb_annotated_genes.columns = range(n_clusters)

# --------------------- number of observed successes, k -----------------------
    gene_count_of_intersections = (
                pd.Series([len(annotated_genes & gene_set) for gene_set in cluster_list])
                                     for annotated_genes in annotation)
    nb_annotated_genes_in_cluster = pd.concat(gene_count_of_intersections, axis=1).T
    nb_annotated_genes_in_cluster.index   = annotation.index
    nb_annotated_genes_in_cluster.columns = range(n_clusters)

    # sum of |annotated genes per GO-term in cluster| across all clusters
    # == |annotated genes per GO-term|
    assert (nb_annotated_genes_in_cluster.sum(axis=1) == annotation.apply(len)).all()

# ------------ all of this just to execute a single scipy function -------------
    pvalues = pd.DataFrame(1-hypergeom.cdf(M = nb_of_annoteted_genes.values,
                                        N = size_of_clusters.values,
                                        n = nb_annotated_genes.values,
                                        k = nb_annotated_genes_in_cluster.values-1),
                        index=annotation.index)

    # set pvalues of unannotated cluster in GOterm to nan for assertion checks
    pvalues[nb_annotated_genes_in_cluster == 0] = np.nan
    return pvalues


def assert_nan_values(pvalues, cluster_list, gene2GOset):
    for cluster_idx in pvalues.columns:
        if len(cluster_list[cluster_idx]) == 0:
            assert (pvalues[cluster_idx].isna()).all()
        else:
            GOterms_in_cluster = set.union(*map(gene2GOset.get, cluster_list[cluster_idx]))
            for GOterm in pvalues.index:
                if not GOterm in GOterms_in_cluster:
                    assert np.isnan(pvalues[cluster_idx][GOterm])

# =============================================================================
#  ----------------------------------- MAIN -----------------------------------
# =============================================================================

def main(in_parms):
    RUN   = InputParameters.RUN
    RANGE = InputParameters.RANGE

    network_nx = nx.read_edgelist(Paths(in_parms).NETWORK_FILE)
    annotation_df = pd.read_csv(Paths(in_parms).ANNOTATION_FILE)
    annotation_df = annotation_df[annotation_df.Systematic_ID.isin(network_nx)]

    annotated_geneset = set(annotation_df.Systematic_ID)

    GO2geneset = {go_id: set(genes.Systematic_ID) for go_id, genes in annotation_df.groupby('GO_ID')}
    gene2GOset = {gene : set(go_ids.GO_ID) for gene, go_ids in annotation_df.groupby('Systematic_ID')}

    GO2geneset_s = pd.Series(GO2geneset).sort_index()

# ------------ unrelated statistics: number of un-annotated genes -------------
    nb_unannotated_genes = len(network_nx)-len(annotated_geneset)
    print(f"Network has {len(network_nx)} genes, of which {nb_unannotated_genes} "
          f"({100*nb_unannotated_genes/len(network_nx):.2f}%) are un-annotated.")

# ----------------------- this is where the fun starts ------------------------
    N = len(network_nx)
    M = int(np.sqrt(N/2))

    for n_clusters in range(M-RANGE, M+RANGE+1):
        with open(f"{Paths(in_parms).CLUSTER_DIRECTORY}/{RUN}_{n_clusters}.txt", 'r') as f:
                     cluster_list = [set(line.split()) for line in f]

        # keep only annotated genes in cluster
        annotated_cluster_list = [gene_set & annotated_geneset for gene_set in cluster_list]

        pvalues = get_pvalues(cluster_list    = annotated_cluster_list,
                              annotation      = GO2geneset_s,
                              gene_population = annotated_geneset)

        # assert that un-annotated GO-terms have a p-value of nan
        assert_nan_values(pvalues, annotated_cluster_list, gene2GOset)

        pvalues.to_csv(f"{Paths(in_parms).PVALUE_DIRECTORY}/{RUN}_{n_clusters}.txt")

# =============================================================================
#  ----------------------------------- INIT -----------------------------------
# =============================================================================

network_names = {'GI_Constanzo2016',
                 'systematic_PPI_BioGRID',
                 'systematic_CoEx_COEXPRESdb'
                 }
features = {'GDV'} #'GCV-O+', 'GCV-all', 'triangle'}
metrics  = {'GDV_similarity'
            #'mahalanobis', 'seuclidean', 'hellinger',
            #'cityblock', 'euclidean',
            #'chebyshev',
            #'js_divergence'
            #'canberra', 'cosine',
            #'correlation', 'braycurtis', 'sqeuclidean'
            }
methods  = {'kmedoid'}
aspects  = {'MF','BP','CC'}

loop_product = product(network_names, features, metrics, methods, aspects)
for network_name, feature, metric, method, aspect in loop_product:
    print(network_name, feature, metric, aspect)
    main(InputParameters(network_name, feature, metric, method, aspect))
