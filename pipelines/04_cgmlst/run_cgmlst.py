"""
cgMLST clustering — mirrors Li et al. 2024 exactly.
chewBBACA v3.3.9 → cgmlst-dists → GrapeTree v1.5.0 MSTv2 → Cytoscape v3.10.2
Cluster threshold: ≤15 allele differences (paper's threshold).
"""
import subprocess, mlflow, logging, networkx as nx, pandas as pd
from pathlib import Path
from collections import Counter

logger = logging.getLogger(__name__)
mlflow.set_experiment("kp_pakistan_cgmlst")

CLUSTER_THRESHOLD = 15   # Paper: 15 allele differences = same clonal cluster
CORE_THRESHOLD = 0.95    # Paper: genes in ≥95% of genomes = core


def download_ridom_scheme(scheme_dir: str = "data/external/cgmlst_scheme"):
    """
    Download 2,358-gene K. pneumoniae cgMLST scheme from RIDOM.
    Paper: "public 2,358-gene typing scheme derived from 14,254 genomes"
    Source: https://www.cgmlst.org (requires registration)
    """
    Path(scheme_dir).mkdir(parents=True, exist_ok=True)
    logger.info("MANUAL STEP REQUIRED:")
    logger.info("1. Register at https://www.cgmlst.org")
    logger.info("2. Search 'Klebsiella pneumoniae' cgMLST scheme")
    logger.info("3. Download the 2,358-loci scheme (from Li et al. 2024)")
    logger.info(f"4. Extract to: {scheme_dir}/")
    logger.info("Alternative: chewBBACA.py DownloadSchema -sp 'Klebsiella pneumoniae'")


def run_chewbbaca(assembly_dir: str, scheme_dir: str,
                   output_dir: str, cpu: int = 8) -> str:
    """
    chewBBACA v3.3.9 allele calling.
    Paper: 2,358-gene RIDOM scheme for K. pneumoniae complex.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    assemblies = list(Path(assembly_dir).glob("*.fasta"))

    genome_list = f"{output_dir}/genome_list.txt"
    with open(genome_list, "w") as f:
        for a in assemblies: f.write(str(a)+"\n")

    subprocess.run([
        "chewBBACA.py", "AlleleCall",
        "-i", genome_list, "-g", scheme_dir,
        "-o", f"{output_dir}/allele_calls",
        "--cpu", str(cpu), "--no-inferred"
    ], check=True)

    return f"{output_dir}/allele_calls/results_alleles.tsv"


def compute_distances(allele_matrix: str, output_file: str) -> pd.DataFrame:
    """
    Core allele distance computation.
    Paper: core genes present in >95% of genomes; cgmlst-dists v0.4.0.
    """
    alleles = pd.read_csv(allele_matrix, sep="\t", index_col=0)
    missing = ["0","INF-","PLOT-","LOTSC","NIPH","NIPHEM","ALM","ASM"]

    present = alleles.applymap(lambda v: not any(str(v).startswith(m) for m in missing))
    core_genes = present.columns[present.mean(axis=0) >= CORE_THRESHOLD]
    logger.info(f"Core genes (≥{100*CORE_THRESHOLD:.0f}%): {len(core_genes)}/{len(alleles.columns)}")

    core = alleles[core_genes]
    core_file = str(Path(output_file).parent/"core_alleles.tsv")
    core.to_csv(core_file, sep="\t")

    with open(output_file, "w") as f:
        subprocess.run(["cgmlst-dists", core_file], stdout=f, check=True)

    return pd.read_csv(output_file, sep="\t", index_col=0)


def assign_clonal_clusters(dist_df: pd.DataFrame,
                            output_file: str) -> pd.DataFrame:
    """
    Assign clonal clusters using ≤15 allele difference threshold.
    Paper: 89% of isolates clustered into 131 clonal groups using this threshold.
    """
    G = nx.Graph()
    samples = dist_df.index.tolist()
    G.add_nodes_from(samples)

    for i, s1 in enumerate(samples):
        for j, s2 in enumerate(samples):
            if j <= i: continue
            try:
                d = float(dist_df.loc[s1, s2])
                if d <= CLUSTER_THRESHOLD:
                    G.add_edge(s1, s2, weight=d)
            except (ValueError, TypeError): continue

    clusters = {}
    for cid, comp in enumerate(
        sorted(nx.connected_components(G), key=len, reverse=True), 1):
        for s in comp: clusters[s] = f"C{cid}"
    for s in samples:
        if s not in clusters: clusters[s] = "singleton"

    df = pd.DataFrame(list(clusters.items()), columns=["sample","clonal_cluster"])
    df.to_csv(output_file, sep="\t", index=False)

    n_clustered = sum(1 for v in clusters.values() if v!="singleton")
    n_clusters = len(set(v for v in clusters.values() if v!="singleton"))
    logger.info(f"Clusters: {n_clusters} | Clustered: {n_clustered}/{len(samples)} "
                f"({100*n_clustered/len(samples):.1f}%)")
    return df


def run_grapetree_mst(allele_matrix: str, output_dir: str):
    """
    GrapeTree v1.5.0 with MSTv2 algorithm.
    Paper: MST construction using GrapeTree MSTv2.
    Cytoscape v3.10.2 used for bipartite network (Figure 3B).
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run([
            "grapetree", "--profile", allele_matrix,
            "--method", "MSTreeV2",   # Paper: MSTv2 algorithm
            "--output", f"{output_dir}/kp_pakistan_MST.nwk"
        ], check=True)
        logger.info(f"GrapeTree MST: {output_dir}/kp_pakistan_MST.nwk")
    except FileNotFoundError:
        logger.info("GrapeTree: install via pip install grapetree")
        logger.info(f"Or upload {allele_matrix} to: https://achtman-lab.github.io/GrapeTree/")
    logger.info("Cytoscape v3.10.2: import MST + metadata for clonal network (Figure 3B equivalent)")


def run_cgmlst_pipeline():
    assembly_dir = "data/processed/assemblies_passed_qc"
    scheme_dir = "data/external/cgmlst_scheme"
    out_dir = Path("data/processed/cgmlst"); out_dir.mkdir(parents=True, exist_ok=True)

    with mlflow.start_run(run_name="cgmlst_chewbbaca"):
        mlflow.log_params({
            "tool": "chewBBACA_v3.3.9",
            "scheme_loci": 2358,
            "scheme_source": "RIDOM_cgmlst.org",
            "cluster_threshold_alleles": CLUSTER_THRESHOLD,
            "core_gene_threshold_pct": int(100*CORE_THRESHOLD),
            "mst_algorithm": "GrapeTree_MSTv2",
            "network_tool": "Cytoscape_v3.10.2"
        })

        allele_matrix = run_chewbbaca(assembly_dir, scheme_dir, str(out_dir))
        dist_df = compute_distances(allele_matrix, str(out_dir/"cgmlst_distances.tsv"))
        cluster_df = assign_clonal_clusters(dist_df, str(out_dir/"clonal_clusters.tsv"))
        run_grapetree_mst(str(out_dir/"core_alleles.tsv"), str(out_dir/"grapetree"))

        n_clustered = (cluster_df["clonal_cluster"]!="singleton").sum()
        mlflow.log_metrics({
            "n_clonal_clusters": cluster_df["clonal_cluster"].nunique() - 1,
            "n_clustered": int(n_clustered),
            "clustering_rate_pct": round(100*n_clustered/len(cluster_df), 2)
        })
        mlflow.log_artifact(str(out_dir/"clonal_clusters.tsv"))

    subprocess.run(["dvc","add","data/processed/cgmlst/"], check=True)
    subprocess.run(["git","add","-A"], check=True)
    subprocess.run(["git","commit","-m","analysis: cgMLST clustering complete"], check=True)


if __name__ == "__main__":
    run_cgmlst_pipeline()
