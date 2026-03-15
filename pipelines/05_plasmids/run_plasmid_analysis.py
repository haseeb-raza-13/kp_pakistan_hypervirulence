"""
Complete plasmid analysis pipeline — Li et al. 2024 exact workflow.
MOB-Suite v3.19 → Mash v2.2 (k=13,s=5000) → Louvain v0.16 →
digIS v1.2 → Prokka v1.13.4 → BRIG v0.95
"""
import subprocess, mlflow, logging, json, shutil, networkx as nx, pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter

try:
    import community as community_louvain
except ImportError:
    subprocess.run(["pip","install","python-louvain==0.16"], check=True)
    import community as community_louvain

logger = logging.getLogger(__name__)
mlflow.set_experiment("kp_pakistan_plasmids")

# Paper's exact Mash parameters
MASH_KMER   = 13      # Paper: k-mer length = 13
MASH_SKETCH = 5000    # Paper: sketch size = 5,000
# Paper's exact Louvain threshold
LOUVAIN_MASH_THRESHOLD = 0.95   # Paper: Mash distance ≤ 0.95
LOUVAIN_MIN_COMMUNITY  = 10     # Paper: min community size = 10


def run_mob_recon(assembly_dir: str, output_dir: str, threads: int = 8):
    """
    MOB-Suite v3.19 plasmid reconstruction and typing.
    Paper: MOB-typer for relaxase/replicon typing and MOB-cluster codes.
    Mobility: conjugative (relaxase+MPF), mobilizable (relaxase/oriT), non-mobilizable.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    plasmid_dir = Path(output_dir)/"plasmid_fastas"
    plasmid_dir.mkdir(parents=True, exist_ok=True)
    all_records = []

    for assembly in Path(assembly_dir).glob("*.fasta"):
        sid = assembly.stem
        sample_out = Path(output_dir)/sid
        try:
            subprocess.run([
                "mob_recon", "--infile", str(assembly),
                "--outdir", str(sample_out),
                "--run_typer", "--num_threads", str(threads), "--force"
            ], check=True, capture_output=True)

            report = sample_out/"contig_report.txt"
            if report.exists():
                df = pd.read_csv(report, sep="\t"); df["sample_id"] = sid
                all_records.append(df)
                for pf in sample_out.glob("plasmid_*.fasta"):
                    shutil.copy2(pf, plasmid_dir/f"{sid}_{pf.name}")
        except subprocess.CalledProcessError as e:
            logger.warning(f"MOB-recon failed {sid}: {e}")

    if all_records:
        full = pd.concat(all_records, ignore_index=True)
        full.to_csv(f"{output_dir}/mob_suite_plasmids.tsv", sep="\t", index=False)
        logger.info(f"MOB-Suite: {len(full)} plasmid contigs across all samples")
        return full, str(plasmid_dir)
    return pd.DataFrame(), str(plasmid_dir)


def run_mash_distances(plasmid_dir: str, output_file: str) -> pd.DataFrame:
    """
    Mash v2.2 pairwise distances.
    Paper: k=13, sketch=5000. Distance=0→identical; distance=1→completely dissimilar.
    Similarity = 1 - Mash_distance.
    """
    fastas = list(Path(plasmid_dir).glob("*.fasta"))
    if len(fastas) < 2:
        logger.warning("Not enough plasmids for Mash"); return pd.DataFrame()

    sketch = str(Path(output_file).parent/"plasmids.msh")
    sketch_cmd = ["mash","sketch","-k",str(MASH_KMER),"-s",str(MASH_SKETCH),
                  "-o",sketch,"-l"]
    proc = subprocess.Popen(sketch_cmd, stdin=subprocess.PIPE)
    proc.communicate(input="\n".join([str(f) for f in fastas]).encode())

    result = subprocess.run(
        ["mash","dist","-p","8",f"{sketch}.msh",f"{sketch}.msh"],
        capture_output=True, text=True
    )

    records = []
    for line in result.stdout.strip().split("\n"):
        parts = line.split("\t")
        if len(parts) >= 3:
            records.append({"query": Path(parts[0]).stem,
                            "reference": Path(parts[1]).stem,
                            "mash_distance": float(parts[2]),
                            "similarity": 1-float(parts[2])})
    df = pd.DataFrame(records)
    df.to_csv(output_file, sep="\t", index=False)
    logger.info(f"Mash: {len(df)} pairwise distances (k={MASH_KMER}, s={MASH_SKETCH})")
    return df


def detect_plasmid_communities(mash_df: pd.DataFrame,
                                output_file: str) -> pd.DataFrame:
    """
    Louvain community detection — paper method exactly.
    python-louvain v0.16, Mash distance threshold = 0.95 (min community = 10).
    PC1=largest, PC2=second, etc. (paper had PC1–PC10 for CRhvKP China data).
    """
    G = nx.Graph()
    for _, row in mash_df.iterrows():
        if row["query"] == row["reference"]: continue
        if row["mash_distance"] <= LOUVAIN_MASH_THRESHOLD:
            G.add_edge(row["query"], row["reference"], weight=row["similarity"])

    partition = community_louvain.best_partition(G, weight="weight")
    sizes = Counter(partition.values())
    valid = {k for k,v in sizes.items() if v >= LOUVAIN_MIN_COMMUNITY}
    pc_map = {c: f"PC{i+1}" for i,c in enumerate(sorted(valid, key=lambda x:-sizes[x]))}

    records = [{"plasmid_id": node,
                "plasmid_community": pc_map.get(comm, "other"),
                "community_size": sizes[comm]}
               for node, comm in partition.items()]
    df = pd.DataFrame(records)
    df.to_csv(output_file, sep="\t", index=False)

    logger.info(f"Louvain communities: {len(valid)} (≥{LOUVAIN_MIN_COMMUNITY} members)")
    for i,c in enumerate(sorted(valid, key=lambda x:-sizes[x])[:10], 1):
        logger.info(f"  PC{i}: {sizes[c]} plasmids")
    return df


def run_digis(assembly_dir: str, output_dir: str):
    """
    digIS v1.2 insertion sequence detection.
    Paper: detected IS26, IS903, ISEc36, IS5075 (IS110-like) mediating plasmid fusions
    and chromosomal integrations of blaKPC-2 and hv genes.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    for fasta in Path(assembly_dir).glob("*.fasta"):
        try:
            subprocess.run([
                "digIS", "--input_sequence", str(fasta),
                "--output_dir", f"{output_dir}/{fasta.stem}",
                "--log_file", f"{output_dir}/{fasta.stem}_digis.log"
            ], check=True, capture_output=True)
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            logger.warning(f"digIS {fasta.stem}: {e}")
    logger.info("digIS IS element detection complete")


def annotate_plasmids_prokka(plasmid_dir: str, output_dir: str):
    """Prokka v1.13.4 plasmid gene annotation."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    for pf in Path(plasmid_dir).glob("*.fasta"):
        pid = pf.stem
        try:
            subprocess.run([
                "prokka", "--outdir", f"{output_dir}/{pid}",
                "--prefix", pid, "--kingdom", "Bacteria",
                "--plasmid", pid, "--rfam", "--threads", "4", "--force", str(pf)
            ], check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            logger.warning(f"Prokka {pid}: {e}")


def find_convergent_plasmids(mob_df: pd.DataFrame, pc_df: pd.DataFrame) -> pd.DataFrame:
    """
    Identify CR-hv convergent plasmids (carry BOTH blaCarb AND hv genes).
    Paper found 40 such plasmids across 5 STs — result of IS-mediated plasmid fusions.
    These are clinically critical: combine CR and hypervirulence in a single replicon.
    """
    cr_genes = ["blaKPC","blaNDM","blaOXA","blaIMP","blaVIM"]
    hv_genes = ["iucA","iucB","iutA","rmpA","rmpA2","iroB","iroN","fyuA","ybtS"]

    def has_gene(row, genes):
        text = " ".join([str(v) for v in row.values]).lower()
        return any(g.lower() in text for g in genes)

    mob_df["has_CR"] = mob_df.apply(has_gene, genes=cr_genes, axis=1)
    mob_df["has_HV"] = mob_df.apply(has_gene, genes=hv_genes, axis=1)
    mob_df["is_convergent"] = mob_df["has_CR"] & mob_df["has_HV"]

    convergent = mob_df[mob_df["is_convergent"]]
    logger.info(f"CR-hv convergent plasmids: {len(convergent)}")
    logger.info("Note: Paper found 40 such plasmids in China — first such report from Pakistan expected")
    return convergent


def brig_circular_alignment_instructions():
    """
    BRIG v0.95 circular alignment instructions for convergent plasmid maps.
    Paper: Figure 5 shows circular BLAST ring images.
    Color scheme from paper: AMR=red, VF=blue, IS=green, Replicons=magenta.
    """
    logger.info("\nBRIG v0.95 — Manual step for circular plasmid maps (Figure 5 equivalent):")
    logger.info("1. Download BRIG: http://brig.sourceforge.net")
    logger.info("2. Use convergent plasmid as center (innermost) sequence")
    logger.info("3. Add hv plasmid (PC1-like) and CR plasmid (PC2-like) as comparison rings")
    logger.info("Color coding: AMR=RED | VF=BLUE | IS elements=GREEN | Replicons=MAGENTA")
    logger.info("Output reveals fusion junctions (homologous recombination or IS-mediated)")


def run_plasmid_pipeline():
    assembly_dir = "data/processed/assemblies_passed_qc"
    out_dir = Path("data/processed/plasmids"); out_dir.mkdir(parents=True, exist_ok=True)

    with mlflow.start_run(run_name="plasmid_analysis"):
        mlflow.log_params({
            "mob_suite_version": "3.19",
            "mash_version": "2.2",
            "mash_kmer": MASH_KMER,
            "mash_sketch": MASH_SKETCH,
            "louvain_version": "python-louvain_0.16",
            "louvain_mash_threshold": LOUVAIN_MASH_THRESHOLD,
            "min_community_size": LOUVAIN_MIN_COMMUNITY,
            "is_detection": "digIS_v1.2",
            "annotation": "Prokka_v1.13.4",
            "circular_maps": "BRIG_v0.95"
        })

        mob_df, plasmid_dir = run_mob_recon(assembly_dir, str(out_dir/"mob_recon"))
        annotate_plasmids_prokka(plasmid_dir, str(out_dir/"annotations"))
        run_digis(assembly_dir, str(out_dir/"insertion_sequences"))
        mash_df = run_mash_distances(plasmid_dir, str(out_dir/"mash_distances.tsv"))

        if not mash_df.empty:
            pc_df = detect_plasmid_communities(mash_df, str(out_dir/"plasmid_communities.tsv"))
            if not mob_df.empty:
                conv = find_convergent_plasmids(mob_df.copy(), pc_df)
                conv.to_csv(str(out_dir/"convergent_plasmids.tsv"), sep="\t", index=False)
                mlflow.log_metrics({
                    "n_plasmids": len(mob_df),
                    "n_communities": pc_df["plasmid_community"].nunique(),
                    "n_convergent_crhv": len(conv)
                })
                mlflow.log_artifact(str(out_dir/"convergent_plasmids.tsv"))

        brig_circular_alignment_instructions()

    subprocess.run(["dvc","add","data/processed/plasmids/"], check=True)
    subprocess.run(["git","add","-A"], check=True)
    subprocess.run(["git","commit","-m","analysis: plasmid pipeline complete (MOB+Mash+Louvain)"], check=True)


if __name__ == "__main__":
    run_plasmid_pipeline()
