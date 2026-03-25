"""
Phylogenetic pipeline — EXACT method of Li et al. 2024.
Parsnp v1.2 core-genome SNP alignment → FastTree v2.1 GTR+GAMMA → iTOL.
"""
import subprocess, mlflow, logging, pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)
mlflow.set_experiment("kp_pakistan_phylogenetics")


def run_parsnp(assembly_dir: str, output_dir: str,
               reference: str = "data/external/HS11286_reference.fasta",
               threads: int = 16) -> str:
    """
    Parsnp v1.2 core-genome SNP alignment.
    Paper: Parsnp v1.2 from HarvestTools suite.
    -c flag: include ALL genomes regardless of core % (important for diverse datasets).
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    subprocess.run([
        "parsnp",
        "-d", assembly_dir, "-r", reference,
        "-o", output_dir, "-p", str(threads),
        "-c",   # Include all genomes — paper method
        "-x",   # Recombination detection
    ], check=True)

    # Export SNP alignment from .ggr archive
    subprocess.run([
        "harvesttools",
        "-i", f"{output_dir}/parsnp.ggr",
        "-S", f"{output_dir}/parsnp_snps.fa"
    ], check=True)

    logger.info(f"Parsnp SNP alignment: {output_dir}/parsnp_snps.fa")
    return f"{output_dir}/parsnp_snps.fa"


def run_fasttree(alignment: str, output_dir: str) -> str:
    """
    FastTree v2.1 maximum-likelihood tree.
    Paper: GTR+GAMMA model, mid-point rooting applied in iTOL.
    """
    treefile = f"{output_dir}/kp_pakistan_ML_tree.nwk"
    with open(treefile, "w") as outf:
        subprocess.run([
            "FastTree",
            "-gtr",    # GTR model — paper
            "-gamma",  # GAMMA rate variation — paper
            "-nt",     # Nucleotide input
            "-log", f"{output_dir}/fasttree.log",
            alignment
        ], stdout=outf, check=True)
    logger.info(f"FastTree ML tree: {treefile}")
    logger.info("Apply mid-point rooting in iTOL (as in paper Figure 2A)")
    return treefile


def prepare_itol_annotations(kleborate_df: pd.DataFrame, output_dir: str):
    """
    Generate iTOL annotation files — mirrors Figure 2A of Li et al. 2024.
    Paper annotated: ST, KL type, blaCarb class, blaESBL presence,
                     collection year, location (6 concentric rings).
    Upload tree + annotation files to: https://itol.embl.de
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # ST color strip
    st_palette = {
        "ST23":"#d73027","ST11":"#f46d43","ST15":"#fdae61",
        "ST86":"#74add1","ST65":"#4575b4","ST147":"#313695",
        "ST231":"#762a83","ST258":"#1b7837","ST268":"#8c510a"
    }
    with open(f"{output_dir}/itol_ST_strip.txt","w") as f:
        f.write("DATASET_COLORSTRIP\nSEPARATOR TAB\nDATASET_LABEL\tST\nCOLOR\t#000000\nDATA\n")
        for _, row in kleborate_df.iterrows():
            s = str(row.get("strain", row.iloc[0]))
            st = str(row.get("ST","?"))
            f.write(f"{s}\t{st_palette.get(st,'#aaaaaa')}\t{st}\n")

    # hvKP binary
    with open(f"{output_dir}/itol_hvKP.txt","w") as f:
        f.write("DATASET_BINARY\nSEPARATOR TAB\nDATASET_LABEL\thvKP\n"
                "COLOR\t#d73027\nFIELD_LABELS\thvKP\nDATA\n")
        for _, row in kleborate_df.iterrows():
            s = str(row.get("strain", row.iloc[0]))
            f.write(f"{s}\t{int(row.get('hvkp_label',0))}\n")

    # CRhvKP binary
    with open(f"{output_dir}/itol_CRhvKP.txt","w") as f:
        f.write("DATASET_BINARY\nSEPARATOR TAB\nDATASET_LABEL\tCRhvKP\n"
                "COLOR\t#4d004b\nFIELD_LABELS\tCRhvKP\nDATA\n")
        for _, row in kleborate_df.iterrows():
            s = str(row.get("strain", row.iloc[0]))
            f.write(f"{s}\t{int(row.get('crhvkp_label',0))}\n")

    logger.info(f"iTOL annotation files → {output_dir}")
    logger.info("Upload tree + annotations to: https://itol.embl.de")
    logger.info("Apply mid-point rooting, set tree layout as used in Figure 2A")


def run_phylogenetics_pipeline():
    assembly_dir = "data/processed/assemblies_passed_qc"
    out_dir = Path("data/processed/phylogenetics"); out_dir.mkdir(parents=True, exist_ok=True)
    kleb_df = pd.read_csv("data/processed/virulence/kleborate_classified.tsv", sep="\t")

    with mlflow.start_run(run_name="phylogenetics_parsnp_fasttree"):
        mlflow.log_params({
            "alignment_tool": "Parsnp_v1.2",
            "tree_tool": "FastTree_v2.1",
            "tree_model": "GTR+GAMMA",
            "reference": "NC_016845_HS11286",
            "visualization": "iTOL_v5"
        })

        snp_aln = run_parsnp(assembly_dir, str(out_dir/"parsnp"))
        treefile = run_fasttree(snp_aln, str(out_dir))
        prepare_itol_annotations(kleb_df, str(out_dir/"itol_annotations"))

        mlflow.log_artifact(treefile)

    subprocess.run(["dvc", "add", "data/processed/phylogenetics/"], check=True)
    subprocess.run(["git", "add", "-A"], check=True)
    subprocess.run(["git", "commit", "-m", "analysis: Parsnp+FastTree phylogeny complete"], check=True)


if __name__ == "__main__":
    run_phylogenetics_pipeline()
