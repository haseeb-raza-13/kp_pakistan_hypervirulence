"""
Assembly QC — mirrors Li et al. 2024 EXACTLY.
QUAST v5.2.0 + fastANI v1.33
Reference: K. pneumoniae HS11286 (NC_016845)
Pass thresholds: ANI >= 95%, coverage >= 80%
"""
import subprocess, shutil, mlflow, logging, pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)
mlflow.set_experiment("kp_pakistan_assembly_qc")

REFERENCE = "data/external/HS11286_reference.fasta"
REFERENCE_ACC = "NC_016845"  # K. pneumoniae HS11286 — paper's reference
ANI_THRESHOLD = 95.0         # Paper: ANI >= 95%
COVERAGE_THRESHOLD = 0.80    # Paper: coverage >= 80%


def download_reference():
    """Download K. pneumoniae HS11286 (NC_016845) — paper's reference genome."""
    if Path(REFERENCE).exists(): return
    Path("data/external").mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["efetch", "-db", "nuccore", "-id", REFERENCE_ACC, "-format", "fasta"],
        capture_output=True, text=True, check=True
    )
    with open(REFERENCE, "w") as f: f.write(result.stdout)
    logger.info(f"Reference downloaded: {REFERENCE_ACC}")


def run_quast(assembly_dir: str, output_dir: str):
    """QUAST v5.2.0 assembly quality assessment."""
    assemblies = list(Path(assembly_dir).rglob("*.fasta")) + \
                 list(Path(assembly_dir).rglob("*.fa"))
    if not assemblies:
        logger.error(f"No assemblies found in {assembly_dir}"); return

    subprocess.run([
        "quast.py", "--output-dir", output_dir,
        "--reference", REFERENCE,
        "--threads", "8", "--min-contig", "500", "--no-icarus"
    ] + [str(a) for a in assemblies], check=True)
    logger.info(f"QUAST report: {output_dir}/report.html")


def run_fastani(assembly_dir: str, output_file: str) -> pd.DataFrame:
    """
    fastANI v1.33 species identity verification.
    Paper thresholds: ANI >= 95% AND mapping coverage >= 80%.
    Filters non-K.pneumoniae, K.variicola, K.quasipneumoniae contaminants.
    """
    assemblies = list(Path(assembly_dir).rglob("*.fasta")) + \
                 list(Path(assembly_dir).rglob("*.fa"))

    query_list = "data/metadata/fastani_query_list.txt"
    with open(query_list, "w") as f:
        for a in assemblies: f.write(str(a)+"\n")

    subprocess.run([
        "fastANI", "--ql", query_list, "-r", REFERENCE,
        "-o", output_file, "--threads", "8", "--minFraction", "0.80"
    ], check=True)

    df = pd.read_csv(output_file, sep="\t", header=None,
                     names=["query","reference","ANI","mapped_frags","total_frags"])
    df["coverage"] = df["mapped_frags"] / df["total_frags"]
    # Apply BOTH paper thresholds
    df["pass_qc"] = (df["ANI"] >= ANI_THRESHOLD) & (df["coverage"] >= COVERAGE_THRESHOLD)

    logger.info(f"fastANI pass: {df['pass_qc'].sum()}/{len(df)} "
                f"(ANI≥{ANI_THRESHOLD}% + cov≥{100*COVERAGE_THRESHOLD:.0f}%)")
    return df


def filter_passing_assemblies(fastani_df: pd.DataFrame,
                               output_dir: str = "data/processed/assemblies_passed_qc/") -> list:
    """Copy passing assemblies to QC-passed directory."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    passing = fastani_df[fastani_df["pass_qc"]]
    paths = []
    for _, row in passing.iterrows():
        src = Path(row["query"])
        dest = Path(output_dir) / src.name
        if not dest.exists(): shutil.copy2(src, dest)
        paths.append(str(dest))
    passing[["query","ANI","coverage"]].to_csv(
        "data/metadata/passing_assemblies.tsv", sep="\t", index=False)
    logger.info(f"Passing assemblies: {len(paths)} → {output_dir}")
    return paths


def run_qc_pipeline():
    download_reference()
    qc_dir = "data/processed/qc"
    Path(qc_dir).mkdir(parents=True, exist_ok=True)
    assembly_dir = "data/assemblies"

    with mlflow.start_run(run_name="assembly_qc"):
        mlflow.log_params({
            "reference_genome": REFERENCE_ACC,
            "quast_version": "5.2.0",
            "fastani_version": "1.33",
            "ani_threshold_pct": ANI_THRESHOLD,
            "coverage_threshold": COVERAGE_THRESHOLD
        })

        run_quast(assembly_dir, f"{qc_dir}/quast_report")
        fastani_df = run_fastani(assembly_dir, f"{qc_dir}/fastani_results.tsv")
        passing = filter_passing_assemblies(fastani_df)

        mlflow.log_metrics({
            "total_assemblies": len(fastani_df),
            "passed_qc": len(passing),
            "failed_qc": len(fastani_df) - len(passing),
            "mean_ani_passing": fastani_df[fastani_df["pass_qc"]]["ANI"].mean()
        })
        mlflow.log_artifact(f"{qc_dir}/quast_report/report.html")
        mlflow.log_artifact(f"{qc_dir}/fastani_results.tsv")

    subprocess.run(["dvc", "add", "data/processed/assemblies_passed_qc/"], check=True)
    subprocess.run(["git", "add", "-A"], check=True)
    subprocess.run(["git", "commit", "-m", "data: QC-passed assemblies (QUAST+fastANI)"], check=True)


if __name__ == "__main__":
    run_qc_pipeline()
