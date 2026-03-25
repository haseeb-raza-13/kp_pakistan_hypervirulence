"""
Kleborate v3.0.8 genotyping — EXACT version from Li et al. 2024.
ABRicate v1.0.1 at 90%/90% threshold — EXACT from paper.
Classification: hvKP = iucA + rmpA/rmpA2; CRKP = any blaCarb.
"""
import subprocess, mlflow, logging, pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)
mlflow.set_experiment("kp_pakistan_genotyping")

# All carbapenemase genes tracked — from Li et al. 2024
BLACARB = ["blaKPC","blaNDM","blaOXA-48","blaIMP","blaVIM","blaOXA-23","blaOXA-58"]


def run_kleborate(assembly_dir: str, output_file: str) -> pd.DataFrame:
    """
    Kleborate v3.0.8 with --all --pneumo flags.
    Detects: MLST, virulence (ybt/clb/iro/iuc/rmpA/rmpA2), capsule, AMR.
    """
    assemblies = list(Path(assembly_dir).glob("*.fasta")) + \
                 list(Path(assembly_dir).glob("*.fa"))

    cmd = ["kleborate", "--all", "--pneumo",
           "--assemblies"] + [str(a) for a in assemblies] + ["--output", output_file]
    subprocess.run(cmd, check=True)
    df = pd.read_csv(output_file, sep="\t")
    logger.info(f"Kleborate v3.0.8: {len(df)} genomes typed")
    return df


def classify_isolates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply Li et al. 2024 molecular definitions.
    hvKP: iucA AND (rmpA OR rmpA2) present.
    CRKP: any blaCarb gene present.
    CRhvKP: both.
    """
    def gene_present(row, gene_name):
        for col in df.columns:
            if gene_name.lower().replace("-","") in col.lower().replace("-",""):
                val = str(row.get(col,""))
                if val not in ["","nan","-","none","0"]: return True
        return False

    df["is_hvKP"] = df.apply(
        lambda r: gene_present(r, "iucA") and
                  (gene_present(r, "rmpA") or gene_present(r, "rmpA2")), axis=1)
    df["is_CRKP"] = df.apply(
        lambda r: any(gene_present(r, g) for g in BLACARB), axis=1)
    df["is_CRhvKP"] = df["is_hvKP"] & df["is_CRKP"]
    df["hvkp_label"] = df["is_hvKP"].astype(int)
    df["crhvkp_label"] = df["is_CRhvKP"].astype(int)

    n = len(df)
    logger.info(f"Total: {n} | hvKP: {df['is_hvKP'].sum()} ({100*df['is_hvKP'].mean():.1f}%) | "
                f"CRKP: {df['is_CRKP'].sum()} | CRhvKP: {df['is_CRhvKP'].sum()}")
    return df


def run_abricate(assembly_dir: str, output_dir: str):
    """
    ABRicate v1.0.1 against VFDB and CARD databases.
    Paper threshold: 90% identity / 90% coverage (both required).
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    for db in ["vfdb", "card"]:  # Paper used VFDB and CARD
        out = f"{output_dir}/abricate_{db}.tsv"
        result = subprocess.run([
            "abricate", "--db", db,
            "--minid", "90",   # Paper: 90% sequence identity
            "--mincov", "90",  # Paper: 90% coverage
            "--threads", "4", "--nopath", assembly_dir
        ], capture_output=True, text=True)
        with open(out, "w") as f: f.write(result.stdout)
        # Summary
        sum_result = subprocess.run(["abricate","--summary", out],
                                     capture_output=True, text=True)
        with open(f"{output_dir}/abricate_{db}_summary.tsv", "w") as f:
            f.write(sum_result.stdout)
    logger.info("ABRicate (VFDB + CARD) complete, 90%/90% thresholds")


def run_genotyping_pipeline():
    assembly_dir = "data/processed/assemblies_passed_qc"
    out_dir = Path("data/processed/virulence"); out_dir.mkdir(parents=True, exist_ok=True)

    with mlflow.start_run(run_name="kleborate_genotyping"):
        mlflow.log_params({
            "kleborate_version": "3.0.8",
            "abricate_version": "1.0.1",
            "abricate_min_identity_pct": 90,
            "abricate_min_coverage_pct": 90,
            "hvkp_definition": "iucA + rmpA/rmpA2",
            "crkp_definition": "any_blaCarb"
        })

        kleb_df = run_kleborate(assembly_dir, str(out_dir/"kleborate_raw.tsv"))
        kleb_df = classify_isolates(kleb_df)
        kleb_df.to_csv(str(out_dir/"kleborate_classified.tsv"), sep="\t", index=False)
        run_abricate(assembly_dir, str(out_dir))

        mlflow.log_metrics({
            "total_isolates": len(kleb_df),
            "hvkp_count": int(kleb_df["is_hvKP"].sum()),
            "crkp_count": int(kleb_df["is_CRKP"].sum()),
            "crhvkp_count": int(kleb_df["is_CRhvKP"].sum()),
            "hvkp_prevalence_pct": round(100*kleb_df["is_hvKP"].mean(), 2)
        })
        mlflow.log_artifact(str(out_dir/"kleborate_classified.tsv"))

    subprocess.run(["dvc", "add", "data/processed/virulence/"], check=True)
    subprocess.run(["git", "add", "-A"], check=True)
    subprocess.run(["git", "commit", "-m", "analysis: Kleborate v3.0.8 genotyping complete"], check=True)


if __name__ == "__main__":
    run_genotyping_pipeline()
