"""
Airflow DAG: Parallel data download — BV-BRC assemblies + SRA reads
DVC versioning after each batch.
"""
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from datetime import timedelta
import pandas as pd, subprocess, logging
from pathlib import Path

default_args = {"owner": "meena", "retries": 3,
                "retry_delay": timedelta(minutes=15)}

dag = DAG("kp_pakistan_download", default_args=default_args,
          schedule_interval="@weekly", start_date=days_ago(1), catchup=False,
          tags=["klebsiella", "pakistan"])


def task_queries(**ctx):
    import sys; sys.path.insert(0, "/opt/airflow/pipelines/00_acquisition")
    from query_databases import (query_bvbrc_genomes, query_ncbi_sra,
                                   query_ncbi_pathogen, query_ena, build_manifests)
    for fn in [query_bvbrc_genomes, query_ncbi_sra, query_ncbi_pathogen, query_ena]:
        fn()
    build_manifests()


def task_bvbrc_batch(batch_id, batch_size=100, **ctx):
    import sys; sys.path.insert(0, "/opt/airflow/pipelines/00_acquisition")
    from query_databases import download_bvbrc_fasta
    df = pd.read_csv("data/metadata/assembly_manifest.tsv", sep="\t")
    ids = df[df["source"]=="bvbrc"]["genome_id"].dropna().tolist()
    download_bvbrc_fasta(ids[batch_id*batch_size:(batch_id+1)*batch_size])


def task_sra_batch(batch_id, batch_size=25, **ctx):
    df = pd.read_csv("data/metadata/reads_manifest.tsv", sep="\t")
    acc_col = "Run" if "Run" in df.columns else "run_accession"
    accs = df[acc_col].dropna().tolist()
    batch = accs[batch_id*batch_size:(batch_id+1)*batch_size]
    out = Path(f"data/raw_reads/batch_{batch_id:04d}"); out.mkdir(parents=True, exist_ok=True)
    ckpt = out/".done.txt"; done = set(ckpt.read_text().splitlines()) if ckpt.exists() else set()
    for acc in batch:
        if acc in done: continue
        try:
            subprocess.run(["prefetch", acc, "--output-directory", str(out),
                            "--max-size", "50g"], check=True, timeout=3600)
            subprocess.run(["fasterq-dump", acc, "--outdir", str(out),
                            "--split-files", "--threads", "4"], check=True, timeout=3600)
            subprocess.run(f"gzip {out}/{acc}*.fastq", shell=True)
            sra = out/acc/f"{acc}.sra"
            if sra.exists(): sra.unlink()
            with open(ckpt, "a") as f: f.write(acc+"\n")
        except Exception as e:
            logging.error(f"SRA failed {acc}: {e}")


def task_dvc_version(**ctx):
    for path in ["data/assemblies/", "data/raw_reads/", "data/metadata/"]:
        subprocess.run(["dvc", "add", path], check=True)
    subprocess.run(["git", "add", "-A"], check=True)
    subprocess.run(["git", "commit", "-m", "data: version downloads with DVC"], check=True)


t0 = PythonOperator(task_id="query_databases", python_callable=task_queries, dag=dag)
bvbrc_tasks = [PythonOperator(task_id=f"bvbrc_{i}",
    python_callable=task_bvbrc_batch, op_kwargs={"batch_id": i}, dag=dag) for i in range(10)]
sra_tasks = [PythonOperator(task_id=f"sra_{i}",
    python_callable=task_sra_batch, op_kwargs={"batch_id": i}, dag=dag) for i in range(20)]
t_dvc = PythonOperator(task_id="version_data", python_callable=task_dvc_version, dag=dag)

t0 >> bvbrc_tasks >> t_dvc
t0 >> sra_tasks >> t_dvc
