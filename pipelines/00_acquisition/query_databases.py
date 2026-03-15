"""
Multi-database query for K. pneumoniae Pakistan genomes.
Primary: BV-BRC assemblies (Li et al. 2024 method)
Secondary: NCBI SRA, NCBI Pathogen Detection, ENA (raw reads)
"""
import requests, subprocess, json, time, logging, mlflow
import pandas as pd
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
mlflow.set_experiment("kp_pakistan_data_acquisition")


def query_bvbrc_genomes(country: str = "Pakistan",
                         output: str = "data/metadata/bvbrc_genomes.tsv") -> pd.DataFrame:
    """
    BV-BRC assembled genomes — PRIMARY source following Li et al. 2024.
    Search criteria: organism=K.pneumoniae, country=Pakistan,
                     genome_status=WGS, genome_quality=Good
    """
    url = "https://www.bv-brc.org/api/genome/"
    params = {
        "q": (f'organism_name:"Klebsiella pneumoniae" AND '
              f'isolation_country:{country} AND '
              f'genome_status:WGS AND genome_quality:Good'),
        "fl": ("genome_id,genome_name,strain,taxon_id,isolation_country,"
               "isolation_site,collection_date,host_name,genome_length,"
               "contigs,genome_status,genome_quality,sra_accession,"
               "biosample_accession,genbank_accessions,publication"),
        "rows": 10000, "wt": "json", "sort": "genome_id asc"
    }
    resp = requests.get(url, params=params, headers={"accept": "application/json"}, timeout=120)
    docs = resp.json().get("response", {}).get("docs", []) if resp.status_code == 200 else []
    df = pd.DataFrame(docs)
    df.to_csv(output, sep="\t", index=False)
    logger.info(f"BV-BRC: {len(df)} genomes (WGS, Good quality) from {country}")
    return df


def download_bvbrc_fasta(genome_ids: list, output_dir: str = "data/assemblies/bvbrc/"):
    """Download genome FASTA assemblies from BV-BRC."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    checkpoint = Path(output_dir) / ".downloaded.txt"
    done = set(checkpoint.read_text().splitlines()) if checkpoint.exists() else set()

    for gid in genome_ids:
        if str(gid) in done: continue
        url = (f"https://www.bv-brc.org/api/genome_sequence/"
               f"?eq(genome_id,{gid})&http_accept=application/dna+fasta")
        try:
            resp = requests.get(url, timeout=300, stream=True)
            if resp.status_code == 200 and len(resp.content) > 100:
                with open(f"{output_dir}/{gid}.fasta", "wb") as f:
                    f.write(resp.content)
                with open(checkpoint, "a") as f:
                    f.write(str(gid) + "\n")
                logger.info(f"✓ {gid}")
            else:
                logger.warning(f"Empty: {gid}")
        except Exception as e:
            logger.error(f"✗ {gid}: {e}")
        time.sleep(0.5)


def query_ncbi_sra(output: str = "data/metadata/ncbi_sra.tsv") -> pd.DataFrame:
    """NCBI SRA via EDirect — all Pakistan geo_loc_name variants."""
    query = '"Klebsiella pneumoniae"[Organism] AND "Pakistan"[geo_loc_name]'
    subprocess.run(f"esearch -db sra -query '{query}' | efetch -format runinfo > {output}",
                   shell=True, check=True)
    df = pd.read_csv(output, low_memory=False)
    logger.info(f"NCBI SRA: {len(df)} runs")
    return df


def query_ncbi_pathogen(output: str = "data/metadata/ncbi_pathogen.tsv") -> pd.DataFrame:
    """NCBI Pathogen Detection Browser."""
    base = "https://www.ncbi.nlm.nih.gov/pathogens/api/v1/isolates"
    params = {"q": "isolation_country:Pakistan", "taxon": "Klebsiella_pneumoniae",
               "format": "json", "page_size": 500, "page": 1}
    records = []
    while True:
        resp = requests.get(base, params=params, timeout=60)
        if resp.status_code != 200: break
        batch = resp.json().get("isolates", [])
        if not batch: break
        records.extend(batch)
        params["page"] += 1
        time.sleep(0.35)
    df = pd.json_normalize(records)
    df.to_csv(output, sep="\t", index=False)
    logger.info(f"NCBI Pathogen Detection: {len(df)} isolates")
    return df


def query_ena(output: str = "data/metadata/ena.tsv") -> pd.DataFrame:
    """ENA Portal API — tax_eq(573) = K. pneumoniae."""
    resp = requests.get(
        "https://www.ebi.ac.uk/ena/portal/api/search",
        params={"result": "read_run", "query": 'tax_eq(573) AND country="Pakistan"',
                "fields": "run_accession,sample_accession,country,collection_date,"
                          "instrument_model,library_strategy,read_count,fastq_ftp",
                "format": "tsv", "limit": 0},
        timeout=120
    )
    with open(output, "wb") as f: f.write(resp.content)
    df = pd.read_csv(output, sep="\t")
    logger.info(f"ENA: {len(df)} runs")
    return df


def build_manifests():
    meta = Path("data/metadata")
    bvbrc = pd.read_csv(meta/"bvbrc_genomes.tsv", sep="\t"); bvbrc["source"] = "bvbrc"; bvbrc["data_type"] = "assembly"
    sra   = pd.read_csv(meta/"ncbi_sra.tsv", low_memory=False); sra["source"] = "ncbi_sra"; sra["data_type"] = "reads"
    path  = pd.read_csv(meta/"ncbi_pathogen.tsv", sep="\t"); path["source"] = "pathogen"; path["data_type"] = "assembly"
    ena   = pd.read_csv(meta/"ena.tsv", sep="\t"); ena["source"] = "ena"; ena["data_type"] = "reads"

    asm = pd.concat([bvbrc, path], ignore_index=True)
    reads = pd.concat([sra, ena], ignore_index=True)
    asm.to_csv(meta/"assembly_manifest.tsv", sep="\t", index=False)
    reads.to_csv(meta/"reads_manifest.tsv", sep="\t", index=False)
    logger.info(f"Assemblies: {len(asm)} | Raw reads: {len(reads)}")
    return asm, reads


if __name__ == "__main__":
    Path("data/metadata").mkdir(parents=True, exist_ok=True)
    with mlflow.start_run(run_name="database_queries"):
        bvbrc_df = query_bvbrc_genomes();   mlflow.log_metric("bvbrc_genomes", len(bvbrc_df))
        sra_df   = query_ncbi_sra();        mlflow.log_metric("sra_runs", len(sra_df))
        path_df  = query_ncbi_pathogen();   mlflow.log_metric("pathogen_isolates", len(path_df))
        ena_df   = query_ena();             mlflow.log_metric("ena_runs", len(ena_df))
        asm_m, reads_m = build_manifests()
        mlflow.log_metric("total_assemblies", len(asm_m))
        mlflow.log_metric("total_reads_runs", len(reads_m))
        mlflow.log_artifact("data/metadata/assembly_manifest.tsv")
