# Step 1
conda create -n kp_acquisition -c conda-forge -c bioconda -c defaults \
  python=3.10 \
  sra-tools=3.1.0 \
  entrez-direct \
  requests \
  pandas=2.2.0 \
  biopython=1.83 \
  tqdm \
  dvc \
  mlflow=2.11.0 \
  pip -y && \
conda run -n kp_acquisition pip install ncbi-datasets-pylib apache-airflow==2.8.1


# step 2
# activating environments and verifying packages