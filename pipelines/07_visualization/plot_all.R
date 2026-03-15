# ================================================================
# R Visualization — Adapted from Li et al. 2024 Figure scheme
# Fig 1: Geographic map + virulence year bars + resistance year bars
# Fig 2: ST frequency, K-locus, virulence score by ST, AMR by ST
# Fig 3: Clonal cluster heatmap + transmission network input
# Fig 4: Virulence gene heatmap
# Fig 5: gggenomes gene synteny (chromosomal integrations)
# ================================================================

library(ggplot2); library(dplyr); library(tidyr); library(ggtree)
library(ComplexHeatmap); library(circlize); library(ggpubr)
library(viridis); library(patchwork); library(ape); library(phangorn)
library(gggenomes)           # Paper Figure 6 — gene synteny maps
library(scales); library(RColorBrewer)

# Optional geographic map (requires rnaturalearth)
if (requireNamespace("rnaturalearth", quietly=TRUE)) {
  library(rnaturalearth); library(sf)
}

theme_kp <- theme_classic(base_size=12) +
  theme(text=element_text(family="Helvetica"),
        plot.title=element_text(face="bold",size=14),
        axis.text=element_text(color="black"),
        panel.grid.major.y=element_line(color="grey90",linewidth=0.3))

# Paper color scheme for virulence classes
VIR_COLS <- c("3"="#a6d96a","4"="#f46d43","5"="#d73027",
               "0"="#4575b4","1"="#74add1","2"="#abd9e9")

kleb <- read.delim("data/processed/virulence/kleborate_classified.tsv", stringsAsFactors=FALSE)
clust <- read.delim("data/processed/cgmlst/clonal_clusters.tsv", stringsAsFactors=FALSE)
kleb <- left_join(kleb, clust, by=c("strain"="sample"))

# ── Fig 1B: Virulence score by year ─────────────────────────────────────────
if ("collection_date" %in% colnames(kleb) && "virulence_score" %in% colnames(kleb)) {
  vyr <- kleb %>%
    mutate(year=as.integer(substr(collection_date,1,4))) %>%
    filter(!is.na(year), !is.na(virulence_score), year >= 2005) %>%
    count(year, virulence_score)

  p1b <- ggplot(vyr, aes(x=factor(year), y=n, fill=factor(virulence_score))) +
    geom_col(color="white", linewidth=0.2) +
    scale_fill_manual(values=VIR_COLS, name="Virulence Score") +
    labs(title="Virulence Score by Year — K. pneumoniae Pakistan",
         x="Collection Year", y="Number of Isolates") +
    theme_kp + theme(axis.text.x=element_text(angle=45,hjust=1))
  ggsave("reports/figures/Fig1B_virulence_by_year.pdf", p1b, width=10, height=6, dpi=300)
}

# ── Fig 2B: ST frequency by year ────────────────────────────────────────────
top_sts <- kleb %>% count(ST, sort=TRUE) %>% filter(n>=3) %>% head(10) %>% pull(ST)

if ("collection_date" %in% colnames(kleb)) {
  styr <- kleb %>%
    filter(ST %in% top_sts) %>%
    mutate(year=as.integer(substr(collection_date,1,4))) %>%
    filter(!is.na(year)) %>%
    count(year, ST)

  p2b <- ggplot(styr, aes(x=factor(year), y=n, fill=factor(ST))) +
    geom_col(position="stack", color="white", linewidth=0.2) +
    scale_fill_brewer(palette="Set1", name="Sequence Type") +
    labs(title="Sequence Type Frequency by Year", x="Year", y="Count") +
    theme_kp + theme(axis.text.x=element_text(angle=45,hjust=1))
  ggsave("reports/figures/Fig2B_ST_by_year.pdf", p2b, width=10, height=6, dpi=300)
}

# ── Fig 2D: Virulence score by ST (mirrors paper Figure 2D) ─────────────────
if ("virulence_score" %in% colnames(kleb)) {
  sv <- kleb %>%
    filter(ST %in% top_sts, !is.na(virulence_score)) %>%
    count(ST, virulence_score) %>%
    group_by(ST) %>% mutate(freq=n/sum(n))

  p2d <- ggplot(sv, aes(x=factor(ST), y=freq, fill=factor(virulence_score))) +
    geom_col(position="stack", color="white") +
    scale_fill_manual(values=VIR_COLS, name="Virulence Score") +
    scale_y_continuous(labels=percent) +
    labs(title="Virulence Score by ST", x="ST", y="Frequency") +
    theme_kp
  ggsave("reports/figures/Fig2D_vir_by_ST.pdf", p2d, width=8, height=6, dpi=300)
}

# ── Fig 2E: AMR classes by ST ────────────────────────────────────────────────
amr_cols <- grep("^bla|^qnr|^aac|^aph|^mcr", colnames(kleb), value=TRUE, ignore.case=TRUE)
if (length(amr_cols) > 0) {
  kleb$n_amr <- rowSums(!is.na(kleb[,amr_cols]) & kleb[,amr_cols]!="" & kleb[,amr_cols]!="-")
  sa <- kleb %>%
    filter(ST %in% top_sts, !is.na(n_amr)) %>%
    mutate(amr_bin=cut(n_amr,breaks=c(-1,0,2,4,6,Inf),labels=c("0","1-2","3-4","5-6","7+"))) %>%
    count(ST, amr_bin) %>% group_by(ST) %>% mutate(freq=n/sum(n))
  p2e <- ggplot(sa, aes(x=factor(ST), y=freq, fill=amr_bin)) +
    geom_col(position="stack", color="white") +
    scale_fill_brewer(palette="YlOrRd", name="AMR Classes") +
    scale_y_continuous(labels=percent) +
    labs(title="AMR Classes by Sequence Type", x="ST", y="Frequency") +
    theme_kp
  ggsave("reports/figures/Fig2E_AMR_by_ST.pdf", p2e, width=8, height=6, dpi=300)
}

# ── Virulence gene heatmap ──────────────────────────────────────────────────
hv_gene_cols <- c("iucA","iucB","iucC","iucD","iutA",
                   "iroB","iroN","fyuA","ybtS","irp1",
                   "rmpA","rmpA2","clbA","mrkD","peg344")
avail <- intersect(hv_gene_cols, colnames(kleb))
if (length(avail) >= 3) {
  mat <- sapply(kleb[,avail], function(x) as.integer(!is.na(x) & x!="" & x!="-"))
  rownames(mat) <- kleb$strain

  anno <- rowAnnotation(
    vir_score=kleb$virulence_score,
    hvKP=factor(kleb$is_hvKP),
    CRhvKP=factor(kleb$is_CRhvKP),
    col=list(vir_score=colorRamp2(c(0,2,4,5),c("#313695","#74add1","#f46d43","#a50026")),
             hvKP=c("FALSE"="#4575b4","TRUE"="#d73027"),
             CRhvKP=c("FALSE"="#f0f0f0","TRUE"="#b2182b")))

  pdf("reports/figures/Fig4_virulence_gene_heatmap.pdf", width=16, height=10)
  draw(Heatmap(t(mat), name="Present",
               col=c("0"="#f0f0f0","1"="#d73027"),
               left_annotation=anno,
               show_column_names=FALSE,
               cluster_rows=FALSE, cluster_columns=TRUE,
               column_split=kleb$is_CRhvKP,
               column_title="K. pneumoniae Pakistan — Hypervirulence Gene Matrix",
               row_names_gp=gpar(fontsize=10), border=TRUE))
  dev.off()
}

# ── Phylogenetic tree visualization ─────────────────────────────────────────
tree_file <- "data/processed/phylogenetics/kp_pakistan_ML_tree.nwk"
if (file.exists(tree_file)) {
  tree <- read.tree(tree_file)
  tree <- midpoint(tree)  # Paper: mid-point rooting

  p_tree <- ggtree(tree, layout="circular", size=0.2) %<+% kleb +
    geom_tippoint(aes(color=factor(is_hvKP)), size=1.2, alpha=0.8) +
    scale_color_manual(values=c("FALSE"="#4575b4","TRUE"="#d73027"),
                       name="hvKP", na.value="grey70") +
    labs(title="Core-Genome Phylogeny — K. pneumoniae Pakistan",
         subtitle="Parsnp v1.2 SNP alignment | FastTree v2.1 GTR+GAMMA | Mid-point rooted") +
    theme_tree2() + theme(legend.position="right")

  # Note: For full iTOL-style visualization (paper Figure 2A), upload tree to itol.embl.de
  # with the annotation files from reports/itol/ directory
  ggsave("reports/figures/Fig2A_phylogenetic_tree_R.pdf",
         p_tree, width=14, height=14, dpi=300)
  message("For publication-quality iTOL version: upload tree + annotations to https://itol.embl.de")
}

# ── gggenomes: Gene synteny maps (Figure 6 equivalent) ──────────────────────
# For chromosomal hv gene region comparisons across Pakistani STs
# Requires GFF annotations from Prokka for the chromosomal regions of interest
# Example after getting GFF files:
# gggenomes(seqs=seqs_df, genes=genes_df, links=links_df) %>%
#   geom_seq() %>% geom_gene(aes(fill=feat)) %>%
#   geom_link() %>% scale_fill_brewer(palette="Set1")
# Refer to: https://github.com/thackl/gggenomes

message("\n✅ All figures saved to reports/figures/")
message("📁 iTOL annotations: data/processed/phylogenetics/itol_annotations/")
message("   → Upload to: https://itol.embl.de")
message("📁 GrapeTree MST: data/processed/cgmlst/grapetree/")
message("   → View at: https://achtman-lab.github.io/GrapeTree/")
message("📁 Cytoscape input: data/processed/cgmlst/clonal_clusters.tsv")
message("   → Open in Cytoscape v3.10.2 for bipartite network (Figure 3B)")
