# PhytoRNP LLPS Prediction - 20 Plant Species

## Files

| File | Description |
|------|-------------|
| `predictions_llps_positive.csv` | LLPS positive proteins (LLPS_score > 0.5), 73,476 sequences |
| `llps_tree.nwk` | NJ tree from 5-mer Jaccard distances of LLPS-positive proteins |
| `rna_dep_tree.nwk` | NJ tree from 5-mer Jaccard distances of RNA-dependent proteins |
| `stats.txt` | Per-species statistics (total, LLPS>0.5, RNA_dep>0.5 counts) |

## Species (20)

Amborella_trichopoda, Arabidopsis_thaliana, Brachypodium_distachyon, Chlamydomonas_reinhardtii, Coffea_canephora, Cyanidioschyzon_merolae, Glycine_max, Helianthus_annuus, Klebsormidium_nitens, Marchantia_polymorpha, Medicago_truncatula, Oryza_sativa, Physcomitrium_patens, Populus_trichocarpa, Selaginella_moellendorffii, Solanum_lycopersicum, Sorghum_bicolor, Triticum_aestivum, Vitis_vinifera, Zea_mays

## Summary

| Metric | Value |
|--------|------|
| Total sequences predicted | 762,973 |
| LLPS positive (score > 0.5) | 73,476 (9.6%) |
| RNA dependent (score > 0.5) | 47,574 (6.2%) |
| LLPS score mean | 0.1068 |
| LLPS score median | 0.0062 |

## Methods

- **LLPS prediction**: PhytoRNP_PSLM ResidueLLPSClassifier (ESM2-650M + IUPred3, 6-layer Transformer)
- **RNA dependency**: SVM classifier on 512-dim PhytoRNP pooled features
- **Phylogenetic tree**: Neighbor-Joining on Jaccard distances of 5-mer peptide composition from LLPS-positive proteins

## Prediction Columns

- `protein_name`: FASTA header
- `species`: Species name
- `sequence`: Amino acid sequence
- `LLPS_score`: LLPS probability (sigmoid output, 0-1)
- `RNA_dep_score`: RNA dependency probability (0-1)
