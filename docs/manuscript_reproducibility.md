# Manuscript reproducibility map

Run all computed manuscript outputs with:

```bash
python scripts/reproduce_manuscript.py \
  --release-root data/CleanCam_v2 \
  --output-root output/manuscript
```

| Manuscript item | Reproducible source |
|---|---|
| Figure 1, collection setup | `docs/manuscript_sources/Figure_1_Data_collection_setup.png` |
| Figure 2, real examples | `scripts/reproduce_manuscript.py` and fixed IDs in `configs/manuscript_examples.json` |
| Figure 3, release composition | `scripts/reproduce_manuscript.py` and `metadata/metadata_real.csv` |
| Figure 4, synthetic examples | `scripts/reproduce_manuscript.py` and fixed IDs in `configs/manuscript_examples.json` |
| Figure 5, PCA and empirical distributions | `scripts/reproduce_manuscript.py`; image-level statistics, PCA loadings, and explained variance are saved with the figure |
| Figure 6, end-to-end workflow | `scripts/reproduce_manuscript.py` |
| Table 1, release contents | `scripts/reproduce_manuscript.py` and the extracted release tree |
| Table 2, dataset composition | `scripts/reproduce_manuscript.py` and master metadata |
| Table 3, label taxonomy | `docs/label_taxonomy.csv` |

`output/manuscript/reproducibility_manifest.json` records the command
configuration, package versions, selected image IDs, and SHA-256 checksum of
each generated file. Low-level statistics are evaluated at the released image
resolution using the definitions implemented in
`cleancam_pipeline/analysis/synthetic.py`.
