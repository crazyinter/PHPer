# Data directory

The large binary data files are intentionally **not committed to Git** to keep the repository small
and within the GitHub file-size limits. They are attached to the versioned GitHub **releases**
(and shipped in the separate reproducibility archives) and can also be regenerated from the scripts
in `src/`. The checksum manifest `results/data_sha256_manifest.json` records the exact bytes.

## `processed/`  (create this folder and place the files here)

| File | Approx. size | Contents | Produced by |
|------|--------------|----------|-------------|
| `cherry_processed.npz` | 8.1 MB | CHERRY FCGR tensors, labels, sequence ids | `build_cherry.py`, `build_cgr3d.py` |
| `cherry_cgr3d.npz` | 188.1 MB | dense 3-D FCGR arrays used by the deep runs | `build_cgr3d.py` |
| `cherry_sketches.npz` | 12.1 MB | MinHash sketches for the similarity split | `build_deephost_sketches.py` |
| `cherry_host_fcgr.npz` | 2.1 MB | host reference FCGRs | `build_deephost_host_fcgr.py` |
| `cherry_protein_features.npz` | 0.6 MB | phage ORF-derived 96-d coding features | `build_host_protein.py` |
| `cherry_host_protein_features.npz` | 0.1 MB | host ORF-derived coding features | `build_host_protein.py` |
| `deephost_processed.npz` | 15.8 MB | DeepHost FCGR tensors, labels, ids | `build_deephost_std.py` |
| `deephost_std_processed.npz` | 12.4 MB | standardized DeepHost tensors | `build_deephost_std.py` |
| `deephost_sketches.npz` | 24.0 MB | DeepHost MinHash sketches | `build_deephost_sketches.py` |
| `deephost_std_sketches.npz` | 18.1 MB | standardized DeepHost sketches | `build_deephost_sketches.py` |
| `deephost_host_fcgr.npz` | 1.1 MB | DeepHost host FCGRs | `build_deephost_host_fcgr.py` |
| `deephost_protein_features.npz` | 0.9 MB | phage ORF-derived coding features | `build_deephost_protein.py` |
| `deephost_host_protein_features.npz` | 0.04 MB | host ORF-derived coding features | `build_deephost_protein.py` |
| `host_protein_cache.pkl` | 0.1 MB | cached per-contig gene calls | `rebuild_host_protein_contigs.py` |

## `fasta/`

- `fasta/deephost_hosts/` - one pinned RefSeq representative per host class (108 `*.fasta`), ten
  placeholders for the empty classes, and `manifest.json` with class id, name, accession, and length.
  Downloaded by `download_deephost_hosts.py` / `final_download_hosts.py`; contigs for whole-genome
  shotgun entries are fetched with `fetch_wgs_contigs.py`.
- `fasta/alt_representatives/` - the seven alternative-strain `*_alt.fasta` files used by
  `representative_sensitivity.py`.

## `raw/`  (not committed)

The upstream project archives (`CHERRY.zip`, `DeepHost.zip`, `PhageCGRNet.zip`) and their unpacked
source trees are third-party materials and are not stored here; obtain them from the public upstream
repositories listed in the top-level README. No third-party source code is needed to run the
BMKMPHI training and analysis scripts.
