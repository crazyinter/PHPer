import sys
from pathlib import Path

import numpy as np

GH = Path(r"D:\办公\新论文\phage-host interactions-奥林\投稿Frontiers in microbiology被拒\20260713大修\BMKMPHI_github")
sys.path.insert(0, str(GH))
sys.path.insert(0, str(Path(__file__).parent))

from fasta import read_fasta  # noqa: E402
from fasta2CGR import chaos_game_representation, count_kmers, probabilities  # noqa: E402
from fcgr import encode_fcgr  # noqa: E402


def ref_encode(seq, k=6):
    probs = probabilities(seq, count_kmers(seq, k), k)
    return np.asarray(chaos_game_representation(probs, k), dtype=np.float32)


worst = 0.0
recs = list(read_fasta(GH / "example.fasta"))
print("example records:", len(recs))
for name, seq in recs:
    d = np.abs(ref_encode(seq) - encode_fcgr(seq)).max()
    worst = max(worst, d)
    print(f"  {name[:24]:24s} maxdiff={d:.2e}")

# boundary tests with N and mixed-case / invalid letters
for tseq in ["ACGTACGTNNACGT" * 3, "acgtACGT" * 5, "ACGTACGTXACGT" * 4, "ACG", ""]:
    d = np.abs(ref_encode(tseq) - encode_fcgr(tseq)).max()
    worst = max(worst, d)
    print(f"  synthetic len={len(tseq):3d} maxdiff={d:.2e}")

print("ALL MATCH" if worst < 1e-6 else "MISMATCH")
