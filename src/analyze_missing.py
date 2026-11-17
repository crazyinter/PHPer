import re
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from dataset import build_labeled, load_host_info  # noqa: E402

matched, missing = build_labeled()
# standard GenBank/RefSeq nucleotide accessions, e.g. NC_000000, AB123456, JXSL01
ncbi_pat = re.compile(r"^[A-Z]{1,4}_?\d{2,}(\.\d+)?$")
std = [m for m in missing if ncbi_pat.match(m)]
nonstd = [m for m in missing if not ncbi_pat.match(m)]
print("missing total:", len(missing))
print("  standard NCBI-style accessions (recoverable):", len(std))
print("  non-standard PhageDB names:", len(nonstd))
print("  standard sample:", std[:20])
print("  non-standard sample:", nonstd[:20])

# how many host_info rows are standard vs not
info = load_host_info()
all_std = [r for r in info if ncbi_pat.match(r["acc"])]
print("\nhost_info standard NCBI acc:", len(all_std), "of", len(info))
