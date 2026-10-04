from pathlib import Path
import hashlib

orig = Path("/data/_backup_to_local/vllm_musa_src/vllm_musa")
inst = Path("/usr/local/lib/python3.10/dist-packages/vllm_musa")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


changed = []
missing = []
extra = []

for op in sorted(orig.rglob("*.py")):
    rel = op.relative_to(orig)
    ip = inst / rel
    if not ip.exists():
        missing.append(str(rel))
    elif sha256(op) != sha256(ip):
        changed.append(str(rel))

for ip in sorted(inst.rglob("*.py")):
    rel = ip.relative_to(inst)
    if not (orig / rel).exists():
        extra.append(str(rel))

print("CHANGED_PY_COUNT", len(changed))
for item in changed:
    print("CHANGED", item)
print("MISSING_PY_COUNT", len(missing))
for item in missing:
    print("MISSING", item)
print("EXTRA_PY_COUNT", len(extra))
for item in extra:
    print("EXTRA", item)
