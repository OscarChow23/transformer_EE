"""Writing results: JSON with a sha256 sidecar, and Markdown tables."""
import hashlib
import json
import os


def write_json(path, obj, sidecar=False):
    """Deterministic JSON (sorted keys, no NaN). With sidecar=True also writes <name>.sha256; returns the digest."""
    text = json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with open(path, "w") as f:
        f.write(text)
    digest = hashlib.sha256(text.encode()).hexdigest()
    if sidecar:
        with open(os.path.splitext(path)[0] + ".sha256", "w") as f:
            f.write(f"{digest}  {os.path.basename(path)}\n")
    return digest


def table(header, rows):
    """Markdown table from a header list and row lists."""
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def stats_table(rows, key):
    """The per-row stats table written by eval_by_sample and compare runs; key = 'sample' or 'run'."""
    out = [f"| target | {key} | window | N | bias | resolution | RMS | abs(r)<5% | <10% | <15% |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for x in rows:
        out.append(f"| {x['target']} | {x[key]} | {x['window']} | {x['n']} | {x['bias']:+.2%} | {x['res']:.2%} | "
                   f"{x['rms']:.2%} | {x['f5']:.1%} | {x['f10']:.1%} | {x['f15']:.1%} |")
    return "\n".join(out) + "\n"
