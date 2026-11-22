"""Download the pinned public Inkling 2 Mini checkpoint and verify its file sizes."""

import json
import os
import sys
from pathlib import Path

MODEL = "thinkingmachines/Inkling-2-Mini"
REVISION = "ed40e7da625f594fc9c4990f5f66be24873c22d7"


def check_snapshot(snapshot: Path):
    weights = json.loads((snapshot / "model.safetensors.index.json").read_text())["weight_map"]
    if not isinstance(weights, dict) or not weights:
        raise ValueError("Invalid weight index")
    shards = set(weights.values())
    if any(not isinstance(shard, str) or Path(shard).name != shard for shard in shards):
        raise ValueError("Invalid weight index")
    required = shards | {"config.json", "tokenizer.json", "mtp.safetensors", "chat_template.jinja"}
    for name in sorted(required):
        path = snapshot / name
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"Missing or empty checkpoint file: {name}")


def main():
    from huggingface_hub import HfApi, snapshot_download

    root = Path(os.environ["HF_HOME"])
    info = HfApi().model_info(MODEL, revision=REVISION, token=False, files_metadata=True)
    if info.sha != REVISION or info.private or info.gated:
        raise RuntimeError("Unexpected model revision or access policy.")
    snapshot = Path(snapshot_download(MODEL, revision=REVISION, token=False, max_workers=4))
    for entry in info.siblings:
        path = snapshot / entry.rfilename
        if not path.is_file() or (entry.size is not None and path.stat().st_size != entry.size):
            raise RuntimeError(f"Missing or incomplete checkpoint file: {entry.rfilename}")
    marker = root / "verified-inkling-mini.json"
    temporary = marker.with_suffix(".tmp")
    temporary.write_text(json.dumps({"revision": REVISION,
                                     "snapshot": str(snapshot.relative_to(root))}, indent=2) + "\n")
    temporary.replace(marker)
    print(f"Verified {len(info.siblings)} checkpoint files at {REVISION}")


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--check":
        try:
            check_snapshot(Path(sys.argv[2]))
        except (KeyError, OSError, TypeError, ValueError) as error:
            raise SystemExit(f"Checkpoint is incomplete ({error}). Run ./bin/recipe.sh fetch.")
    else:
        main()
