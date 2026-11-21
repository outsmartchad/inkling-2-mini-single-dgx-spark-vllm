"""Run the 256-token Inkling decode screen on a localhost endpoint."""

import json
from pathlib import Path
import sys

from benchmark import MODEL, api, decode_wave, stream


def main():
    label, filename = sys.argv[1:]
    output = Path(filename)
    if MODEL not in [item["id"] for item in api("/v1/models")["data"]]:
        raise RuntimeError("Wrong model on localhost.")
    _, _, answer = stream("Reply with exactly the word READY.", thinking=False, limit=128, temperature=0)
    if answer.strip().upper().rstrip(".") != "READY":
        raise RuntimeError(f"Unexpected readiness answer: {answer[:100]!r}")
    print("Readiness check passed. Warming up C1 and C5 at the measured shape.", flush=True)
    for streams in (1, 5):
        decode_wave(streams, 0, 256)
    waves = []
    for repetition in range(1, 7):
        order = (1, 5) if repetition % 2 else (5, 1)
        for streams in order:
            row=decode_wave(streams, repetition, 256)
            assert all(r["reasoning_events"]>0 for r in row["requests"])
            waves.append(row)
            data={"model":MODEL,"profile":label,"temperature":0.7,"top_p":1,"top_k":20,
                  "reasoning_effort":"high","output_tokens_per_request":256,"waves":waves}
            temporary=output.with_suffix(".tmp")
            temporary.write_text(json.dumps(data,indent=2,allow_nan=False)+"\n")
            temporary.replace(output)
            print(f"{label} C{streams} {repetition}/6: {row['aggregate_decode_tokens_per_second']:.2f} tok/s",flush=True)


if __name__ == "__main__":
    main()
