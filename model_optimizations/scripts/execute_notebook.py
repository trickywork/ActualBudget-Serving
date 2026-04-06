import argparse
from pathlib import Path

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--cwd", default="/workspace/model_optimizations")
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(in_path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    ep = ExecutePreprocessor(timeout=args.timeout, kernel_name="python3")
    ep.preprocess(nb, {"metadata": {"path": args.cwd}})

    with open(out_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)

    print(f"Executed notebook saved to {out_path}")


if __name__ == "__main__":
    main()
