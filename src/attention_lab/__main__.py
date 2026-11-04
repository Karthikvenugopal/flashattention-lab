import argparse
import json

from .checks import check
from .devices import doctor
from .tiny import run_tiny


def main():
    parser = argparse.ArgumentParser(description="Learn attention, one small forward pass at a time.")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor", help="Inspect this process's CPU/MPS/CUDA visibility")
    tiny = commands.add_parser("tiny", help="Trace the last query row; verify every row")
    tiny.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")
    tiny.add_argument("--all-rows", action="store_true", help="Also trace rows 0 and 1, after the last row")
    checks = commands.add_parser("check", help="Deterministic errors against independent PyTorch references")
    checks.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")
    checks.add_argument("--json", dest="json_path", help="Save raw correctness results")
    bench = commands.add_parser("benchmark", help="Reproducible small forward sweeps")
    bench.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")
    bench.add_argument("--dtype", choices=["float32", "float16", "bfloat16", "float64"], default="float32")
    bench.add_argument("--profile", choices=["small", "nvidia"], default="small")
    bench.add_argument("--output", required=True)
    bench.add_argument("--warmups", type=int, default=3)
    bench.add_argument("--repeats", type=int, default=10)
    bench.add_argument("--threads", type=int, default=1)
    bench.add_argument("--implementations", nargs="+", choices=["standard", "blockwise", "sdpa", "triton"])
    plots = commands.add_parser("plot", help="Plot a saved benchmark directory")
    plots.add_argument("source")
    args = parser.parse_args()
    try:
        if args.command == "doctor":
            print(json.dumps(doctor(), indent=2))
        elif args.command == "tiny":
            run_tiny(device=args.device, all_rows=args.all_rows)
        elif args.command == "benchmark":
            from .benchmark import run_benchmark
            options = vars(args).copy()
            options.pop("command")
            report = run_benchmark(**options)
            print(f"{report['status']}: {len(report['cases'])} cases; {args.output}")
            if report['status'] != 'passed':
                parser.exit(1, f"{report.get('error', 'Benchmark failures; inspect JSON')}\n")
        elif args.command == "plot":
            from .plots import plot_results
            print("\n".join(plot_results(args.source)))
        else:
            check(device=args.device, json_path=args.json_path)
    except (RuntimeError, ValueError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
