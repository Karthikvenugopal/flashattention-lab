"""Regenerate every published plot from saved data; never run a benchmark."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from attention_lab.plots import plot_results


def main():
    for device in ('cpu', 'mps', 'nvidia-l4'):
        for path in plot_results(ROOT / 'results' / device):
            print(Path(path).relative_to(ROOT))


if __name__ == '__main__':
    main()
