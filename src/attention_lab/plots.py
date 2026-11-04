"""Plots from saved measurements only; memory metrics retain their semantics."""
import json
import os
from pathlib import Path


def plot_results(source):
    # Colab exports an inline backend absent from isolated environments. Set
    # the file-rendering backend before Matplotlib validates MPLBACKEND at import.
    os.environ['MPLBACKEND'] = 'Agg'
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root = Path(source)
    report = json.loads((root / 'benchmark.json').read_text())
    cases = [c for c in report['cases'] if c['status'] == 'passed']
    if not cases:
        return []
    paths = []
    fields = ['length', 'dimension', 'batch', 'heads', 'query_block_size', 'kv_block_size']
    for metric in ('latency', 'memory'):
        fig, axes = plt.subplots(2, 3, figsize=(14, 8))
        for ax, field in zip(axes.flat, fields):
            subset = [c for c in cases if c['config']['sweep'] in ('anchor', field)]
            for name in sorted({c['implementation'] for c in subset}):
                rows = sorted([c for c in subset if c['implementation'] == name], key=lambda c: c['config'][field])
                if metric == 'latency':
                    ax.errorbar([c['config'][field] for c in rows], [c['median_ms'] for c in rows],
                                yerr=[[c['median_ms']-c['min_ms'] for c in rows], [c['max_ms']-c['median_ms'] for c in rows]],
                                marker='o', label=name, capsize=2)
                elif rows[0]['memory']['bytes'] is not None:
                    ax.plot([c['config'][field] for c in rows], [c['memory']['bytes']/2**20 for c in rows], marker='o', label=name)
            if metric == 'memory':
                rows = sorted([c for c in subset if c['implementation'] == 'standard'], key=lambda c: c['config'][field])
                ax.plot([c['config'][field] for c in rows], [c['analytical_two_dense_score_buffers_bytes']/2**20 for c in rows],
                        '--', label='analytical two dense buffers (not measured peak)')
            ax.set_xlabel(field)
            ax.set_ylabel('median ms (range bars)' if metric == 'latency' else 'MiB')
            if metric == 'latency':
                ax.set_yscale('log')
            ax.legend(fontsize=6)
            ax.grid(alpha=.25)
        memory_kind = cases[0]['memory']['kind']
        title = f"{report['device']} / {report['dtype']} / {metric}"
        if metric == 'memory':
            title += f' / measured: {memory_kind}; dashed: analytical only'
        fig.suptitle(title)
        fig.tight_layout()
        for suffix in ('png', 'svg'):
            path = root / f'{metric}.{suffix}'
            fig.savefig(path)
            paths.append(str(path))
        plt.close(fig)
    return paths
