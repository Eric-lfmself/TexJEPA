"""Plot our manuscript's aggregate results without loading images or models."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


def plot_paper_results(source: Path, output: Path) -> list[Path]:
    """Export two summary figures from the reported table values."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    source = Path(source)
    payload = json.loads(source.read_text())
    tables = {table['id']: table for table in payload['tables']}
    for name in ('table_iv', 'table_v', 'table_viii', 'table_ix', 'table_x', 'table_xi'):
        if name not in tables:
            raise ValueError(f'Missing source table: {name}')
    sigmas = [0.05, 0.10, 0.20, 0.30]
    columns = ['sigma_0_05', 'sigma_0_10', 'sigma_0_20', 'sigma_0_30']
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    paths = [output / f'{name}.{suffix}' for name in ('main_results', 'post_training_results')
             for suffix in ('png', 'pdf', 'svg')]
    manifest_path = output / 'manifest.json'
    if any(path.exists() for path in paths + [manifest_path]):
        raise FileExistsError('Choose a new figure directory; existing figures are preserved.')
    style = {'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.titlesize': 11,
             'axes.labelsize': 9, 'legend.fontsize': 8, 'axes.spines.top': False,
             'axes.spines.right': False, 'pdf.fonttype': 42, 'svg.fonttype': 'none',
             'savefig.facecolor': 'white'}
    colors = ['#0072B2', '#D55E00', '#009E73', '#7B3294']
    markers = ['o', 's', '^', 'D']
    linestyles = ['-', '--', '-.', ':']

    def curve(ax, row, i, clean=False):
        values = ([row['clean']] if clean else []) + [row[key] for key in columns]
        if any(value is None or not math.isfinite(value) for value in values):
            raise ValueError('These summary curves require explicitly reported finite values.')
        ax.plot(([0.0] if clean else []) + sigmas, values, color=colors[i], marker=markers[i],
                linestyle=linestyles[i], linewidth=1.7, markersize=4.5, label=row['model'])
        ax.set(xlabel=r'Gaussian noise $\sigma$', ylim=(0, 1), xticks=[0, 0.05, 0.10, 0.20, 0.30])
        ax.grid(alpha=0.2)
        ax.legend(loc='best', frameon=False)

    def save(fig, name):
        # Resolve constrained layout before the first raster export.
        fig.canvas.draw()
        fig.savefig(output / f'{name}.png', dpi=200)
        fig.savefig(output / f'{name}.pdf', metadata={'Title': f'TexJEPA: {name}', 'Author': 'TexJEPA'})
        fig.savefig(output / f'{name}.svg')
        plt.close(fig)

    with plt.rc_context(style):
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.1), layout='constrained',
                                 gridspec_kw={'width_ratios': [1, 1, 1.25]})
        for i, row in enumerate(tables['table_iv']['rows']):
            curve(axes[0], row, i, clean=True)
        for i, row in enumerate(tables['table_v']['rows']):
            curve(axes[1], row, i)
        axes[0].set(title='A  Classification under noise', ylabel='Macro AUROC')
        axes[1].set(title='B  Representation drift', ylabel='1 - cosine similarity')
        for i, row in enumerate(tables['table_viii']['rows']):
            center, lower, upper = (row[key] for key in ('delta_lesion', 'ci95_lower', 'ci95_upper'))
            if not all(math.isfinite(v) for v in (center, lower, upper)) or not lower <= center <= upper:
                raise ValueError('Invalid reported lesion interval.')
            color = colors[0] if row['model'].startswith('I-JEPA') else colors[1]
            axes[2].errorbar(center, i, xerr=[[center-lower], [upper-center]], fmt='o', color=color,
                             capsize=3, markersize=5)
        axes[2].set(yticks=range(len(tables['table_viii']['rows'])),
                    yticklabels=[r['model'] for r in tables['table_viii']['rows']],
                    xlabel='Delta Logit Drop (95% bootstrap CI)', xlim=(0, 0.27),
                    title='C  Classification-aligned lesion sensitivity')
        axes[2].invert_yaxis()
        axes[2].grid(axis='x', alpha=0.2)
        fig.suptitle('TexJEPA: robustness and lesion sensitivity', fontsize=14)
        fig.supxlabel('Reported experimental results | Tables IV, V and VIII', fontsize=9)
        save(fig, 'main_results')

        fig, axes = plt.subplots(1, 3, figsize=(13, 4.1), layout='constrained')
        for i, row in enumerate(tables['table_ix']['rows'][:4]):
            curve(axes[0], row, i, clean=True)
        for i, row in enumerate(tables['table_x']['rows'][:4]):
            curve(axes[1], row, i)
        axes[0].set(title='A  Post-training robustness', ylabel='Macro AUROC')
        axes[1].set(title='B  Post-training drift', ylabel='1 - cosine similarity')
        rows = tables['table_xi']['rows'][:4]
        for i, row in enumerate(rows):
            value = row['delta_lesion']
            if value is None or not math.isfinite(value):
                raise ValueError('Missing reported post-training lesion value.')
            axes[2].scatter(value, i, color=colors[i], marker=markers[i], s=35, zorder=3)
            axes[2].annotate(f'{value:.3f}', (value, i), xytext=(7, 0), textcoords='offset points', va='center')
        axes[2].set(yticks=range(len(rows)), yticklabels=[r['model'] for r in rows],
                    xlabel='Delta Logit Drop', xlim=(0, 0.24),
                    title='C  Lesion sensitivity after post-training')
        axes[2].invert_yaxis()
        axes[2].grid(axis='x', alpha=0.2)
        fig.suptitle('TexJEPA post-training: robustness-semantics trade-off', fontsize=14)
        fig.supxlabel('Reported experimental results | Tables IX, X and XI; panel C shows reported point estimates', fontsize=9)
        save(fig, 'post_training_results')
    manifest = {'schema_version': 1, 'evidence': 'paper_reported',
                'source': 'results/paper/tables.json',
                'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                'transformations': ['Select the named source tables.', 'Draw lines through reported noise levels.',
                                    'Render Table VIII confidence intervals as horizontal error bars.'],
                'unreported_values': 'We do not estimate missing values or add confidence intervals to Table XI.',
                'figures': [{'file': p.name, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths]}
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('results/paper/tables.json'))
    parser.add_argument('--output', type=Path, default=Path('outputs/paper_figures'))
    args = parser.parse_args()
    for path in plot_paper_results(args.source, args.output):
        print(path)


if __name__ == '__main__':
    main()
