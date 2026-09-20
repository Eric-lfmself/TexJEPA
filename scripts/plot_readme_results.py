"""Render our README result gallery from the committed aggregate tables."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory

# Display names describe the same historical checkpoints recorded in the tables.
MODEL_LABELS = {
    'v3.1 (Baseline)': 'I-JEPA-H/201 · v3.1',
    'v4 (+noise)': 'TexJEPA-N · v4',
    'v5 (+reg)': 'TexJEPA-R · v5',
    'v6 (+vicreg)': 'TexJEPA-C · v6',
    'v3.1 baseline': 'I-JEPA-H/201',
    'v4 + noise': 'TexJEPA-N',
    'v5 + register': 'TexJEPA-R',
    'v6 + var/cov': 'TexJEPA-C',
}
COLORS = {
    'baseline': '#3C5488', 'noise': '#4DBBD5', 'register': '#00A087',
    'covariance': '#E64B35', 'mae': '#8491B4', 'eva': '#F39B7F', 'rad': '#7E6148',
}
# Redundant marker/line encodings preserve model identity beyond color.
MODEL_STYLES = {
    'v3.1 (Baseline)': ('baseline', 'o', '--'),
    'v4 (+noise)': ('noise', 's', '-'),
    'v5 (+reg)': ('register', '^', '-'),
    'v6 (+vicreg)': ('covariance', 'D', '-'),
    'v3.1 baseline': ('baseline', 'o', '--'),
    'v4 + noise': ('noise', 's', '-'),
    'v5 + register': ('register', '^', '-'),
    'v6 + var/cov': ('covariance', 'D', '-'),
    'MAE-H/300': ('mae', 'v', '--'),
    'EVA-X-B/300': ('eva', 'P', '-.'),
    'RAD-DINO-B/300': ('rad', 'X', ':'),
}
INK = '#26334A'
MUTED = '#526176'
GRID = '#E8ECF0'
STYLE = {
    'font.family': 'DejaVu Sans', 'font.size': 14, 'text.color': INK,
    'axes.labelcolor': MUTED, 'xtick.color': MUTED, 'ytick.color': MUTED,
    'axes.titlesize': 17, 'axes.titleweight': 'bold', 'axes.labelsize': 14,
    'xtick.labelsize': 12, 'ytick.labelsize': 12, 'legend.fontsize': 12,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.edgecolor': '#CBD5E1', 'axes.linewidth': .8,
    'pdf.fonttype': 42, 'svg.fonttype': 'none', 'svg.hashsalt': 'texjepa-readme',
    'savefig.facecolor': 'white', 'figure.facecolor': 'white',
}

FIGURE_DEFINITIONS = {
    'baseline_robustness': {
        'tables': ['table_iv', 'table_v'],
        'title': 'High clean utility, fragile representations',
        'subtitle': 'Frozen linear probes reveal different responses to Gaussian noise.',
        'caption': 'Tables IV–V. Macro AUROC under a frozen linear probe and cosine representation '
                   'drift (1 − cosine similarity) for I-JEPA-H/300 and MAE-H/300. Sigma is the '
                   'input-intensity noise standard deviation in our evaluation preprocessing. '
                   'AUROC uses a shared 0.40–1.00 display range across the gallery; drift uses its '
                   'full 0–2 range. Lines join reported conditions; no interpolation is evaluated.',
        'alt': 'Two line plots compare I-JEPA-H/300 and MAE-H/300 across Gaussian noise levels. '
               'I-JEPA starts with higher clean AUROC but loses more utility at light noise and '
               'has greater representation drift at all four reported noise levels.',
        'footer': 'Tables IV–V · Gaussian noise σ: input-intensity standard deviation · Drift range: 0–2',
    },
    'post_training': {
        'tables': ['table_ix', 'table_x'],
        'title': 'Post-training changes the robustness profile',
        'subtitle': 'Seven checkpoints, measured at the same reported noise levels.',
        'caption': 'Tables IX–X. Macro AUROC and cosine drift (1 − cosine similarity) for all seven '
                   'reported checkpoints. The post-training baseline is I-JEPA-H/201 (v3.1), '
                   'distinct from I-JEPA-H/300 in Tables IV–VII. TexJEPA-N/R/C map to v4/v5/v6. '
                   'Sigma is input-intensity standard deviation. AUROC is displayed over 0.40–1.00 '
                   'and drift over the full 0–2 range. Lines join reported conditions only.',
        'alt': 'Two line plots show all seven reported checkpoints. TexJEPA variants preserve '
               'AUROC at light noise; TexJEPA-C maintains the lowest drift at every reported '
               'noise level, while its AUROC also declines at the strongest noise.',
        'footer': 'Tables IX–X · v3.1 = I-JEPA-H/201 · σ: input-intensity standard deviation · Drift range: 0–2',
    },
    'lesion_tradeoff': {
        'tables': ['table_viii', 'table_xi'],
        'title': 'Stability and lesion sensitivity tell different stories',
        'subtitle': 'A lesion-specific readout complements global representation drift.',
        'caption': 'Left: Table VIII lesion-minus-matched-control logit drop with the reported '
                   '95% bootstrap confidence intervals; bootstrap sample count is not specified '
                   'in the aggregate tables. Right: Table XI pairs cosine drift at σ = 0.20 with '
                   'lesion-specific logit drop for all five reported models. The two panels '
                   'use different checkpoint sets, shown by their labels. No confidence intervals '
                   'are inferred for Table XI and no trend or regression is fitted. Cosine drift '
                   'is dimensionless and shown over its full 0–2 range; lesion differences are in logit units.',
        'alt': 'A forest plot shows larger lesion-minus-control logit drops for the three '
               'I-JEPA checkpoints than for the two MAE checkpoints, with reported confidence '
               'intervals. A separate scatter plot shows that the lowest-drift post-training '
               'variant, TexJEPA-C, also has a small lesion-specific logit drop.',
        'footer': 'Tables VIII & XI · Lesion sensitivity: lesion-minus-control logit drop · No fitted trend',
    },
    'frequency_and_probes': {
        'tables': ['table_vi', 'table_vii'],
        'title': 'Diagnosing texture dependence',
        'subtitle': 'Frequency perturbations and probe capacity test complementary explanations.',
        'caption': 'Left: Table VI, I-JEPA-H/300 frozen linear-probe AUROC for every reported '
                   'frequency condition; cutoffs and bands are normalized frequencies. Right: '
                   'Table VII, clean and Gaussian-noise σ = 0.05 AUROC for all six model/protocol '
                   'combinations. Hollow circles show clean inputs and solid squares show noise. '
                   'Both panels use the same AUROC range, 0.40–1.00. The source table’s missing '
                   'clean drift/drop and MLP drift entries are retained as null in the manifest '
                   'and are not plotted or filled in. This figure plots original AUROCs only; '
                   'the reported Table VI Drop discrepancy is unchanged.',
        'alt': 'A point plot shows I-JEPA-H/300 AUROC for clean, low-pass and three band-corruption '
               'conditions, with the largest loss in the middle band. A paired point plot shows '
               'clean and noise AUROC for linear, MLP and partial fine-tuning protocols for '
               'I-JEPA-H/300 and MAE-H/300.',
        'footer': 'Tables VI–VII · Frequency boundaries are normalized · Noise σ = 0.05 · AUROC range: 0.40–1.00',
    },
}


def _finite(value, context):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'Reported finite numeric value required: {context}')
    return value


def _style(model):
    key, marker, line = MODEL_STYLES.get(model, ('baseline', 'o', '-'))
    if model.startswith('MAE'):
        key = 'mae'
    return {'color': COLORS[key], 'marker': marker, 'linestyle': line}


def extract_gallery(source: Path) -> dict:
    """Select whole records without rewriting values, nulls, or source precision."""
    source = Path(source)
    raw = source.read_bytes()
    document = json.loads(raw)
    tables = {table['id']: table for table in document['tables']}
    selections = {}
    for name, definition in FIGURE_DEFINITIONS.items():
        selected = []
        for table_id in definition['tables']:
            if table_id not in tables:
                raise ValueError(f'Missing source table: {table_id}')
            table = deepcopy(tables[table_id])
            for column in table['columns']:
                if column['type'] == 'number':
                    for row in table['rows']:
                        value = row[column['key']]
                        if value is not None:
                            _finite(value, f"{table_id}.{column['key']}")
            selected.append(table)
        selections[name] = selected
    return {'source': 'results/paper/tables.json',
            'source_sha256': hashlib.sha256(raw).hexdigest(),
            'source_metadata': deepcopy(document['source']),
            'selections': selections}


def _curve_records(table, include_clean):
    columns = [column for column in table['columns'] if column.get('condition') == 'gaussian_noise']
    if include_clean:
        columns = [column for column in table['columns'] if column.get('condition') == 'clean'] + columns
    records = []
    for index, row in enumerate(table['rows']):
        values = [_finite(row[c['key']], f"{table['id']} row {index} {c['key']}") for c in columns]
        records.append({'id': f"{table['id']}/row/{index}", 'source_row': index,
                        'model': row['model'], 'label': MODEL_LABELS.get(row['model'], row['model']),
                        'columns': [c['key'] for c in columns],
                        'x': [c.get('sigma', 0.0) for c in columns], 'y': values})
    return records


def _frame(plt, name, *, legend=False, wide_labels=False):
    definition = FIGURE_DEFINITIONS[name]
    fig = plt.figure(figsize=(13.2, 7.4 if legend else 7.0))
    fig.text(.055, .945, definition['title'], fontsize=23, weight='bold', color=INK, va='top')
    fig.text(.055, .878, definition['subtitle'], fontsize=14, color=MUTED, va='top')
    fig.text(.055, .031, definition['footer'], fontsize=10.5, color=MUTED, va='bottom')
    grid = fig.add_gridspec(1, 2, left=.095 if not wide_labels else .17,
                            right=.97, bottom=.28 if legend else .18,
                            top=.755, wspace=.29 if not wide_labels else .75)
    axes = [fig.add_subplot(grid[0, i]) for i in range(2)]
    for ax in axes:
        ax.set_axisbelow(True)
        ax.grid(axis='y', color=GRID, linewidth=.8)
        ax.tick_params(length=0, pad=8)
    return fig, axes


def _panel(ax, letter, title):
    ax.set_title(f'{letter}   {title}', loc='left', pad=20)


def _noise_axes(ax, *, drift):
    ax.set(xlim=(-.005, .31), xticks=[0, .05, .1, .2, .3],
           xticklabels=['Clean', '.05', '.10', '.20', '.30'],
           xlabel='Gaussian noise σ', ylim=(0, 2) if drift else (.4, 1),
           ylabel='Cosine drift  ↓' if drift else 'Macro AUROC  ↑',
           yticks=[0, .5, 1, 1.5, 2] if drift else [.4, .5, .6, .7, .8, .9, 1])
    if not drift:
        ax.axhline(.5, linewidth=1, color='#BCC7D3', linestyle=':', zorder=0)


def _render_curves(plt, name, tables):
    has_many = name == 'post_training'
    fig, axes = _frame(plt, name, legend=True)
    plotted = []
    for ax, table, clean in zip(axes, tables, (True, False)):
        records = _curve_records(table, clean)
        for record in records:
            ax.plot(record['x'], record['y'], label=record['label'],
                    gid=record['id'], linewidth=2.5, markersize=6.5,
                    markeredgewidth=1.1, markeredgecolor='white', **_style(record['model']))
        _noise_axes(ax, drift=not clean)
        plotted.extend(records)
    _panel(axes[0], 'A', 'Classification under noise')
    _panel(axes[1], 'B', 'Representation drift')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(.5, .081),
               ncol=4 if has_many else 2, frameon=False, columnspacing=1.45,
               handlelength=2.4, labelspacing=1.3)
    if not has_many:
        for ax, records in zip(axes, [_curve_records(tables[0], True), _curve_records(tables[1], False)]):
            for record in records:
                value = record['y'][-1]
                offset = 10 if record['model'].startswith('I-JEPA') else -16
                ax.annotate(f'{value:.3f}', (record['x'][-1], value),
                            xytext=(-4, offset), textcoords='offset points', ha='right',
                            color=_style(record['model'])['color'], fontsize=12, weight='bold')
    return fig, plotted


def _render_lesions(plt, tables):
    fig, axes = _frame(plt, 'lesion_tradeoff', wide_labels=True)
    forest, scatter = axes
    records = []
    for index, row in enumerate(tables[0]['rows']):
        center, lower, upper = [_finite(row[k], k) for k in ('delta_lesion', 'ci95_lower', 'ci95_upper')]
        if not lower <= center <= upper:
            raise ValueError('Invalid reported 95% confidence interval')
        sid = f"{tables[0]['id']}/row/{index}"
        color = _style(row['model'])['color']
        container = forest.errorbar(center, index, xerr=[[center-lower], [upper-center]],
                                    color=color, fmt='o', capsize=5, elinewidth=2.5,
                                    markersize=8, markeredgecolor='white', markeredgewidth=1.1)
        container.lines[0].set_gid(sid)
        for collection in container.lines[2]:
            collection.set_gid(sid + '/ci95')
        forest.annotate(f'{center:.3f}', (center, index), xytext=(0, -21),
                        textcoords='offset points', ha='center', color=color, fontsize=11)
        records.append({'id': sid, 'model': row['model'], 'x': [center], 'y': [index],
                        'ci95_lower': lower, 'ci95_upper': upper})
    forest.set(yticks=range(len(tables[0]['rows'])),
               yticklabels=[r['model'] for r in tables[0]['rows']],
               xlabel='Lesion-specific logit drop  ↑', xlim=(0, .28),
               xticks=[0, .1, .2], ylim=(len(tables[0]['rows'])-.35, -.65))
    forest.grid(False)
    forest.grid(axis='x', color=GRID)
    _panel(forest, 'A', 'Lesion sensitivity')
    forest.text(0, 1.01, 'Reported 95% bootstrap CI', transform=forest.transAxes,
                fontsize=10.5, color=MUTED)
    offsets = {'v3.1 baseline': (10, 8), 'v4 + noise': (10, 8),
               'v5 + register': (10, -17), 'v6 + var/cov': (8, 13), 'MAE-H/300': (10, -18)}
    for index, row in enumerate(tables[1]['rows']):
        x, y = (_finite(row[k], k) for k in ('drift_sigma_0_20', 'delta_lesion'))
        sid = f"{tables[1]['id']}/row/{index}"
        style = _style(row['model'])
        scatter.scatter([x], [y], color=style['color'], marker=style['marker'],
                        s=110, edgecolor='white', linewidth=1, zorder=4, gid=sid)
        scatter.annotate(MODEL_LABELS.get(row['model'], row['model']), (x, y),
                         xytext=offsets.get(row['model'], (10, 8)), textcoords='offset points',
                         color=INK, fontsize=11, weight='bold')
        records.append({'id': sid, 'model': row['model'], 'x': [x], 'y': [y]})
    scatter.set(xlim=(0, 2), ylim=(0, .26), xticks=[0, .5, 1, 1.5, 2],
                yticks=[0, .05, .1, .15, .2, .25],
                xlabel='Cosine drift at σ = .20  ↓', ylabel='Lesion-specific logit drop  ↑')
    _panel(scatter, 'B', 'Post-training trade-off')
    return fig, records


def _render_frequency(plt, tables):
    fig, axes = _frame(plt, 'frequency_and_probes', wide_labels=True)
    for ax in axes:
        bounds = ax.get_position()
        ax.set_position([bounds.x0, .22, bounds.width, bounds.y1 - .22])
    freq, probe = axes
    records = []
    condition_labels = {'Low-pass cutoff 0.15': 'Low-pass\ncutoff .15',
                        'Band corrupt 0.00–0.20': 'Band corrupt\n.00–.20',
                        'Band corrupt 0.20–0.45': 'Band corrupt\n.20–.45',
                        'Band corrupt 0.45–1.00': 'Band corrupt\n.45–1.00'}
    for index, row in enumerate(tables[0]['rows']):
        value = _finite(row['auroc'], 'frequency AUROC')
        sid = f"{tables[0]['id']}/row/{index}"
        freq.scatter([value], [index], color=COLORS['baseline'], s=90,
                     marker='o' if index == 0 else 'D', edgecolor='white', linewidth=1, gid=sid, zorder=3)
        freq.annotate(f'{value:.3f}', (value, index), xytext=(11, 0),
                      textcoords='offset points', va='center', fontsize=11, color=INK)
        records.append({'id': sid, 'condition': row['condition'], 'x': [value], 'y': [index]})
    freq.set(yticks=range(len(tables[0]['rows'])),
             yticklabels=[condition_labels.get(r['condition'], r['condition']) for r in tables[0]['rows']],
             ylim=(len(tables[0]['rows'])-.35, -.65))
    _panel(freq, 'A', 'Frequency sensitivity')
    freq.text(0, 1.01, 'I-JEPA-H/300 · frozen linear probe', transform=freq.transAxes,
              fontsize=10.5, color=MUTED)
    positions = [0, 1, 2, 3.5, 4.5, 5.5]
    if len(tables[1]['rows']) != len(positions):
        raise ValueError('Expected all six reported model/protocol combinations')
    labels = []
    for index, (row, y) in enumerate(zip(tables[1]['rows'], positions)):
        clean, noise = [_finite(row[key], key) for key in ('clean', 'sigma_0_05')]
        color = _style(row['model'])['color']
        sid = f"{tables[1]['id']}/row/{index}"
        probe.plot([noise, clean], [y, y], color=color, alpha=.55, linewidth=2,
                    gid=sid + '/pair')
        probe.scatter([clean], [y], facecolors='white', edgecolors=color, s=90,
                       linewidth=2, marker='o', zorder=3, gid=sid + '/clean')
        probe.scatter([noise], [y], color=color, s=70, marker='s', zorder=3,
                       gid=sid + '/sigma_0_05')
        short = row['protocol'].replace(' probe', '')
        labels.append(short)
        records.append({'id': sid, 'model': row['model'], 'protocol': row['protocol'],
                        'clean': clean, 'sigma_0_05': noise, 'position': y})
    probe.set(yticks=positions, yticklabels=labels, ylim=(6.15, -.65))
    probe.text(.005, .972, 'I-JEPA-H/300', transform=probe.transAxes,
               color=COLORS['baseline'], fontsize=10.5, va='bottom')
    probe.text(.005, .452, 'MAE-H/300', transform=probe.transAxes,
               color=COLORS['mae'], fontsize=10.5, va='bottom')
    _panel(probe, 'B', 'Probe capacity')
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([], [], marker='o', color=COLORS['baseline'],
                               markerfacecolor='white', linestyle='None', label='Clean', markersize=8),
                        Line2D([], [], marker='s', color=COLORS['baseline'], linestyle='None',
                               label='Noise σ = .05', markersize=8)],
               loc='lower center', bbox_to_anchor=(.70, .071), frameon=False, ncol=2)
    for ax in axes:
        ax.set(xlim=(.4, 1), xticks=[.4, .6, .8, 1], xlabel='Macro AUROC  ↑')
        ax.grid(False)
        ax.grid(axis='x', color=GRID)
    return fig, records


def build_figure(name: str, selection: list[dict]):
    """Return a figure and the measurements encoded by its data artists."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    with plt.rc_context(STYLE):
        if name in ('baseline_robustness', 'post_training'):
            return _render_curves(plt, name, selection)
        if name == 'lesion_tradeoff':
            return _render_lesions(plt, selection)
        if name == 'frequency_and_probes':
            return _render_frequency(plt, selection)
    raise ValueError(f'Unknown gallery figure: {name}')


def plot_readme_results(source: Path, output: Path) -> list[Path]:
    """Write PNG/SVG/PDF figures and provenance, preserving any existing files."""
    import matplotlib
    import matplotlib.pyplot as plt
    gallery = extract_gallery(source)
    output = Path(output)
    relative = [f'{name}.{suffix}' for name in FIGURE_DEFINITIONS for suffix in ('png', 'svg', 'pdf')]
    relative.append('manifest.json')
    if any((output / name).exists() or (output / name).is_symlink() for name in relative):
        raise FileExistsError('Choose a new output directory; existing gallery files are preserved.')
    output.mkdir(parents=True, exist_ok=True)
    manifest = {k: v for k, v in gallery.items() if k != 'selections'}
    manifest.update({'schema_version': 1, 'evidence': 'paper_reported',
                     'matplotlib_version': matplotlib.__version__,
                     'destination': 'GitHub README, approximately 850 px wide',
                     'unreported_values': 'Retain nulls in selected records; do not impute or draw them.',
                     'transformations': ['Select complete source tables without changing values or display precision.',
                                         'Map historical checkpoint names to TexJEPA variant names for labels.',
                                         'Join reported noise conditions with straight segments, without evaluated interpolation.',
                                         'Display original AUROCs only for frequency and probe comparisons.',
                                         'Draw original Table VIII lower/upper confidence limits.',
                                         'Pair Table XI drift and lesion values within each original row; fit no trend.'],
                     'figures': []})
    with TemporaryDirectory(prefix='.gallery-', dir=output) as staging:
        staging = Path(staging)
        for name, definition in FIGURE_DEFINITIONS.items():
            fig, plotted = build_figure(name, gallery['selections'][name])
            try:
                with plt.rc_context(STYLE):
                    fig.canvas.draw()
                    for suffix in ('png', 'svg', 'pdf'):
                        metadata = {'Title': definition['title']}
                        if suffix in ('svg', 'pdf'):
                            metadata['Creator'] = 'TexJEPA'
                        if suffix == 'pdf':
                            metadata.update({'CreationDate': None, 'ModDate': None})
                        if suffix == 'svg':
                            metadata['Date'] = None
                        fig.savefig(staging / f'{name}.{suffix}', dpi=180, metadata=metadata)
                manifest['figures'].append({
                    'name': name, 'caption': definition['caption'], 'alt': definition['alt'],
                    'selected_source_tables': gallery['selections'][name],
                    'palette': {'description': 'Nature Publishing Group-inspired categorical colors',
                                'model_colors': COLORS},
                    'plotted_measurements': plotted,
                    'size_inches': list(fig.get_size_inches()), 'png_dpi': 180,
                    'files': [{'file': f'{name}.{suffix}',
                               'sha256': hashlib.sha256((staging / f'{name}.{suffix}').read_bytes()).hexdigest()}
                              for suffix in ('png', 'svg', 'pdf')],
                })
            finally:
                plt.close(fig)
        (staging / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
        for name in relative:
            # A second check catches a target created during rendering.
            if (output / name).exists() or (output / name).is_symlink():
                raise FileExistsError(f'Preserving existing gallery file: {name}')
        for name in relative:
            os.replace(staging / name, output / name)
    return [output / name for name in relative]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('results/paper/tables.json'))
    parser.add_argument('--output', type=Path, default=Path('outputs/readme_results'))
    args = parser.parse_args()
    for path in plot_readme_results(args.source, args.output):
        print(path)


if __name__ == '__main__':
    main()
