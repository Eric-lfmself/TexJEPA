"""Audit that our homepage figures encode the released measurements faithfully."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pytest

from scripts.plot_readme_results import (
    FIGURE_DEFINITIONS, build_figure, extract_gallery, plot_readme_results,
)

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'results/paper/tables.json'


@pytest.fixture(scope='module')
def gallery():
    return extract_gallery(SOURCE)


def test_selection_preserves_complete_source_records_and_nulls(gallery):
    original = {table['id']: table for table in json.loads(SOURCE.read_text())['tables']}
    selected = {}
    for tables in gallery['selections'].values():
        for table in tables:
            assert table == original[table['id']]
            selected[table['id']] = table
    assert selected['table_vi']['rows'][0]['drop'] is None
    assert selected['table_vi']['rows'][0]['drift'] is None
    assert selected['table_vi']['rows'][3]['drop'] == -0.317
    assert selected['table_vii']['rows'][1]['drift_sigma_0_05'] is None
    assert selected['table_vii']['rows'][4]['drift_sigma_0_05'] is None
    assert gallery['source_sha256'] == hashlib.sha256(SOURCE.read_bytes()).hexdigest()


@pytest.mark.parametrize('name,expected_models', [('baseline_robustness', 2), ('post_training', 7)])
def test_every_robustness_artist_matches_original_measurements(gallery, name, expected_models):
    tables = gallery['selections'][name]
    fig, _ = build_figure(name, tables)
    try:
        for ax, table in zip(fig.axes, tables):
            lines = {line.get_gid(): line for line in ax.lines if line.get_gid()}
            assert len(lines) == expected_models == len(table['rows'])
            columns = [c for c in table['columns'] if c.get('condition') in ('clean', 'gaussian_noise')]
            for index, row in enumerate(table['rows']):
                line = lines[f"{table['id']}/row/{index}"]
                np.testing.assert_array_equal(line.get_xdata(), [c.get('sigma', 0) for c in columns])
                np.testing.assert_array_equal(line.get_ydata(), [row[c['key']] for c in columns])
            if table['id'] in ('table_v', 'table_x'):
                assert all(0 not in line.get_xdata() for line in lines.values())
                assert ax.get_ylim() == (0, 2)
        if name == 'post_training':
            labels = fig.axes[0].get_legend_handles_labels()[1]
            assert 'I-JEPA-H/201 · v3.1' in labels
            assert not any('I-JEPA-H/300' in label for label in labels)
            assert {'TexJEPA-N · v4', 'TexJEPA-R · v5', 'TexJEPA-C · v6'} <= set(labels)
    finally:
        plt.close(fig)


def test_lesion_artists_preserve_exact_intervals_and_row_pairing(gallery):
    tables = gallery['selections']['lesion_tradeoff']
    fig, _ = build_figure('lesion_tradeoff', tables)
    try:
        forest, scatter = fig.axes
        lines = {line.get_gid(): line for line in forest.lines if line.get_gid()}
        intervals = {c.get_gid(): c for c in forest.collections if c.get_gid()}
        points = {c.get_gid(): c for c in scatter.collections if c.get_gid()}
        for index, row in enumerate(tables[0]['rows']):
            sid = f'table_viii/row/{index}'
            np.testing.assert_array_equal(lines[sid].get_xdata(), [row['delta_lesion']])
            np.testing.assert_allclose(intervals[sid + '/ci95'].get_segments()[0],
                                       [[row['ci95_lower'], index], [row['ci95_upper'], index]],
                                       rtol=0, atol=1e-15)
        assert len(points) == len(tables[1]['rows']) == 5
        assert not scatter.lines
        for index, row in enumerate(tables[1]['rows']):
            np.testing.assert_array_equal(points[f'table_xi/row/{index}'].get_offsets(),
                                          [[row['drift_sigma_0_20'], row['delta_lesion']]])
    finally:
        plt.close(fig)


def test_frequency_and_all_probe_pairs_show_original_auroc(gallery):
    tables = gallery['selections']['frequency_and_probes']
    fig, _ = build_figure('frequency_and_probes', tables)
    try:
        freq, probe = fig.axes
        points = {c.get_gid(): c for c in freq.collections if c.get_gid()}
        for index, row in enumerate(tables[0]['rows']):
            np.testing.assert_array_equal(points[f'table_vi/row/{index}'].get_offsets(),
                                          [[row['auroc'], index]])
        points = {c.get_gid(): c for c in probe.collections if c.get_gid()}
        assert len(points) == 12
        for index, row in enumerate(tables[1]['rows']):
            for column in ('clean', 'sigma_0_05'):
                assert points[f'table_vii/row/{index}/{column}'].get_offsets()[0, 0] == row[column]
        assert freq.get_xlim() == probe.get_xlim() == (.4, 1)
    finally:
        plt.close(fig)


def test_nonfinite_source_measurements_and_invalid_intervals_fail(tmp_path, gallery):
    source = json.loads(SOURCE.read_text())
    source['tables'][3]['rows'][0]['clean'] = float('nan')
    path = tmp_path / 'invalid.json'
    path.write_text(json.dumps(source))
    with pytest.raises(ValueError, match='finite'):
        extract_gallery(path)
    selection = deepcopy(gallery['selections']['lesion_tradeoff'])
    selection[0]['rows'][0]['ci95_upper'] = 0
    try:
        with pytest.raises(ValueError, match='confidence interval'):
            build_figure('lesion_tradeoff', selection)
    finally:
        plt.close('all')


def test_exports_have_complete_provenance_and_refuse_overwrite(tmp_path, gallery):
    output = tmp_path / 'gallery'
    paths = plot_readme_results(SOURCE, output)
    assert len(paths) == 13
    manifest = json.loads((output / 'manifest.json').read_text())
    assert manifest['source_sha256'] == hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    assert {f['name'] for f in manifest['figures']} == set(FIGURE_DEFINITIONS)
    for figure in manifest['figures']:
        assert figure['selected_source_tables'] == gallery['selections'][figure['name']]
        for artifact in figure['files']:
            assert hashlib.sha256((output / artifact['file']).read_bytes()).hexdigest() == artifact['sha256']
    before = {p.name: p.read_bytes() for p in paths}
    with pytest.raises(FileExistsError):
        plot_readme_results(SOURCE, output)
    assert {p.name: p.read_bytes() for p in paths} == before
    partial = tmp_path / 'partial'
    partial.mkdir()
    (partial / 'lesion_tradeoff.pdf').write_bytes(b'Preserve this figure')
    with pytest.raises(FileExistsError):
        plot_readme_results(SOURCE, partial)
    assert list(partial.iterdir()) == [partial / 'lesion_tradeoff.pdf']
    assert (partial / 'lesion_tradeoff.pdf').read_bytes() == b'Preserve this figure'
