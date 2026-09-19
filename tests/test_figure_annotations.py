"""Validate the public manuscript figure annotations and their CSV export."""
import csv
import importlib.util
import re

import pytest
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('figure_results_export', ROOT / 'scripts/export_paper_results.py')
EXPORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXPORT)


def test_figure_annotations_preserve_source_matrix_and_display_precision():
    data = json.loads((ROOT / 'results/paper/figure_annotations.json').read_text())
    assert data['evidence'] == 'paper_reported'
    assert data['source']['path'] == 'texjepa-manuscript'
    assert data['source']['availability'] == 'not_distributed'
    assert re.fullmatch(r'[0-9a-f]{64}', data['source']['sha256'])
    tables = json.loads((ROOT / 'results/paper/tables.json').read_text())
    assert data['source']['sha256'] == tables['source']['sha256']
    EXPORT.validate_figure_annotations(data, tables)
    rows = data['records']
    assert len(rows) == 90
    assert len({(r['metric'], r['model'], r['condition']) for r in rows}) == len(rows)
    assert {r['pdf_page'] for r in rows} == {8}
    assert {r['figure'] for r in rows} == {'4'}
    for row in rows:
        assert math.isfinite(row['value'])
        assert row['value'] == float(row['displayed_value'])
        assert row['decimal_places'] == 2
        assert row['displayed_value'] == f"{row['value']:.2f}"
        assert row['evidence'] == 'paper_reported'
    groups = {}
    for row in rows:
        groups.setdefault((row['metric'], row['model']), set()).add(row['condition'])
    assert len(groups) == 6
    assert all(len(conditions) == 15 for conditions in groups.values())
    assert all(conditions == next(iter(groups.values())) for conditions in groups.values())


def test_figure_annotations_csv_matches_canonical_json():
    data = json.loads((ROOT / 'results/paper/figure_annotations.json').read_text())
    with (ROOT / 'results/paper/figure_04_robustness_matrix.csv').open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    assert rows == [{k: str(v) for k, v in record.items()} for record in data['records']]


def test_figure_annotation_validation_rejects_changed_display_precision():
    data = json.loads((ROOT / 'results/paper/figure_annotations.json').read_text())
    tables = json.loads((ROOT / 'results/paper/tables.json').read_text())
    data['records'][0]['displayed_value'] = '0.910'
    with pytest.raises(ValueError, match='precision'):
        EXPORT.validate_figure_annotations(data, tables)


def test_figure_annotation_validation_rejects_mismatched_source():
    data = json.loads((ROOT / 'results/paper/figure_annotations.json').read_text())
    tables = json.loads((ROOT / 'results/paper/tables.json').read_text())
    data['source']['sha256'] = '0' * 64
    with pytest.raises(ValueError, match='SHA-256'):
        EXPORT.validate_figure_annotations(data, tables)


def test_figure_csv_validation_rejects_changed_values(tmp_path):
    data = json.loads((ROOT / 'results/paper/figure_annotations.json').read_text())
    source = ROOT / 'results/paper/figure_04_robustness_matrix.csv'
    changed = tmp_path / source.name
    changed.write_text(source.read_text().replace('0.91', '0.92', 1))
    with pytest.raises(ValueError, match='differs'):
        EXPORT.validate_figure_csv(changed, data)
