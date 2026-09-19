#!/usr/bin/env python3
"""Validate our manuscript transcriptions and export CSV/Markdown using stdlib only.

Run from a source checkout: python scripts/export_paper_results.py [--check].
The aggregate results are repository assets, not wheel package data.
The private manuscript is not distributed; --source-pdf optionally verifies it.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import re
from pathlib import Path

ROMANS = ('I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X', 'XI', 'XII', 'XIII')
SOURCE_PAGES = (3, 4, 4, 5, 6, 7, 8, 8, 8, 9, 10, 10, 11)
PROVENANCE = ('evidence', 'dataset_type', 'source_pdf', 'source_pdf_page', 'source_table')


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_source(source: dict, source_pdf: Path | None = None) -> None:
    """Validate the private manuscript locator and optionally its supplied bytes."""
    _require(source['path'] == 'texjepa-manuscript', 'Unexpected manuscript identifier.')
    _require(source['availability'] == 'not_distributed', 'The manuscript must remain not_distributed.')
    _require(isinstance(source['sha256'], str) and re.fullmatch(r'[0-9a-f]{64}', source['sha256']) is not None, 'Invalid manuscript SHA-256.')
    if source_pdf is not None:
        _require(source_pdf.is_file(), f'Manuscript not found: {source_pdf}. Supply an existing private file with --source-pdf.')
        digest = hashlib.sha256(source_pdf.read_bytes()).hexdigest()
        _require(digest == source['sha256'], 'The manuscript SHA-256 differs from the transcription source.')


def validate(data: dict, supplementary: dict, source_pdf: Path | None = None) -> None:
    """Reject mismatched provenance, malformed cells, or false precision without a PDF."""
    validate_source(data['source'], source_pdf)
    _require(data['schema_version'] == supplementary['schema_version'] == '1.0.0', 'Unsupported aggregate schema version.')
    _require(data['source']['pdf_pages'] == 13, 'Unexpected manuscript page count.')
    _require(data['source'] == supplementary['source'], 'Supplementary data uses a different manuscript source.')
    for document in (data, supplementary):
        _require(document['publication_status'] == 'unsubmitted_manuscript', 'Unexpected manuscript publication status.')
        _require(document['evidence'] == 'paper_reported', 'Expected paper_reported evidence.')
        _require(document['dataset_type'] == 'aggregate_experimental_results', 'Expected aggregate experimental results.')
    tables = data['tables']
    _require(data['table_count'] == len(tables) == 13, 'Expected all 13 manuscript tables.')
    _require(tuple(t['table'] for t in tables) == ROMANS, 'Table order/identifiers differ from I-XIII.')
    for table, page in zip(tables, SOURCE_PAGES):
        _require(table['id'] == f"table_{table['table'].lower()}", 'Table id mismatch.')
        _require(table['source'] == {'path': data['source']['path'], 'pdf_page': page, 'table': table['table']}, f"Source locator mismatch for {table['id']}.")
        _require(table['evidence'] == data['evidence'] and table['dataset_type'] == data['dataset_type'], 'Table evidence metadata mismatch.')
        columns = table['columns']
        keys = [c['key'] for c in columns]
        _require(len(keys) == len(set(keys)), f"Duplicate column keys in {table['id']}.")
        _require(len(table['rows']) == len(table['reported_rows']), f"Row count mismatch in {table['id']}.")
        documented_nulls = {(x['row_index'], x['column']) for x in table.get('missing_cells', [])}
        actual_nulls = set()
        for i, (row, reported) in enumerate(zip(table['rows'], table['reported_rows'])):
            _require(set(row) == set(keys), f"Row schema mismatch in {table['id']} row {i}.")
            _require(len(reported) == len(table['reported_headers']), f"Printed row width mismatch in {table['id']}.")
            for column in columns:
                value = row[column['key']]
                if value is None:
                    actual_nulls.add((i, column['key']))
                elif column['type'] == 'number':
                    _require(type(value) in (int, float) and math.isfinite(value), 'Invalid numeric cell.')
                    _require(abs(value - round(value, column['decimal_places'])) < 1e-12, 'Unreported extra numeric precision.')
                    if column['unit'] == 'macro AUROC':
                        _require(0 <= value <= 1, 'AUROC outside [0, 1].')
                    if column['unit'] == 'cosine drift':
                        _require(0 <= value <= 2, 'Cosine drift outside [0, 2].')
                else:
                    _require(isinstance(value, str), 'Invalid textual cell.')
            if table['table'] == 'VIII':
                expected = [row['model'], f"{row['delta_lesion']:.3f}", f"[{row['ci95_lower']:.3f}, {row['ci95_upper']:.3f}]"]
                _require(row['ci95_lower'] <= row['delta_lesion'] <= row['ci95_upper'], 'Invalid confidence interval ordering.')
            else:
                expected = ['–' if row[c['key']] is None else f"{row[c['key']]:.3f}" if c['type'] == 'number' else row[c['key']] for c in columns]
                if table['table'] == 'VI':
                    expected[2] = expected[2].replace('-', '−')
            _require(reported == expected, f"Printed and numeric cells disagree in {table['id']} row {i}.")
        _require(actual_nulls == documented_nulls, f"Undocumented or filled-in null cells in {table['id']}.")
    by_id = {table['id']: table for table in tables}
    _require(by_id['table_xii']['duplicate_of'] == 'table_ix', 'Missing Table XII duplication metadata.')
    _require(by_id['table_ix']['rows'] == by_id['table_xii']['rows'], 'Repeated Tables IX and XII disagree.')
    records = supplementary['records']
    _require(supplementary['record_count'] == len(records), 'Supplementary count mismatch.')
    _require(len({record['id'] for record in records}) == len(records), 'Duplicate supplementary record ids.')
    for record in records:
        _require(record['evidence'] == 'paper_reported' and record['dataset_type'] == 'aggregate_experimental_results', 'Supplementary evidence mismatch.')
        _require(record['source']['path'] == data['source']['path'] and 1 <= record['source']['pdf_page'] <= data['source']['pdf_pages'], 'Supplementary source mismatch.')
        if record['precision'] == 'approximate_range':
            _require(record['value'] is None and record['range_lower'] <= record['range_upper'], 'An approximate range must retain its bounds without an inferred point estimate.')
        else:
            _require(type(record['value']) in (int, float) and math.isfinite(record['value']), 'Invalid supplementary value.')
        if record['precision'] == 'reported_2_decimal_places':
            _require(record['reported'] == f"{record['value']:.2f}", 'Figure annotation precision changed.')


def validate_figure_annotations(data: dict, tables: dict) -> None:
    """Validate the complete printed Figure 4 matrix against aggregate provenance."""
    validate_source(data['source'])
    _require(data['schema_version'] == 1, 'Unsupported figure annotation schema version.')
    _require(data['evidence'] == 'paper_reported' and data['data_type'] == 'aggregate_experimental_results', 'Figure annotation evidence mismatch.')
    _require(data['source']['sha256'] == tables['source']['sha256'], 'Figure annotation manuscript SHA-256 mismatch.')
    _require(data['source']['figure'] == 4 and data['source']['pdf_page'] == 8, 'Figure annotation source location mismatch.')
    records = data['records']
    _require(len(records) == 90, 'Expected all 90 figure annotations.')
    _require(len({(r['metric'], r['model'], r['condition']) for r in records}) == 90, 'Duplicate figure annotations.')
    keys = {'figure', 'pdf_page', 'metric', 'model', 'condition', 'value', 'displayed_value', 'decimal_places', 'evidence'}
    groups = {}
    for record in records:
        _require(set(record) == keys, 'Figure annotation record schema mismatch.')
        _require(record['figure'] == '4' and record['pdf_page'] == 8, 'Figure annotation record location mismatch.')
        _require(record['evidence'] == 'paper_reported', 'Figure annotation record evidence mismatch.')
        value = record['value']
        _require(type(value) in (int, float) and math.isfinite(value), 'Invalid figure annotation value.')
        _require(record['decimal_places'] == 2 and record['displayed_value'] == f"{value:.2f}", 'Figure annotation display precision changed.')
        groups.setdefault((record['metric'], record['model']), set()).add(record['condition'])
    _require(len(groups) == 6 and all(len(conditions) == 15 for conditions in groups.values()), 'Incomplete figure annotation matrix.')
    _require(all(conditions == next(iter(groups.values())) for conditions in groups.values()), 'Inconsistent figure annotation conditions.')


def validate_figure_csv(path: Path, annotations: dict) -> None:
    """Require the public CSV to retain every canonical annotation and its precision."""
    with path.open(newline='', encoding='utf-8') as handle:
        reader = csv.DictReader(handle)
        _require(reader.fieldnames == list(annotations['records'][0]), 'Figure annotation CSV schema mismatch.')
        rows = list(reader)
    _require(rows == [{key: str(value) for key, value in record.items()} for record in annotations['records']], 'Figure annotation CSV differs from its canonical JSON.')


def _csv(headers: list[str], rows: list[list]) -> str:
    stream = io.StringIO(newline='')
    writer = csv.writer(stream, lineterminator='\n')
    writer.writerow(headers)
    writer.writerows(rows)
    return stream.getvalue()


def _escape(value: str) -> str:
    return value.replace('|', '\\|').replace('\n', ' ')


def exports(data: dict, supplementary: dict) -> dict[str, str]:
    """Build deterministic source-precision CSVs and readable manuscript tables."""
    outputs = {}
    markdown = ['# Historical manuscript tables', '', 'We report these aggregate results in our unsubmitted TexJEPA manuscript. We preserve the displayed values, historical model names, and missing cells. Protocol and interpretation tables are identified separately.', '', f"Source: private manuscript {data['source']['path']} (not distributed). SHA-256: `{data['source']['sha256']}`.", '']
    for table in data['tables']:
        columns = table['columns']
        headers = [column['key'] for column in columns] + list(PROVENANCE)
        rows = []
        for row in table['rows']:
            values = ['' if row[c['key']] is None else f"{row[c['key']]:.{c['decimal_places']}f}" if c['type'] == 'number' else row[c['key']] for c in columns]
            values += [table['evidence'], table['dataset_type'], table['source']['path'], table['source']['pdf_page'], table['source']['table']]
            rows.append(values)
        outputs[table['csv']] = _csv(headers, rows)
        markdown += [f"## Table {table['table']}. {table['title']}", '', f"PDF page {table['source']['pdf_page']} · {table['table_kind']} · [CSV]({table['csv']})", '', '| ' + ' | '.join(_escape(x) for x in table['reported_headers']) + ' |', '| ' + ' | '.join('---' for _ in table['reported_headers']) + ' |']
        markdown += ['| ' + ' | '.join(_escape(x) for x in row) + ' |' for row in table['reported_rows']]
        markdown += ['']
        markdown += [f'- {note}' for note in table['notes']]
        markdown += ['']
    markdown += ['## Caption, figure-annotation, and prose values', '', 'We keep the explicitly printed Figure 7 annotations and approximate NCA/preprocessing values in [supplementary.json](supplementary.json) and [supplementary.csv](csv/supplementary.csv). Their precision and source locations are recorded per value.', '', 'The printed Figure 4 heatmap annotations are stored separately in [figure_annotations.json](figure_annotations.json).', '']
    outputs['TABLES.md'] = '\n'.join(markdown)
    headers = ['id', 'model', 'reported_model_label', 'metric', 'class_name', 'condition', 'value', 'range_lower', 'range_upper', 'reported', 'precision', 'unit', 'evidence', 'dataset_type', 'source_pdf', 'source_pdf_page', 'source_location', 'notes']
    rows = []
    for record in supplementary['records']:
        values = {**record, 'source_pdf': record['source']['path'], 'source_pdf_page': record['source']['pdf_page'], 'source_location': record['source']['location']}
        if record['precision'] == 'reported_2_decimal_places':
            values['value'] = f"{record['value']:.2f}"
        elif record['precision'] == 'reported_3_decimal_places':
            values['value'] = f"{record['value']:.3f}"
        rows.append([values.get(key) for key in headers])
    outputs['csv/supplementary.csv'] = _csv(headers, rows)
    outputs['source.sha256'] = f"{data['source']['sha256']}  {data['source']['path']}\n"
    return outputs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1], help='Source checkout root containing results/paper/tables.json.')
    parser.add_argument('--source-pdf', type=Path, help='Optional private manuscript location; additionally verify its SHA-256.')
    parser.add_argument('--check', action='store_true', help='Validate and check committed exports without changing files.')
    args = parser.parse_args(argv)
    directory = args.root / 'results' / 'paper'
    try:
        data = json.loads((directory / 'tables.json').read_text(encoding='utf-8'))
        supplementary = json.loads((directory / 'supplementary.json').read_text(encoding='utf-8'))
        validate(data, supplementary, args.source_pdf)
        annotations = json.loads((directory / 'figure_annotations.json').read_text(encoding='utf-8'))
        validate_figure_annotations(annotations, data)
        validate_figure_csv(directory / 'figure_04_robustness_matrix.csv', annotations)
        outputs = exports(data, supplementary)
        stale = []
        for relative, content in outputs.items():
            path = directory / relative
            if args.check:
                if not path.is_file() or path.read_text(encoding='utf-8') != content:
                    stale.append(relative)
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content, encoding='utf-8', newline='')
        if stale:
            parser.exit(1, 'Exports need regeneration: ' + ', '.join(stale) + '\n')
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f'Manuscript export validation failed: {error}\nUse a source checkout with results/paper; the private manuscript is optional.\n')
    print(f"Validated {len(data['tables'])} tables, {len(supplementary['records'])} supplementary records, and {len(annotations['records'])} figure annotations; {'checked' if args.check else 'wrote'} {len(outputs)} exports.")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
