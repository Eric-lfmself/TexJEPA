"""Check source provenance and prevent accidental changes to historical results."""
from __future__ import annotations

import copy
import importlib.util
import hashlib
import shutil
import tempfile
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('paper_results_export', ROOT / 'scripts' / 'export_paper_results.py')
EXPORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXPORT)


class PaperResultsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = json.loads((ROOT / 'results/paper/tables.json').read_text())
        cls.supplementary = json.loads((ROOT / 'results/paper/supplementary.json').read_text())

    def test_source_and_all_tables_validate(self):
        EXPORT.validate(self.data, self.supplementary)
        self.assertEqual(sum(len(t['rows']) for t in self.data['tables']), 61)

    def test_exports_match_committed_files(self):
        for relative, expected in EXPORT.exports(self.data, self.supplementary).items():
            self.assertEqual((ROOT / 'results/paper' / relative).read_text(), expected, relative)

    def test_optional_private_source_with_matching_fingerprint_validates(self):
        payload = b'%PDF-1.4\nprivate manuscript fixture\n%%EOF\n'
        with tempfile.TemporaryDirectory() as directory:
            pdf = Path(directory) / 'manuscript.pdf'
            pdf.write_bytes(payload)
            data = copy.deepcopy(self.data)
            supplementary = copy.deepcopy(self.supplementary)
            data['source']['sha256'] = hashlib.sha256(payload).hexdigest()
            supplementary['source']['sha256'] = data['source']['sha256']
            EXPORT.validate(data, supplementary, pdf)

    def test_optional_source_fingerprint_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            pdf = Path(directory) / 'different.pdf'
            pdf.write_bytes(b'%PDF-1.4\ndifferent manuscript\n%%EOF\n')
            with self.assertRaisesRegex(ValueError, 'SHA-256'):
                EXPORT.validate(self.data, self.supplementary, pdf)

    def test_explicit_missing_private_source_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, 'Manuscript not found'):
                EXPORT.validate(self.data, self.supplementary, Path(directory) / 'missing.pdf')

    def test_invalid_source_fingerprint_is_rejected_without_pdf(self):
        data = copy.deepcopy(self.data)
        data['source']['sha256'] = 'invalid'
        with self.assertRaisesRegex(ValueError, 'SHA-256'):
            EXPORT.validate(data, self.supplementary)

    def test_manuscript_provenance_is_preserved_without_a_public_file(self):
        source = self.data['source']
        self.assertEqual(source['path'], 'texjepa-manuscript')
        self.assertEqual(source['availability'], 'not_distributed')
        self.assertEqual(source['sha256'], 'ca4a03a5f94dbb5b60396f76834d68d54b39eb760e73ef2d02c913dd74dc9ab7')
        self.assertEqual(source['pdf_pages'], 13)

    def test_check_succeeds_in_checkout_without_any_pdf(self):
        with tempfile.TemporaryDirectory() as directory:
            checkout = Path(directory)
            shutil.copytree(ROOT / 'results/paper', checkout / 'results/paper', ignore=shutil.ignore_patterns('*.pdf'))
            self.assertEqual(list(checkout.rglob('*.pdf')), [])
            self.assertEqual(EXPORT.main(['--root', str(checkout), '--check']), 0)

    def test_check_rejects_stale_table_csv_without_pdf(self):
        with tempfile.TemporaryDirectory() as directory:
            checkout = Path(directory)
            shutil.copytree(ROOT / 'results/paper', checkout / 'results/paper', ignore=shutil.ignore_patterns('*.pdf'))
            (checkout / 'results/paper/csv/table_iv.csv').write_text('stale\n')
            with self.assertRaises(SystemExit) as error:
                EXPORT.main(['--root', str(checkout), '--check'])
            self.assertEqual(error.exception.code, 1)

    def test_missing_drift_cannot_be_replaced_with_zero(self):
        data = copy.deepcopy(self.data)
        table = next(t for t in data['tables'] if t['id'] == 'table_vii')
        table['rows'][1]['drift_sigma_0_05'] = 0.0
        table['reported_rows'][1][-1] = '0.000'
        with self.assertRaisesRegex(ValueError, 'null cells'):
            EXPORT.validate(data, self.supplementary)

    def test_unreported_extra_precision_is_rejected(self):
        data = copy.deepcopy(self.data)
        table = next(t for t in data['tables'] if t['id'] == 'table_iv')
        table['rows'][0]['clean'] = 0.9101
        with self.assertRaisesRegex(ValueError, 'precision'):
            EXPORT.validate(data, self.supplementary)

    def test_table_vi_retains_printed_difference(self):
        table = next(t for t in self.data['tables'] if t['id'] == 'table_vi')
        self.assertEqual(table['rows'][3]['drop'], -0.317)
        self.assertEqual(table['reported_rows'][3][2], '−0.317')
        self.assertIsNone(table['rows'][0]['drift'])

    def test_figure_seven_retains_signed_labels_and_precision(self):
        records = [r for r in self.supplementary['records'] if r['id'].startswith('fig7_')]
        self.assertEqual(len(records), 16)
        ild = next(r for r in records if r['class_name'] == 'ILD' and r['model'] == 'MAE-H/300')
        self.assertEqual((ild['value'], ild['reported']), (-0.09, '-0.09'))
        self.assertTrue(all(r['precision'] == 'reported_2_decimal_places' for r in records))

    def test_approximate_range_has_no_invented_point(self):
        record = next(r for r in self.supplementary['records'] if r['id'] == 'fig8_smoothing_auroc_range')
        self.assertIsNone(record['value'])
        self.assertEqual((record['range_lower'], record['range_upper']), (0.73, 0.75))
        supplementary = copy.deepcopy(self.supplementary)
        changed = next(r for r in supplementary['records'] if r['id'] == record['id'])
        changed['value'] = 0.74
        with self.assertRaisesRegex(ValueError, 'inferred point'):
            EXPORT.validate(self.data, supplementary)


if __name__ == '__main__':
    unittest.main()
