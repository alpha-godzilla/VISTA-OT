import tempfile
import unittest
from pathlib import Path

from eval_data_loader import COCODataSet, read_image_ids_file
from scripts.export_chair_image_ids import read_result_image_ids


class FixedCOCOSubsetTests(unittest.TestCase):
    def test_manifest_order_is_preserved(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "COCO_val2014_000000000042.jpg").touch()
            (root / "COCO_val2014_000000123456.jpg").touch()
            manifest = root / "ids.txt"
            manifest.write_text("123456\n42\n", encoding="utf-8")

            image_ids = read_image_ids_file(manifest)
            dataset = COCODataSet(root, trans=None, image_ids=image_ids)

            self.assertEqual(image_ids, [123456, 42])
            self.assertEqual(
                dataset.img_files,
                [
                    "COCO_val2014_000000123456.jpg",
                    "COCO_val2014_000000000042.jpg",
                ],
            )

    def test_duplicate_ids_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest = Path(temp_dir) / "ids.txt"
            manifest.write_text("42\n42\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "Duplicate COCO image ID"):
                read_image_ids_file(manifest)

    def test_missing_image_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "COCO_val2014_000000000042.jpg").touch()

            with self.assertRaisesRegex(FileNotFoundError, "123456"):
                COCODataSet(root, trans=None, image_ids=[42, 123456])

    def test_export_reads_result_order(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            result = Path(temp_dir) / "result.jsonl"
            result.write_text(
                '{"image_id": 123456, "caption": "first"}\n'
                '{"image_id": 42, "caption": "second"}\n',
                encoding="utf-8",
            )

            self.assertEqual(read_result_image_ids(result), [123456, 42])


if __name__ == "__main__":
    unittest.main()
