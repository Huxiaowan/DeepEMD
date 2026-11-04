"""Workflow regressions; GPU/model smoke runs are documented separately."""
from contextlib import redirect_stdout, redirect_stderr
import csv
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import tifffile

from deepemd import runner, scan
from deepemd.completed import create_pdf, validate_result
from deepemd.visualization import TiffStack

ROOT = runner.ROOT
INR = "r9a_inr_baseline_decoder_pe6_noanneal_tilek64"


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.input = self.root / "input with spaces.tif"
        self.volume = np.arange(5 * 64 * 64, dtype=np.uint16).reshape(5, 64, 64)
        tifffile.imwrite(self.input, self.volume, photometric="minisblack", metadata={"axes": "ZYX"})
        self.output = self.root / "result"

    def args(self, *extra):
        return runner.parser().parse_args(["--input", str(self.input), "--output", str(self.output), *extra])

    def prepared_result(self, *extra):
        spec, content, out = runner.prepare(self.args(*extra))
        out.mkdir()
        runner.write_json(out / "run_spec.json", spec)
        (out / "denoised_tiff").mkdir()
        start, end = spec["frame_start"], spec["frame_end"] or len(self.volume)
        tifffile.imwrite(out / "denoised_tiff" / "denoised.tif", self.volume[start:end], photometric="minisblack")
        return spec, out

    def test_frozen_defaults(self):
        digest = "6a1050b5a3ca5b5ad3eabc495d68895fc5aae782f42057941c8073396b941b77"
        self.assertEqual(hashlib.sha256(runner.DEFAULT_CONFIG.read_bytes()).hexdigest(), digest)
        bernoulli = ROOT / "config" / "bern40_ts768_ov050_tilek64.yml"
        self.assertEqual(hashlib.sha256(bernoulli.read_bytes()).hexdigest(),
                         "d774ae3bf5f21ee92f78e5f947e55309fc6b0b9530bc9e4bcd64819cf9ac9b5b")
        self.assertEqual(set((ROOT / "config").glob("*.yml")),
                         {runner.DEFAULT_CONFIG, bernoulli})

    def test_default_is_one_inr_job(self):
        spec, content, out = runner.prepare(self.args())
        self.assertEqual(spec["config_name"], INR)
        cmd = scan.command_for(spec, Path("one.yml"), out)
        self.assertIn("deepemd.inr_engine", cmd)
        self.assertNotIn("deepemd.engine", cmd)
        self.assertEqual(cmd.count("--config"), 1)
        self.assertEqual(content, runner.DEFAULT_CONFIG.read_bytes())

    def test_inr_is_one_independent_job(self):
        spec, _, out = runner.prepare(self.args("--config", str(ROOT / "config" / (INR + ".yml"))))
        self.assertEqual(spec["config"]["num_iter"], 200)
        cmd = scan.command_for(spec, Path("inr.yml"), out)
        self.assertIn("deepemd.inr_engine", cmd)
        self.assertNotIn("deepemd.engine", cmd)

    def test_override_does_not_change_preset(self):
        before = runner.DEFAULT_CONFIG.read_bytes()
        spec, _, _ = runner.prepare(self.args("--tile-size", "256", "--mask-type", "stripe_h", "--mask-ratio", "20"))
        self.assertEqual(spec["config"]["train_mask_ratio"], [20, 20])
        self.assertEqual(spec["config"]["infer_mask_ratio"], [20, 20])
        self.assertEqual(spec["config"]["scan_tile_size"], 256)
        self.assertEqual(runner.DEFAULT_CONFIG.read_bytes(), before)

    def test_dry_run_does_not_touch_real_status(self):
        self.output.mkdir()
        (self.output / "status.csv").write_bytes(b"old status\n")
        with patch("deepemd.runner.input_identity", side_effect=AssertionError("read input")), patch("deepemd.scan.execute", side_effect=AssertionError("ran model")):
            self.assertEqual(runner.run(self.args("--dry-run")), 0)
        self.assertEqual((self.output / "status.csv").read_bytes(), b"old status\n")
        plan = json.loads((self.output / "_dry_run" / "plan.json").read_text())
        self.assertEqual(plan["jobs"], 1)

    def test_config_mismatch_preserves_result(self):
        spec, out = self.prepared_result()
        before = (out / "denoised_tiff" / "denoised.tif").read_bytes()
        with patch("deepemd.scan.execute", side_effect=AssertionError("ran model")), self.assertRaisesRegex(ValueError, "different input/config/range"):
            runner.run(self.args("--tile-size", "256"))
        self.assertEqual((out / "denoised_tiff" / "denoised.tif").read_bytes(), before)

    def test_completed_result_skips_model_even_with_stale_status(self):
        spec, out = self.prepared_result()
        (out / "status.csv").write_text("status\nrunning\n")
        with patch("deepemd.scan.subprocess.Popen", side_effect=AssertionError("ran model")):
            self.assertEqual(scan.execute(spec, ["unused"], out, "4"), 0)
        with (out / "status.csv").open() as handle:
            self.assertEqual(next(csv.DictReader(handle))["status"], "skipped_existing")

    def test_partial_file_is_not_completed(self):
        spec, out = self.prepared_result()
        (out / "denoised_tiff" / "denoised.tif").rename(out / "denoised_tiff" / "denoised_partial.tif")
        with self.assertRaisesRegex(ValueError, "partial TIFFs"):
            validate_result(out, spec, "mid")

    def test_truncated_frame_count_is_rejected(self):
        spec, out = self.prepared_result()
        tifffile.imwrite(out / "denoised_tiff" / "denoised.tif", self.volume[:2], photometric="minisblack")
        with self.assertRaisesRegex(ValueError, "Incomplete TIFF"):
            validate_result(out, spec, "mid")

    def test_source_range_alignment_and_readonly_pdf(self):
        spec, out = self.prepared_result("--frame-start", "1", "--frame-end", "4")
        (out / "status.csv").write_text("status\nrunning\n")
        status = (out / "status.csv").read_bytes()
        before = (out / "denoised_tiff" / "denoised.tif").read_bytes()
        create_pdf(out, frames="first,mid,last", crop_size=24, num_crops=1)
        report = json.loads((out / "completed_comparison.json").read_text())
        self.assertEqual(report["frames"], [1, 2, 3])
        self.assertEqual((out / "status.csv").read_bytes(), status)
        self.assertEqual((out / "denoised_tiff" / "denoised.tif").read_bytes(), before)
        self.assertTrue((out / "completed_comparison.pdf").read_bytes().startswith(b"%PDF"))

    def test_input_contract_rejects_float_and_color(self):
        for data, photo in ((self.volume.astype("float32"), "minisblack"), (np.zeros((64, 64, 3), dtype="uint8"), "rgb")):
            tifffile.imwrite(self.input, data, photometric=photo)
            with self.assertRaises(ValueError):
                runner.prepare(self.args())

    def test_invalid_tiles_and_ranges(self):
        for options in (("--tile-size", "65"), ("--tile-overlap", "1"), ("--window-size", "4"), ("--frame-end", "9")):
            with self.subTest(options=options), self.assertRaises(ValueError):
                runner.prepare(self.args(*options))

    def test_placeholder_gt_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "independent clean reference"):
            runner.prepare(self.args("--gt", str(self.input)))

    def test_single_page_volume_uses_memmap_for_selected_slice(self):
        tifffile.imwrite(self.input, self.volume, photometric="minisblack", volumetric=True,
                         metadata={"axes": "ZYX"})
        with TiffStack(self.input) as stack:
            self.assertEqual(len(stack.tf.pages), 1)
            self.assertEqual(stack.n_frames, 5)
            with patch.object(stack.tf, "asarray", side_effect=AssertionError("decoded entire stack")):
                np.testing.assert_array_equal(stack.read(3), self.volume[3])
            self.assertIsNotNone(stack.mapped)

    def test_failed_process_records_failure_and_preserves_previous_log(self):
        spec, _, out = runner.prepare(self.args())
        out.mkdir()
        (out / "logs").mkdir()
        log = out / "logs" / (INR + ".log")
        log.write_text("previous experiment log")
        with patch("deepemd.scan.subprocess.Popen") as process:
            process.return_value.wait.return_value = 7
            self.assertEqual(scan.execute(spec, ["fake", "one"], out, "4"), 1)
        previous = list(log.parent.glob("*.previous_*.log"))
        self.assertEqual(len(previous), 1)
        self.assertEqual(previous[0].read_text(), "previous experiment log")
        with (out / "status.csv").open() as handle:
            self.assertEqual(next(csv.DictReader(handle))["status"], "failed")

    def test_gpu_list_is_rejected(self):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as exc:
            runner.main(["--input", str(self.input), "--gpu", "4 5 6 7"])
        self.assertEqual(exc.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
