"""Focused regression tests for the independent phase-two input gate."""
from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from preflight_independent_phase2_inputs import audit_contract


class IndependentPreflightTest(unittest.TestCase):
    def make_row(self, profile: Path) -> dict[str, object]:
        digest = hashlib.sha256(profile.read_bytes()).hexdigest()
        return {
            "sample_id": "s1", "station_id": "AAA", "analysis_time_utc": "2017-01-01T01:00:00Z", "split": "test_2017",
            "ztd_gnss_mm": 2400.0, "zhd_model_mm": 2200.0, "zwd_gnss_mm": 200.0, "zwd_derivation": "ztd_minus_zhd",
            "gnss_source_id": "gnss_a", "gnss_epoch_id": "g1", "zhd_source_id": "zhd_a", "gnss_available_at_utc": "2017-01-01T00:59:00Z",
            "profile_path": profile.name, "profile_sha256": digest, "profile_source": "phase1_causal", "profile_issue_time_utc": "2017-01-01T01:00:00Z", "profile_max_observation_time_utc": "2017-01-01T00:00:00Z", "p1_model_id": "p1", "p1_model_sha256": "abc",
            "pwv_label_mm": 32.0, "label_time_utc": "2017-01-01T01:00:00Z", "label_source": "EC", "label_source_id": "label_b", "label_epoch_id": "l1", "label_available_at_utc": "2017-01-01T02:00:00Z",
            "input_lineage_root": "gnss_chain", "label_lineage_root": "ec_chain", "match_tolerance_seconds": 1800,
        }

    def test_accepts_independent_causal_sample(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            profile = root / "s1.csv"
            profile.write_text("ELV,TS,PS,WPS\n0,290,1010,18\n100,289,1000,17\n", encoding="utf-8")
            report, errors = audit_contract(pd.DataFrame([self.make_row(profile)]), root, root, {"HOLD"}, {"AAA"}, 0.01)
            self.assertEqual("pass", report["status"])
            self.assertTrue(errors.empty)

    def test_rejects_closed_loop_profile(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            profile = root / "s1.csv"
            profile.write_text("ELV,TS,PS,WPS,PWV\n0,290,1010,18,32\n100,289,1000,17,31\n", encoding="utf-8")
            report, errors = audit_contract(pd.DataFrame([self.make_row(profile)]), root, root, {"HOLD"}, {"AAA"}, 0.01)
            self.assertEqual("fail", report["status"])
            self.assertIn("profile_contains_forbidden_target_columns", errors.iloc[0]["errors"])


if __name__ == "__main__":
    unittest.main()
