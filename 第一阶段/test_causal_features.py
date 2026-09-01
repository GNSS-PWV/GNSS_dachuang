"""Regression tests for the no-current-value causal feature contract."""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from causal_features import make_causal_inference_sequences, make_causal_sequences


class CausalInferenceSequencesTest(unittest.TestCase):
    def frame(self) -> pd.DataFrame:
        n = 70
        time = pd.date_range("2017-01-01", periods=n, freq="h")
        return pd.DataFrame({
            "TIME": time, "YEAR": time.year, "DOY": time.dayofyear,
            "LAT": 31.2, "LON": 121.5, "ELV": 10.0, "station_id": "TEST",
            "PS": np.linspace(1000.0, 1004.0, n), "WPS": np.linspace(12.0, 16.0, n),
            "TS": np.linspace(280.0, 284.0, n), "Tm": np.linspace(275.0, 279.0, n),
        })

    def test_inference_does_not_require_current_target(self) -> None:
        frame = self.frame()
        frame.loc[50, "PS"] = np.nan
        x, meta, features = make_causal_inference_sequences(frame, time_steps=24)
        self.assertGreater(len(x), 0)
        self.assertEqual(len(x), len(meta))
        self.assertNotIn("PS", features)
        self.assertIn(frame.loc[50, "TIME"], set(meta["TIME"]))

    def test_inference_alignment_matches_supervised_sequences(self) -> None:
        frame = self.frame()
        supervised_x, _, supervised_meta, features = make_causal_sequences(frame, target="PS", time_steps=24)
        inference_x, inference_meta, inference_features = make_causal_inference_sequences(frame, time_steps=24)
        np.testing.assert_allclose(supervised_x, inference_x)
        self.assertEqual(features, inference_features)
        self.assertEqual(supervised_meta[["TIME", "ELV"]].to_dict("records"), inference_meta[["TIME", "ELV"]].to_dict("records"))

    def test_missing_history_breaks_windows(self) -> None:
        frame = self.frame()
        frame.loc[40, "PS"] = np.nan
        _, meta, _ = make_causal_inference_sequences(frame, time_steps=24)
        forbidden = frame.loc[41:44, "TIME"].tolist()
        self.assertTrue(set(forbidden).isdisjoint(set(meta["TIME"])))


if __name__ == "__main__":
    unittest.main()
