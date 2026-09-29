from __future__ import annotations

import unittest

from futures_strategy.gpt_tuning import (
    DEFAULT_MAJORS_SEARCH_SPACE,
    clamp_and_coerce_value,
    coerce_candidate_params,
    extract_response_text,
    offline_candidate_proposals,
)


class GptTuningHelpersTest(unittest.TestCase):
    def test_clamp_and_coerce_value_respects_bounds(self) -> None:
        spec = DEFAULT_MAJORS_SEARCH_SPACE["htf_adx_threshold"]
        self.assertEqual(clamp_and_coerce_value(99, spec, 18), 26.0)
        self.assertEqual(clamp_and_coerce_value(11.4, spec, 18), 12.0)

    def test_coerce_candidate_params_enforces_relationships(self) -> None:
        base = {
            "htf_adx_threshold": 18.0,
            "atr_stop_mult": 1.2,
            "trail_atr_mult": 1.8,
            "partial_rr": 1.2,
            "majors_impulse_threshold": 0.01,
            "majors_pullback_atr_tolerance": 0.35,
            "majors_min_retrace_from_extreme": 0.002,
            "majors_max_retrace_from_extreme": 0.012,
            "majors_vwap_extension_cap": 0.01,
            "majors_volume_ratio_threshold": 0.9,
            "majors_entry_rsi_low": 48.0,
            "majors_entry_rsi_high": 62.0,
            "majors_exit_rsi": 45.0,
        }
        candidate = coerce_candidate_params(
            {
                "atr_stop_mult": 1.5,
                "trail_atr_mult": 1.2,
                "majors_entry_rsi_low": 55,
                "majors_entry_rsi_high": 50,
                "majors_min_retrace_from_extreme": 0.01,
                "majors_max_retrace_from_extreme": 0.005,
            },
            base,
            DEFAULT_MAJORS_SEARCH_SPACE,
        )
        self.assertGreaterEqual(candidate["trail_atr_mult"], candidate["atr_stop_mult"])
        self.assertGreater(candidate["majors_entry_rsi_high"], candidate["majors_entry_rsi_low"])
        self.assertGreater(candidate["majors_max_retrace_from_extreme"], candidate["majors_min_retrace_from_extreme"])

    def test_extract_response_text_reads_nested_content(self) -> None:
        payload = {
            "output": [
                {
                    "type": "message",
                    "content": [
                        {
                            "type": "output_text",
                            "text": '{"candidates":[]}',
                        }
                    ],
                }
            ]
        }
        self.assertEqual(extract_response_text(payload), '{"candidates":[]}')

    def test_offline_candidate_proposals_are_unique(self) -> None:
        base = {
            "htf_adx_threshold": 18.0,
            "atr_stop_mult": 1.2,
            "trail_atr_mult": 1.8,
            "partial_rr": 1.2,
            "majors_impulse_threshold": 0.01,
            "majors_pullback_atr_tolerance": 0.35,
            "majors_min_retrace_from_extreme": 0.002,
            "majors_max_retrace_from_extreme": 0.012,
            "majors_vwap_extension_cap": 0.01,
            "majors_volume_ratio_threshold": 0.9,
            "majors_entry_rsi_low": 48.0,
            "majors_entry_rsi_high": 62.0,
            "majors_exit_rsi": 45.0,
        }
        proposals = offline_candidate_proposals(
            base_params=base,
            history=[],
            search_space=DEFAULT_MAJORS_SEARCH_SPACE,
            candidates_per_round=5,
            round_index=1,
            seed=7,
        )
        self.assertEqual(len(proposals), 5)
        signatures = {str(proposal["params"]) for proposal in proposals}
        self.assertEqual(len(signatures), len(proposals))


if __name__ == "__main__":
    unittest.main()
