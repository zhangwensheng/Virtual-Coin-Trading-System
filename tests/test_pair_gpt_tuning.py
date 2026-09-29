from __future__ import annotations

import unittest

from futures_strategy.pair_gpt_tuning import (
    DEFAULT_PAIR_SEARCH_SPACE,
    clamp_and_coerce_value,
    coerce_candidate_params,
    extract_response_text,
    offline_candidate_proposals,
)


class PairGptTuningHelpersTest(unittest.TestCase):
    def test_clamp_and_coerce_value_for_int(self) -> None:
        spec = DEFAULT_PAIR_SEARCH_SPACE["beta_window"]
        self.assertEqual(clamp_and_coerce_value(999, spec, 576), 720)
        self.assertEqual(clamp_and_coerce_value(10, spec, 576), 288)

    def test_coerce_candidate_params_enforces_ordering(self) -> None:
        base = {
            "beta_window": 576.0,
            "zscore_window": 576.0,
            "entry_z": 2.2,
            "exit_z": 0.2,
            "stop_z": 4.5,
            "min_correlation": 0.9,
            "correlation_exit_buffer": 0.1,
            "max_holding_bars": 96.0,
        }
        candidate = coerce_candidate_params(
            {
                "entry_z": 2.0,
                "exit_z": 2.4,
                "stop_z": 1.0,
                "min_correlation": 0.81,
                "correlation_exit_buffer": 0.95,
            },
            base,
            DEFAULT_PAIR_SEARCH_SPACE,
        )
        self.assertLess(candidate["exit_z"], candidate["entry_z"])
        self.assertGreater(candidate["stop_z"], candidate["entry_z"])
        self.assertLess(candidate["correlation_exit_buffer"], candidate["min_correlation"])

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
            "beta_window": 576.0,
            "zscore_window": 576.0,
            "entry_z": 2.2,
            "exit_z": 0.2,
            "stop_z": 4.5,
            "min_correlation": 0.9,
            "correlation_exit_buffer": 0.1,
            "max_holding_bars": 96.0,
        }
        proposals = offline_candidate_proposals(
            base_params=base,
            history=[],
            search_space=DEFAULT_PAIR_SEARCH_SPACE,
            candidates_per_round=4,
            round_index=1,
            seed=7,
        )
        self.assertEqual(len(proposals), 4)
        signatures = {str(proposal["params"]) for proposal in proposals}
        self.assertEqual(len(signatures), len(proposals))


if __name__ == "__main__":
    unittest.main()
