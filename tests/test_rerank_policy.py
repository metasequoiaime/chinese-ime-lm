"""Check that evaluation uses the runtime candidate facts."""

import unittest

from neural.rerank_policy import select_candidates


class RerankPolicyTests(unittest.TestCase):
    def test_explicit_coverage_keeps_different_length_answers(self):
        candidates = [
            {"text": "现", "source": 9, "answers_key": True},
            {"text": "先", "source": 9, "answers_key": False},
            {"text": "西安", "source": 9, "answers_key": True},
        ]
        selected, leader_answers_key, trusted = select_candidates(candidates)
        self.assertEqual([candidate["text"] for candidate in selected], ["现", "西安"])
        self.assertTrue(leader_answers_key)
        self.assertFalse(trusted)

    def test_explicit_coverage_excludes_equal_length_other_key(self):
        candidates = [
            {"text": "公司", "source": 9, "answers_key": True},
            {"text": "公事", "source": 9, "answers_key": False},
            {"text": "公式", "source": 9, "answers_key": True},
        ]
        selected, _, _ = select_candidates(candidates)
        self.assertEqual([candidate["text"] for candidate in selected], ["公司", "公式"])

    def test_dictionary_fact_overrides_source(self):
        candidates = [
            {"text": "公司", "source": 0, "trusted_dictionary_hit": False},
            {"text": "公式", "source": 9},
        ]
        _, _, trusted = select_candidates(candidates)
        self.assertFalse(trusted)
        candidates[0]["trusted_dictionary_hit"] = True
        candidates[0]["source"] = 9
        _, _, trusted = select_candidates(candidates)
        self.assertTrue(trusted)

    def test_old_dump_uses_leader_width_and_source(self):
        candidates = [
            {"text": "现", "source": 0},
            {"text": "先", "source": 9},
            {"text": "西安", "source": 9},
        ]
        selected, leader_answers_key, trusted = select_candidates(candidates)
        self.assertEqual([candidate["text"] for candidate in selected], ["现", "先"])
        self.assertTrue(leader_answers_key)
        self.assertTrue(trusted)

    def test_null_facts_use_legacy_fallback(self):
        candidates = [
            {"text": "现", "source": 1, "answers_key": None, "trusted_dictionary_hit": None},
            {"text": "西安", "source": 9, "answers_key": None},
        ]
        selected, leader_answers_key, trusted = select_candidates(candidates)
        self.assertEqual(len(selected), 1)
        self.assertTrue(leader_answers_key)
        self.assertTrue(trusted)

    def test_leader_can_decline_comparison(self):
        candidates = [
            {"text": "现", "source": 9, "answers_key": False},
            {"text": "西安", "source": 9, "answers_key": True},
            {"text": "西岸", "source": 9, "answers_key": True},
        ]
        selected, leader_answers_key, _ = select_candidates(candidates)
        self.assertEqual(len(selected), 2)
        self.assertFalse(leader_answers_key)

    def test_limit_and_empty_list(self):
        candidates = [{"text": str(index), "source": 9} for index in range(12)]
        selected, _, _ = select_candidates(candidates)
        self.assertEqual([candidate["text"] for candidate in selected], [str(index) for index in range(9)])
        self.assertEqual(select_candidates([]), ([], False, False))


if __name__ == "__main__":
    unittest.main()
