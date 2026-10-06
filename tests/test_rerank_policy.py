"""Check that evaluation uses the runtime candidate facts."""

import importlib.util
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

from neural import rerank_policy
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

        torch = ModuleType("torch")
        torch.__path__ = []
        torch.backends = SimpleNamespace(
            mps=SimpleNamespace(is_available=lambda: False),
            cuda=SimpleNamespace(is_available=lambda: False),
        )
        torch.cuda = torch.backends.cuda
        torch.no_grad = lambda: lambda function: function
        torch_nn = ModuleType("torch.nn")
        torch_nn.__path__ = []
        safetensors = ModuleType("safetensors")
        safetensors.__path__ = []
        safetensors_torch = ModuleType("safetensors.torch")
        safetensors_torch.load_file = lambda path: None
        model = ModuleType("model")
        model.BOS, model.CharLM, model.Config = 0, object, object
        modules = {
            "torch": torch,
            "torch.nn": torch_nn,
            "torch.nn.functional": ModuleType("torch.nn.functional"),
            "safetensors": safetensors,
            "safetensors.torch": safetensors_torch,
            "model": model,
            "rerank_policy": rerank_policy,
        }
        source = Path(__file__).resolve().parents[1] / "neural/rerank_eval.py"
        spec = importlib.util.spec_from_file_location("rerank_eval_test", source)
        evaluator = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, modules):
            spec.loader.exec_module(evaluator)

        evaluator.load = lambda path, device: (None, None, None)
        case = {"input": "xian", "gold": "西安", "candidates": candidates}
        with tempfile.TemporaryDirectory() as directory:
            cases_path = Path(directory) / "cases.jsonl"
            rows_path = Path(directory) / "rows.jsonl"
            cases_path.write_text(json.dumps(case, ensure_ascii=False) + "\n", encoding="utf-8")
            arguments = ["rerank_eval.py", "--model", "unused", "--cases", str(cases_path), "--per-case", str(rows_path)]
            with patch.object(sys, "argv", arguments), redirect_stdout(io.StringIO()):
                evaluator.main()
            row = json.loads(rows_path.read_text(encoding="utf-8"))

        self.assertEqual(row["bucket"], "decoded")
        self.assertTrue(row["reachable"])
        self.assertFalse(row["engine"])
        self.assertFalse(row["reranked"])

    def test_limit_and_empty_list(self):
        candidates = [{"text": str(index), "source": 9} for index in range(12)]
        selected, _, _ = select_candidates(candidates)
        self.assertEqual([candidate["text"] for candidate in selected], [str(index) for index in range(9)])
        self.assertEqual(select_candidates([]), ([], False, False))


if __name__ == "__main__":
    unittest.main()
