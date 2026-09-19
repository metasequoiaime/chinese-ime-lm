"""Exercise the case-by-case comparison, which is the thing that decides which model ships.

The set arithmetic is small enough to look correct and still be wrong in a way that would quietly
justify the wrong model: recovered and broken are easy to conflate, and a strict superset is easy
to claim when the sets merely differ in size.
"""

import importlib.util
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1] / "neural/compare_models.py"


def load_module():
    spec = importlib.util.spec_from_file_location("compare_models", SOURCE)
    module = importlib.util.module_from_spec(spec)
    sys.modules["compare_models"] = module
    spec.loader.exec_module(module)
    return module


compare_models = load_module()


def write(directory, name, rows):
    path = Path(directory) / f"{name}.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    return str(path)


def case(identifier, *, reachable=True, engine=False, reranked=False, bucket="decoded"):
    return {
        "id": identifier,
        "bucket": bucket,
        "reachable": reachable,
        "engine": engine,
        "reranked": reranked,
    }


class McNemar(unittest.TestCase):
    def test_no_disagreement_is_no_evidence(self):
        self.assertEqual(compare_models.mcnemar_exact(0, 0), 1.0)

    def test_known_values(self):
        # Two-sided exact binomial over the discordant pairs. Hand-checkable: with three
        # disagreements all falling one way the two-sided tail is 2 * (1/8).
        self.assertAlmostEqual(compare_models.mcnemar_exact(3, 0), 0.25)
        self.assertAlmostEqual(compare_models.mcnemar_exact(1, 0), 1.0)
        self.assertAlmostEqual(compare_models.mcnemar_exact(5, 0), 0.0625)
        self.assertAlmostEqual(compare_models.mcnemar_exact(10, 0), 0.001953125)

    def test_symmetric_in_its_arguments(self):
        self.assertEqual(
            compare_models.mcnemar_exact(7, 2), compare_models.mcnemar_exact(2, 7)
        )

    def test_an_even_split_is_maximally_unconvincing(self):
        self.assertEqual(compare_models.mcnemar_exact(4, 4), 1.0)

    def test_never_exceeds_one(self):
        # The doubled tail can cross 1 when the split is near even, and a p-value above 1 would be
        # nonsense that a reader might not notice.
        for wins in range(8):
            for losses in range(8):
                self.assertLessEqual(compare_models.mcnemar_exact(wins, losses), 1.0)


class Reporting(unittest.TestCase):
    def run_compare(self, files, bucket="decoded"):
        argv = sys.argv
        sys.argv = ["compare_models.py", *files, "--bucket", bucket]
        buffer = io.StringIO()
        try:
            with redirect_stdout(buffer):
                compare_models.main()
        finally:
            sys.argv = argv
        return buffer.getvalue()

    def test_recovered_and_broken_are_not_conflated(self):
        """A model that recovers three and breaks two scores +1, and both numbers must show.

        Reporting only the total would make this look identical to a model that recovered one and
        broke nothing, which is not the same product.
        """
        with tempfile.TemporaryDirectory() as directory:
            rows = [
                case("a", engine=True, reranked=True),
                case("b", engine=True, reranked=False),
                case("c", engine=True, reranked=False),
                case("d", engine=False, reranked=True),
                case("e", engine=False, reranked=True),
                case("f", engine=False, reranked=True),
            ]
            output = self.run_compare([write(directory, "noisy", rows)])
        self.assertIn("recovered 3", output)
        self.assertIn("broke 2", output)

    def test_a_clean_model_reports_no_broken_cases(self):
        with tempfile.TemporaryDirectory() as directory:
            rows = [
                case("a", engine=True, reranked=True),
                case("b", engine=False, reranked=True),
            ]
            output = self.run_compare([write(directory, "clean", rows)])
        self.assertIn("recovered 1", output)
        self.assertNotIn("broke", output)

    def test_strict_superset_is_claimed_only_when_it_holds(self):
        with tempfile.TemporaryDirectory() as directory:
            big = [
                case("a", reranked=True),
                case("b", reranked=True),
                case("c", reranked=False),
            ]
            small = [
                case("a", reranked=True),
                case("b", reranked=False),
                case("c", reranked=False),
            ]
            output = self.run_compare(
                [write(directory, "big", big), write(directory, "small", small)]
            )
        self.assertIn("strict superset", output)

        with tempfile.TemporaryDirectory() as directory:
            # Same totals, different cases: neither contains the other, so neither may claim it.
            left = [case("a", reranked=True), case("b", reranked=False)]
            right = [case("a", reranked=False), case("b", reranked=True)]
            output = self.run_compare(
                [write(directory, "left", left), write(directory, "right", right)]
            )
        self.assertNotIn("strict superset", output)

    def test_unreachable_cases_are_separated_from_model_failures(self):
        """Cases whose gold is absent from the candidate list bound what any model can score.

        Counting them as model failures would make every model look worse than it is and would hide
        that the remaining headroom belongs to candidate generation.
        """
        with tempfile.TemporaryDirectory() as directory:
            rows = [
                case("reachable-hit", reachable=True, reranked=True),
                case("reachable-miss", reachable=True, reranked=False),
                case("no-gold", reachable=False, reranked=False),
            ]
            output = self.run_compare([write(directory, "model", rows)])
        self.assertIn("gold present among candidates     2", output)
        self.assertIn("no-gold", output)
        self.assertIn("reachable but missed by every model (1)", output)

    def test_dictionary_hits_are_excluded_by_default(self):
        """They are gated away from the model, so they cannot separate two models."""
        with tempfile.TemporaryDirectory() as directory:
            rows = [
                case("decoded-one", bucket="decoded", reranked=True),
                case("dict-one", bucket="dictionary", engine=True, reranked=True),
            ]
            path = write(directory, "model", rows)
            decoded = self.run_compare([path])
            everything = self.run_compare([path], bucket="all")
        self.assertIn("1 comparable cases", decoded)
        self.assertIn("2 comparable cases", everything)

    def test_files_from_different_dumps_are_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            one = write(directory, "one", [case("a")])
            other = write(directory, "other", [case("z")])
            with self.assertRaises(SystemExit):
                self.run_compare([one, other])


if __name__ == "__main__":
    unittest.main()
