"""Exercise actual preprocessing functions without downloading a private/large corpus."""
import importlib.util
import io
import shutil
import subprocess
import sys
from pathlib import Path
import tempfile
import unittest

SOURCE = Path(__file__).resolve().parents[1] / "corpus/clean"


def load(name):
    spec = importlib.util.spec_from_file_location(name, SOURCE / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PreprocessingTests(unittest.TestCase):
    def test_clean_and_segment(self):
        for name in ("extract_connected_hanzi_components", "extract_wiki_connected_hanzi_components"):
            with self.subTest(module=name), tempfile.TemporaryDirectory() as directory:
                source = Path(directory) / "input.txt"
                source.write_text("水杉IME，输入法123！\nABC\n𠀀中文\n", encoding="utf-8")
                output = io.StringIO()
                load(name).process_file(source, output)
                self.assertEqual(output.getvalue(), "水杉\n输入法\n𠀀中文\n")

    def test_character_spacing(self):
        for name in ("split_all_cleaned_txt_using_space", "split_wiki_txt_using_space"):
            module = load(name)
            self.assertEqual(module.split_hanzi_with_space(" 水杉输入法\n"), "水 杉 输 入 法")
            self.assertEqual(module.split_hanzi_with_space("𠀀中文"), "𠀀 中 文")
            self.assertEqual(module.split_hanzi_with_space("\n"), "")

    def test_command_line_pipeline_uses_repository_data(self):
        with tempfile.TemporaryDirectory() as directory:
            checkout = Path(directory) / "checkout"
            scripts = checkout / "corpus/clean"
            shutil.copytree(SOURCE, scripts)
            data = checkout / "data"
            (data / "wiki_zh").mkdir(parents=True)
            (data / "wiki_zh/article.json").write_text('{"text": "水杉输入法123。"}\n', encoding="utf-8")
            (data / "news_train.json").write_text('{"text": "合成语料。"}\n', encoding="utf-8")
            for script in (
                "extract_connected_hanzi_components.py",
                "extract_wiki_connected_hanzi_components.py",
                "split_all_cleaned_txt_using_space.py",
                "split_wiki_txt_using_space.py",
            ):
                result = subprocess.run([sys.executable, str(scripts / script)],
                                        cwd=directory, capture_output=True, text=True, timeout=20)
                self.assertEqual(result.returncode, 0, result.stderr)
            expected = {
                "all_cleaned.txt": "水杉输入法\n合成语料\n",
                "all_cleaned_only_wiki_zh.txt": "水杉输入法\n",
                "all_cleaned_spaced.txt": "水 杉 输 入 法\n合 成 语 料\n",
                "all_cleaned_only_wiki_zh_spaced.txt": "水 杉 输 入 法\n",
            }
            for filename, content in expected.items():
                output = data / "output" / filename
                self.assertTrue(output.is_file(), f"Pipeline did not create {output}")
                self.assertEqual(output.read_text(encoding="utf-8"), content)

    def test_corpus_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            (base / "wiki_zh/sub").mkdir(parents=True)
            wiki = base / "wiki_zh/sub/article.json"
            train = base / "news_train.json"
            for path in (wiki, train, base / "unrelated.json"):
                path.touch()
            all_inputs = {Path(p) for p in load("extract_connected_hanzi_components").iter_input_files(base)}
            wiki_inputs = {Path(p) for p in load("extract_wiki_connected_hanzi_components").iter_input_files(base)}
            self.assertEqual(all_inputs, {wiki, train})
            self.assertEqual(wiki_inputs, {wiki})


if __name__ == "__main__":
    unittest.main()
