"""
用空格分隔已经清洗过的 wiki 数据每一行的纯汉字文本中的每个字符，生成新的文本文件
"""

import os


def split_hanzi_with_space(line: str) -> str:
    # 假设每行已是“纯汉字串”，直接用空格拼接每个字符
    return " ".join(line.strip())


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    input_path = os.path.join(base_dir, "data/output/all_cleaned_only_wiki_zh.txt")
    output_path = os.path.join(
        base_dir, "data/output/all_cleaned_only_wiki_zh_spaced.txt"
    )

    with open(input_path, "r", encoding="utf-8") as infile, open(
        output_path, "w", encoding="utf-8"
    ) as outfile:
        for line in infile:
            line = line.strip()
            if not line:
                continue
            outfile.write(split_hanzi_with_space(line) + "\n")


if __name__ == "__main__":
    main()
