"""
替换维基百科文本中的非中文字符为空格，并且将每个汉字的连通块写入新的一行
"""

import regex as re
import os


TOTAL_WRITTEN_LINES = 0


def keep_chinese(text):
    """保留中文字符，其它字符替换为空格"""
    return re.sub(r"\P{Han}", " ", text)


def process_file(input_file, outfile):
    """读取输入文件，处理内容，写入到已打开的输出文件"""
    global TOTAL_WRITTEN_LINES
    with open(input_file, "r", encoding="utf-8") as infile:
        for line in infile:
            # 处理每一行的文本
            cleaned_line = keep_chinese(line)

            # 使用正则表达式按空格切分文本
            words = re.split(r"\s+", cleaned_line.strip())  # \s+ 表示一个或多个空格

            # 将每个词或字符写入新的一行
            for word in words:
                if word:  # 如果是非空词，才写入
                    outfile.write(word + "\n")
                    TOTAL_WRITTEN_LINES += 1
                    if TOTAL_WRITTEN_LINES % 1000000 == 0:
                        print(f"{TOTAL_WRITTEN_LINES // 1000000} 百万行")


def iter_input_files(data_dir):
    """递归收集 data/wiki_zh 下的所有文件"""
    wiki_root = os.path.join(data_dir, "wiki_zh")
    for root, _, files in os.walk(wiki_root):
        for name in files:
            yield os.path.join(root, name)


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_dir = os.path.join(base_dir, "data")
    output_file = os.path.join(base_dir, "data/output/all_cleaned_only_wiki_zh.txt")

    # 确保输出文件的父目录存在，如果不存在则创建
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(output_file, "w", encoding="utf-8") as outfile:
        for input_file in iter_input_files(data_dir):
            process_file(input_file, outfile)


if __name__ == "__main__":
    main()
