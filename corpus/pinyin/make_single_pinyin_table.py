"""
根据已有的 model.arpa 文件，生成单汉字对应拼音的表格，供后续根据拼音查询单汉字使用，以便在输入法中根据拼音生成一个句子来供模型打分。
"""

"""
第一步，把 model.arpa 中的单汉字提取出来，生成 single_char.arpa 文件
"""

import os


input_file_path = os.path.join(
    os.path.dirname(__file__), "..", "..", "model", "all", "model.arpa"
)
output_file_path = os.path.join(
    os.path.dirname(__file__), "..", "..", "tmp", "single_char.arpa"
)
os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

with open(input_file_path, "r", encoding="utf-8") as fin, open(
    output_file_path, "w", encoding="utf-8"
) as fout:
    for i, line in enumerate(fin):
        cur_split_res_cnt = line.strip().split("\t")
        if len(cur_split_res_cnt) < 3:
            continue
        if len(cur_split_res_cnt) == 3 and len(cur_split_res_cnt[1]) == 2:
            break
        fout.write(line)


"""
第二步，把 single_char.arpa 中的单汉字转换为简体，生成 single_char_sc.arpa 文件
"""
import opencc

input_file_path = os.path.join(
    os.path.dirname(__file__), "..", "..", "tmp", "single_char.arpa"
)
output_file_path = os.path.join(
    os.path.dirname(__file__), "..", "..", "tmp", "single_char_sc.arpa"
)
converter = opencc.OpenCC("t2s.json")

han_set = set()

with open(input_file_path, "r", encoding="utf-8") as fin:
    for i, line in enumerate(fin):
        cur_han = line.strip().split("\t")[1]
        cur_han = converter.convert(cur_han)
        if cur_han not in han_set:
            han_set.add(cur_han)

with open(output_file_path, "w", encoding="utf-8") as fout:
    for han in han_set:
        fout.write(han + "\n")

"""
第三步，利用上面生成的 single_char_sc.arpa 文件，查询 cutted_flyciku_with_jp.db 数据库，得到每个单汉字对应的拼音，生成 pinyin_han_dict.txt 文件
"""
import os
import sqlite3

# 声母映射
flyInitialDict = {"sh": "u", "ch": "i", "zh": "v"}

# 韵母映射
flyFinalDict = {
    "iu": "q",
    "ei": "w",
    "e": "e",
    "uan": "r",
    "ue": "t",
    "ve": "t",
    "un": "y",
    "u": "u",
    "i": "i",
    "uo": "o",
    "o": "o",
    "ie": "p",
    "a": "a",
    "ong": "s",
    "iong": "s",
    "ai": "d",
    "en": "f",
    "eng": "g",
    "ang": "h",
    "an": "j",
    "uai": "k",
    "ing": "k",
    "uang": "l",
    "iang": "l",
    "ou": "z",
    "ua": "x",
    "ia": "x",
    "ao": "c",
    "ui": "v",
    "v": "v",
    "in": "b",
    "iao": "n",
    "ian": "m",
}


def cvt_pinyin_to_ul(pinyin: str) -> str:
    """
    把单个拼音转换为小鹤双拼
    """
    if len(pinyin) == 1:
        # 如果是 n、a 这种单声母或者韵母，且，是单字母的拼音
        pinyin = pinyin + pinyin
    elif len(pinyin) == 2:
        # 双字母保持全拼方式，如：li，xi，wu，ng，an
        pinyin = pinyin
    elif len(pinyin) > 2:
        # 如果是 ang 这种单韵母拼音，且为三字母，规则是首字母加韵母所在键
        if pinyin in flyFinalDict.keys():
            pinyin = pinyin[0] + flyFinalDict[pinyin]
        # 如果声母是两个字母
        elif pinyin[:2] in flyInitialDict.keys():
            pinyin = flyInitialDict[pinyin[:2]] + flyFinalDict[pinyin[2:]]
        # 声母是单字母
        else:
            pinyin = pinyin[0] + flyFinalDict[pinyin[1:]]
    return pinyin


pinyin_set = set()
pinyin_file_path = os.path.join(os.path.dirname(__file__), "assets", "pinyin.txt")
with open(
    pinyin_file_path,
    "r",
    encoding="utf-8",
) as fin:
    for line in fin:
        cur_line = line.strip()
        if len(cur_line) > 0:
            if cur_line not in pinyin_set:
                pinyin_set.add(cur_line)


local_app_data_dir = os.environ.get("LOCALAPPDATA")
db_path = os.path.join(
    local_app_data_dir, "metasequoiaime", "cutted_flyciku_with_jp.db"
)


def choose_tbl(sp_str: str) -> str:
    word_len = len(sp_str) // 2
    base_tbl = "tbl_{}_{}"
    return base_tbl.format(word_len if word_len < 8 else "others", sp_str[0])


def query_by_key(key: str):
    tbl_name = choose_tbl(key)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        cursor.execute(
            f"SELECT name FROM sqlite_master WHERE type='table' AND name=?", (tbl_name,)
        )
        if cursor.fetchone() is None:
            print(f"Table {tbl_name} Not Exists")
            return
    except sqlite3.OperationalError as e:
        print(f"Exception when executing sql: {e}")
        return

    try:
        cursor.execute(
            f"SELECT key, jp, value, weight FROM {tbl_name} WHERE key = ? OR key LIKE ?",
            (key, f"{key}%"),
        )
        rows = cursor.fetchall()
        han_list = []
        if not rows:
            print("No results found.")
            return None
        else:
            for row in rows:
                # print(f"拼音: {row[0]}  简拼: {row[1]}  词: {row[2]}  权重: {row[3]}")
                han_list.append(row[2])
            return han_list
    except sqlite3.OperationalError as e:
        print(f"Exception when executing sql: {e}")
    finally:
        conn.close()


single_char_set = set()
single_char_sc_file_path = os.path.join(
    os.path.dirname(__file__), "..", "..", "tmp", "single_char_sc.arpa"
)
with open(single_char_sc_file_path, "r", encoding="utf-8") as fin:
    for line in fin:
        cur_han = line.strip()
        if cur_han not in single_char_set:
            single_char_set.add(cur_han)

pinyin_han_dict = {}
for pinyin in pinyin_set:
    han_list = query_by_key(cvt_pinyin_to_ul(pinyin))
    if han_list:
        new_han_list = []
        for han in han_list:
            if han in single_char_set:
                new_han_list.append(han)
        pinyin_han_dict[pinyin] = new_han_list

pinyin_han_dict_outputpath = os.path.join(
    os.path.dirname(__file__), "..", "..", "tmp", "pinyin_han_dict.txt"
)
with open(pinyin_han_dict_outputpath, "w", encoding="utf-8") as fout:
    for pinyin, han_list in pinyin_han_dict.items():
        fout.write(f"{pinyin}\t{','.join(han_list)}\n")

"""
第四步，排序 pinyin_han_dict.txt 中的每一行，生成 pinyin_han_dict_sorted.txt 文件
"""

pinyin_han_dict_sorted_outputpath = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "data",
    "output",
    "pinyin_han_dict_sorted.txt",
)
with open(pinyin_han_dict_outputpath, "r", encoding="utf-8") as fin, open(
    pinyin_han_dict_sorted_outputpath, "w", encoding="utf-8"
) as fout:
    all_lines = fin.readlines()
    all_lines.sort()
    fout.writelines(all_lines)
