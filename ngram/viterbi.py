import kenlm
import time
import os

model_path = os.path.join(
    os.path.dirname(__file__), "..", "model", "all", "model.binary"
)
model = kenlm.Model(model_path)

pinyin_han_dict = {}
pinyin_han_file_path = os.path.join(
    os.path.dirname(__file__), "..", "data", "output", "pinyin_han_dict_sorted.txt"
)

# 读取拼音 → 汉字
with open(
    pinyin_han_file_path,
    "r",
    encoding="utf-8",
) as f:
    for line in f:
        if line.strip() == "" or line.strip() == "biang":
            continue
        # print(line)
        pinyin, han = line.strip().split("\t")
        han_list = han.split(",")

        # 👉 建议限制候选数量（非常重要）
        pinyin_han_dict[pinyin] = han_list[:30]


def viterbi_no_pruning(candidates, model):
    dp = []

    # 初始化
    dp0 = {}
    for w in candidates[0]:
        score = model.score(w, bos=True, eos=False)
        dp0[w] = (score, None)
    dp.append(dp0)

    # 递推
    for i in range(1, len(candidates)):
        dpi = {}

        for w in candidates[i]:
            best_score = float("-inf")
            best_prev = None

            for prev_w, (prev_score, _) in dp[i - 1].items():
                # 构造前缀句子
                sent = []
                cur = prev_w
                j = i - 1

                while cur is not None:
                    sent.append(cur)
                    _, cur = dp[j][cur]
                    j -= 1

                sent.reverse()
                sent.append(w)

                score = model.score(" ".join(sent), bos=True, eos=False)

                if score > best_score:
                    best_score = score
                    best_prev = prev_w

            dpi[w] = (best_score, best_prev)

        dp.append(dpi)

    # 找最优结尾
    last_layer = dp[-1]
    best_last = max(last_layer.items(), key=lambda x: x[1][0])[0]

    # 回溯
    result = []
    cur = best_last
    for i in reversed(range(len(dp))):
        result.append(cur)
        _, cur = dp[i][cur]
    result.reverse()

    final_score = model.score(" ".join(result), bos=True, eos=True)

    return result, final_score


def viterbi_dp_no_bos_eos(candidates, model):
    dp = []

    # =========================
    # 初始化（第一个词）
    # =========================
    dp0 = {}

    for w in candidates[0]:
        state = kenlm.State()
        model.NullContextWrite(state)  # ⭐ 无 BOS

        next_state = kenlm.State()
        score = model.BaseScore(state, w, next_state)

        dp0[w] = (score, None, next_state)

    dp.append(dp0)

    # =========================
    # 递推
    # =========================
    for i in range(1, len(candidates)):
        dpi = {}

        for w in candidates[i]:
            best_score = float("-inf")
            best_prev = None
            best_state = None

            for prev_w, (prev_score, _, prev_state) in dp[i - 1].items():
                next_state = kenlm.State()

                # ⭐ 核心：增量概率
                score = prev_score + model.BaseScore(prev_state, w, next_state)

                if score > best_score:
                    best_score = score
                    best_prev = prev_w
                    best_state = next_state

            dpi[w] = (best_score, best_prev, best_state)

        dp.append(dpi)

    # =========================
    # 选最优结尾
    # =========================
    last_layer = dp[-1]
    best_last = max(last_layer.items(), key=lambda x: x[1][0])[0]

    # =========================
    # 回溯
    # =========================
    result = []
    cur = best_last

    for i in reversed(range(len(dp))):
        result.append(cur)
        _, prev, _ = dp[i][cur]
        cur = prev

    result.reverse()

    final_score = last_layer[best_last][0]

    return result, final_score


# =========================
# ✅ 新增：拼音输入处理
# =========================


def decode_pinyin(pinyin_input):
    # 支持：wo ai bei jing
    pinyin_list = pinyin_input.strip().split()

    candidates = []

    for py in pinyin_list:
        if py in pinyin_han_dict:
            candidates.append(pinyin_han_dict[py])
        else:
            # 👉 兜底（防止崩）
            candidates.append([py])

    return candidates


# =========================
# ✅ 测试
# =========================

if __name__ == "__main__":

    pinyin_input_list = []
    pinyin_input_list.append("wo ai bei jing")
    pinyin_input_list.append("ni ai bei jing")
    pinyin_input_list.append("chu li chang ju zi shi shen me")
    pinyin_input_list.append("ni shuo ni ma ne")
    pinyin_input_list.append("ke bu shi ma")
    pinyin_input_list.append("na hai yong shuo ma")
    pinyin_input_list.append("jin tian tian qi zen me yang")
    pinyin_input_list.append("wo kan ni shi xiang si le")

    # 你是优势有什么事儿吗
    # 你是有是有什么事儿吗
    pinyin_input_list.append("ni shi you shi you shen me shi er ma")
    pinyin_input_list.append("ru guo nin bu liao jie zhe ge ji chu")
    pinyin_input_list.append("jian yi xian yue du yi xia jiao cheng")
    pinyin_input_list.append("nan jing da xue ren gong zhi neng xue yuan")
    pinyin_input_list.append("bei jing yu yan da xue")
    pinyin_input_list.append("shu yang ren min huan ying nin")
    pinyin_input_list.append("wo shuo ni ma ne")
    pinyin_input_list.append("wo ke qu ni ma de ba")
    pinyin_input_list.append("nan dao shi yu liao de wen ti")
    pinyin_input_list.append("ci pin shu ju ji")
    pinyin_input_list.append("tian qing se deng yan yu")
    pinyin_input_list.append("ming tian shang wu qu tie che mo")
    pinyin_input_list.append("wo zhu shi ni bing leng de shou")
    pinyin_input_list.append("dong ye bu dong rang wo hao nan guo")
    pinyin_input_list.append("ni zai gan shen me a tuan zhang")
    pinyin_input_list.append("bei jing shi yi ge mei li de cheng shi")
    pinyin_input_list.append(
        "ni men que ding zhe xie qi guai de ju zi neng zheng que shu chu"
    )
    pinyin_input_list.append(
        "yi xi jin ping tong zhi wei zong shu ji de dang zhong yang"
    )

    for pinyin_input in pinyin_input_list:
        candidates = decode_pinyin(pinyin_input)
        prev_time = time.time()
        # best_sent, best_score = viterbi_dp_no_bos_eos(candidates, model)
        best_sent, best_score = viterbi_no_pruning(candidates, model)
        cur_time = time.time()
        elsapse_time = (cur_time - prev_time) * 1000
        print(f"花费了 {elsapse_time:.2f} 毫秒")

        print("结果:", "".join(best_sent))
        print("Score:", best_score)
