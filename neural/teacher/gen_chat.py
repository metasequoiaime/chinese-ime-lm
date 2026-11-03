"""Generate one-sided chat conversations with a large language model, as training text for the chat cases.

The keystroke model trains on LCCC, open-domain exchanges from social media, where a turn rarely follows the same person's earlier turns, and it never saw the shape a chat input method sees: one person's messages, one after another, to a parent, a colleague or a friend. Here a model writes such conversations: only the sender's side, 8 to 14 messages, everyday Mandarin, with the people, pets, times and plans that later messages refer back to.

Every conversation is drawn from a seeded combination of a sender, a recipient and a topic, so the set covers family, work and friends evenly and a rerun asks for the same ones. Output is one JSON object per conversation (`id`, `sender`, `recipient`, `topic`, `model`, `messages`); a run that stops continues after the ids it has written.

The model is DeepSeek's (`deepseek-v4-pro` through EveryAPI): its terms of use assign the outputs to the user and allow using them to train other models, distillation included (section 4.2, checked 2026-11-03). Any model card that uses this text says it is synthetic.

usage:
  python neural/teacher/gen_chat.py --headers ~/.everyapi-promo-headers --count 3000 --out <data>/synth/chat.jsonl [--jobs 6]
"""

# ruff: noqa: RUF001 (the prompt and the sender list are Chinese prose, full-width punctuation included)
import argparse
import http.client
import json
import random
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

SEED = "chat-synth-20261103"
ENDPOINT = "https://api.everyapi.ai/v1/chat/completions"
MODEL = "deepseek-v4-pro"

SENDERS = [
    "在杭州做产品经理的三十岁男生",
    "在成都做会计的三十多岁女性",
    "在北京读研的女生",
    "在深圳做程序员的男生",
    "在上海做设计的女生",
    "在武汉当中学老师的女性",
    "在广州开小饭馆的中年男人",
    "退休在家的六十岁阿姨",
    "在南京做销售的男生",
    "刚毕业在西安找工作的男生",
    "在长沙做护士的女生",
    "带两个孩子的全职妈妈",
    "在苏州工厂做技术员的男生",
    "在厦门做外贸的女生",
    "在重庆跑外卖的小伙子",
    "在天津做律师助理的女生",
    "大学刚入学的新生",
    "在郑州做公务员的男性",
    "在青岛做健身教练的男生",
    "在昆明开民宿的夫妻中的妻子",
]
RECIPIENTS = [
    ("妈妈", "family"),
    ("爸爸", "family"),
    ("姐姐", "family"),
    ("弟弟", "family"),
    ("老公", "family"),
    ("老婆", "family"),
    ("奶奶", "family"),
    ("孩子的班主任", "school"),
    ("直属领导", "work"),
    ("同组的同事", "work"),
    ("合作方的对接人", "work"),
    ("新来的实习生", "work"),
    ("大学室友", "friend"),
    ("多年的好朋友", "friend"),
    ("一起打球的朋友", "friend"),
    ("房东", "life"),
    ("装修师傅", "life"),
    ("快递小哥", "life"),
]
TOPICS = {
    "family": [
        "周末回家吃饭",
        "家里谁过生日",
        "老人身体和看病",
        "孩子上学和考试",
        "宠物生病",
        "家里装修和搬家",
        "过年回家的车票",
        "给家里买东西",
    ],
    "school": ["孩子请假", "家长会", "作业和成绩", "学校活动"],
    "work": ["项目上线和排期", "开会和汇报", "出差安排", "报销和合同", "需求变更", "周报和总结", "请假调休"],
    "friend": ["约饭约玩", "追剧和看演出", "旅游计划", "吐槽工作", "借东西还东西", "生日聚会"],
    "life": ["交房租和水电", "修东西", "收发快递", "预约上门时间"],
}

PROMPT = """写一段微信聊天记录，只写其中一个人发出的消息，对方的回复一律不写。
发消息的人：{sender}。聊天对象：{recipient}。话题：{topic}。
写 8 到 14 条消息，一条一行，像真人在手机上随手打字：长短不一，可以省主语，可以带“啊、吧、呢、哈、嘛、了”这类语气词；用普通话，不用方言词；不要表情符号、不要英文字母、不要阿拉伯数字（数字用汉字写）。
前面提到的人和事，后面能自然指回去的就用“他、她、那个、这事”指回去，只有真的养了宠物才写宠物；要有具体的时间（比如下周四、月底、明天下午）、地点和安排。别写成作文，别每条都很完整。
只输出 JSON：{{"messages": ["...", "..."]}}"""


def plan(count):
    """The seeded (sender, recipient, topic) of every conversation, recipients cycling so that each kind of chat gets its share."""
    rng = random.Random(SEED)
    out = []
    for n in range(count):
        recipient, kind = RECIPIENTS[n % len(RECIPIENTS)]
        out.append(
            {
                "id": f"synth-{n:05d}",
                "sender": rng.choice(SENDERS),
                "recipient": recipient,
                "topic": rng.choice(TOPICS[kind]),
            }
        )
    return out


def headers(path):
    out = {"User-Agent": "chinese-ime-lm-gen-chat/1.0", "Content-Type": "application/json"}
    for line in Path(path).expanduser().read_text(encoding="utf-8").splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            out[key.strip()] = value.strip()
    return out


HAN = re.compile(r"[一-鿿]")
LATIN_OR_DIGIT = re.compile(r"[A-Za-z0-9]")


def clean(messages):
    """The messages worth keeping: strings with Chinese in them and no Latin letters or digits, which pinyin cannot type the way the gold reads."""
    kept = []
    for message in messages:
        if not isinstance(message, str):
            continue
        message = message.strip()
        if message and HAN.search(message) and not LATIN_OR_DIGIT.search(message) and "\n" not in message:
            kept.append(message)
    return kept


def generate(item, hdr, attempts=6):
    body = json.dumps(
        {
            "model": MODEL,
            "messages": [{"role": "user", "content": PROMPT.format(**item)}],
            "temperature": 1.0,
            "max_tokens": 6000,
        },
        ensure_ascii=False,
    ).encode()
    for attempt in range(attempts):
        try:
            request = urllib.request.Request(ENDPOINT, data=body, headers=hdr)
            with urllib.request.urlopen(request, timeout=300) as response:
                reply = json.load(response)
            text = reply["choices"][0]["message"]["content"]
            start, end = text.find("{"), text.rfind("}")
            messages = clean(json.loads(text[start : end + 1]).get("messages", []))
            if len(messages) >= 6:
                return {**item, "model": MODEL, "messages": messages}
        except (
            urllib.error.URLError,
            http.client.HTTPException,
            ConnectionError,
            json.JSONDecodeError,
            KeyError,
            ValueError,
            TimeoutError,
        ) as error:
            print(f"{item['id']}: attempt {attempt + 1}: {type(error).__name__}: {str(error)[:120]}", file=sys.stderr)
            # The gateway rate-limits a key; back off longer when it says so.
            if isinstance(error, urllib.error.HTTPError) and error.code == 429:
                time.sleep(30 * (attempt + 1))
                continue
        time.sleep(5 * (attempt + 1))
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--headers", required=True, help="a file of HTTP headers, one per line, carrying the EveryAPI key"
    )
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--jobs", type=int, default=6)
    args = parser.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        done = {json.loads(line)["id"] for line in out.read_text(encoding="utf-8").splitlines() if line.strip()}
    todo = [item for item in plan(args.count) if item["id"] not in done]
    hdr = headers(args.headers)
    lock = threading.Lock()
    started, written, failed = time.time(), 0, 0
    print(f"{len(done)} written before, {len(todo)} to go", file=sys.stderr)
    with ThreadPoolExecutor(args.jobs) as pool, open(out, "a", encoding="utf-8") as handle:
        for future in as_completed([pool.submit(generate, item, hdr) for item in todo]):
            row = future.result()
            with lock:
                if row is None:
                    failed += 1
                else:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    handle.flush()
                    written += 1
                if (written + failed) % 20 == 0:
                    print(
                        f"{written + len(done)}/{args.count} written, {failed} failed, {time.time() - started:.0f} s",
                        file=sys.stderr,
                    )
    print(
        f"{written + len(done)}/{args.count} written, {failed} failed, {time.time() - started:.0f} s", file=sys.stderr
    )


if __name__ == "__main__":
    main()
