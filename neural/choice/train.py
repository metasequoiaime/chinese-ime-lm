"""P2 of `docs/choice-model.md`: train the choice model, in two phases.

- `--phase lm`: pretrain the decoder as a character LM on a packed corpus (`neural/train.py`'s `tokens.u16` and `vocab.json`, so the 2.0 corpus is reused as it is), so it knows Chinese before it sees a pool.
- `--phase pools`: train on candidate pools, `--init` from the LM phase. Each pool's candidates are scored and the page is ranked with a listwise loss: cross-entropy on the gold and, for each `--teacher` that scored the pool, `w T^2 KL(teacher || student)` at that teacher's weight and temperature (the 9B LoRA and Zhiyi, whose scores are on different scales). A little LM loss on corpus windows (`--lm-weight`) keeps the language from drifting.
- `--lm-term` (pools phase) turns on the model's `lm_term`: each candidate's score also sums its characters' log-likelihood through the tied table, the gold candidate's characters are trained as a conditioned LM (`--candidate-lm-weight`), and every LM loss carries the self-normalisation penalty `--z` so the keystroke path can skip the softmax.

The context is the last `CONTEXT_CHARS` characters of the case's context, the letters are what was typed (`input`), as at inference. The dev pools pick the highest score, the first on a tie; the checkpoint with the most dev hits is kept.

usage:
  python neural/choice/train.py --phase lm --tokens <run>/tokens.u16 --vocab <run>/vocab.json --shape 8x384 --steps 20000 --out <dir>
  python neural/choice/train.py --phase pools --init <dir>/checkpoint.pt --tokens ... --vocab ... --pools <train.jsonl> --teacher 9b=<teacher.jsonl>:1 --teacher zhiyi=<zhiyi.jsonl>:0.5 --dev <dev-select.jsonl> --epochs 2 --out <dir2>
"""

import argparse
import dataclasses
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import END, PAD, SEGMENTS, UNK, ChoiceConfig, ChoiceModel, letter_ids  # noqa: E402

CONTEXT_CHARS = 64
LETTERS_MAX = 96


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def learning_rate(step, total, peak, warmup):
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return 0.1 * peak + 0.45 * peak * (1 + math.cos(math.pi * progress))


def lm_windows(tokens, batch, length, device, generator):
    while True:
        starts = torch.randint(0, len(tokens) - length - 1, (batch,), generator=generator)
        rows = np.stack([tokens[s : s + length + 1].astype(np.int64) for s in starts.tolist()])
        rows = torch.from_numpy(rows)
        x, y = rows[:, :-1], rows[:, 1:].clone()
        # The corpus's document separator (BOS, id 2 in neural/train.py's vocabulary) is END here; it stays as input but is never a target.
        y[y == END] = PAD
        yield x.to(device), y.to(device)


class Encoder:
    def __init__(self, tokens):
        self.ids = {t: i for i, t in enumerate(tokens) if len(t) == 1}

    def chars(self, text):
        return [self.ids.get(c, UNK) for c in text]

    def rows(self, case):
        """One row per candidate: (chars, letters, segments, end)."""
        context = self.chars(case.get("context", "")[-CONTEXT_CHARS:])
        letters = letter_ids(case["input"])[:LETTERS_MAX]
        out = []
        for candidate in case["candidates"]:
            body = self.chars(candidate) + [END]
            chars = context + [PAD] * len(letters) + body
            lets = [0] * len(context) + letters + [0] * len(body)
            segs = [SEGMENTS["context"]] * len(context) + [SEGMENTS["letters"]] * len(letters) + [SEGMENTS["candidate"]] * len(body)
            out.append((chars, lets, segs, len(chars) - 1))
        return out


def batch_rows(encoder, cases, device):
    rows, spans = [], []
    for case in cases:
        made = encoder.rows(case)
        spans.append((len(rows), len(rows) + len(made)))
        rows.extend(made)
    # Widths rounded up to a multiple of 32, so MPS reuses its cached buffers (see neural/teacher/distill.py).
    width = -(-max(len(r[0]) for r in rows) // 32) * 32
    chars = torch.full((len(rows), width), PAD, dtype=torch.long)
    lets = torch.zeros((len(rows), width), dtype=torch.long)
    segs = torch.full((len(rows), width), SEGMENTS["candidate"], dtype=torch.long)
    ends = torch.tensor([r[3] for r in rows], dtype=torch.long)
    for i, (c, l, s, _) in enumerate(rows):
        chars[i, : len(c)] = torch.tensor(c)
        lets[i, : len(l)] = torch.tensor(l)
        segs[i, : len(s)] = torch.tensor(s)
    return chars.to(device), lets.to(device), segs.to(device), ends.to(device), spans


def parse_teacher(spec):
    """`name=path[:weight[:temperature]]`."""
    name, rest = spec.split("=", 1)
    parts = rest.split(":")
    return {"name": name, "path": parts[0], "weight": float(parts[1]) if len(parts) > 1 else 1.0, "temperature": float(parts[2]) if len(parts) > 2 else 1.0}


def page_loss(scores, spans, cases, teachers):
    """Cross-entropy on the gold, and for each teacher that scored the pool `w T^2 KL(teacher || student)` at its own temperature, averaged over the pools it scored. Returns (gold, {teacher name: kd})."""
    gold_terms = []
    kd_terms = {t["name"]: [] for t in teachers}
    for case, (a, b) in zip(cases, spans):
        s = scores[a:b]
        gold_terms.append(-F.log_softmax(s, dim=0)[case["candidates"].index(case["gold"])])
        for teacher in teachers:
            t = teacher["scores"].get(case["id"])
            if t is None or len(t) != b - a:
                continue
            t = torch.tensor(t, dtype=torch.float32, device=s.device)
            temp = teacher["temperature"]
            kd_terms[teacher["name"]].append(temp * temp * F.kl_div(F.log_softmax(s / temp, dim=0), F.log_softmax(t / temp, dim=0), log_target=True, reduction="sum"))
    gold = torch.stack(gold_terms).mean()
    kd = {name: torch.stack(terms).mean() if terms else scores.new_zeros(()) for name, terms in kd_terms.items()}
    return gold, kd


@torch.no_grad()
def evaluate(model, encoder, cases, device, batch=64):
    model.eval()
    hits = 0
    for i in range(0, len(cases), batch):
        chunk = cases[i : i + batch]
        chars, lets, segs, ends, spans = batch_rows(encoder, chunk, device)
        scores = model.rows(chars, lets, segs, ends).float().cpu().tolist()
        for case, (a, b) in zip(chunk, spans):
            row = scores[a:b]
            hits += case["candidates"][row.index(max(row))] == case["gold"]
    model.train()
    return hits


def report(path, key, step, steps, elapsed, detail):
    """One stage of the dashboard's progress file (`teacher/build_corpus.py`'s format), written atomically."""
    if not path:
        return
    path = Path(path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError):
        data = {}
    per = elapsed / max(1, step)
    data[key] = {"status": "running", "progress": step / steps, "detail": detail, "eta_s": (steps - step) * per, "live": {"last_it": step, "total": steps, "sec_per_it": per, "log_at": time.time()}}
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def read_jsonl(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def save(path, model, step, extra):
    torch.save({"config": dataclasses.asdict(model.cfg), "model": model.state_dict(), "step": step, **extra}, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--phase", choices=["lm", "pools"], required=True)
    parser.add_argument("--tokens", required=True, help="a packed corpus, neural/train.py's tokens.u16")
    parser.add_argument("--vocab", required=True, help="the vocab.json that goes with --tokens")
    parser.add_argument("--out", required=True)
    parser.add_argument("--shape", default="8x384", help="LAYERSxWIDTH (lm phase; pools takes it from --init)")
    parser.add_argument("--init", help="checkpoint to start from (pools phase)")
    parser.add_argument("--steps", type=int, default=20000, help="lm phase")
    parser.add_argument("--window", type=int, default=128, help="lm phase window")
    parser.add_argument("--pools")
    parser.add_argument("--teacher", action="append", default=[], help="name=path[:weight[:temperature]], a scorer's output (id, scores) over some of --pools; repeat for several teachers")
    parser.add_argument("--dev", help="dev pools for picking the checkpoint")
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--batch", type=int, default=64, help="lm: windows per step; pools: pools per step")
    parser.add_argument("--lm-batch", type=int, default=32)
    parser.add_argument("--lm-weight", type=float, default=0.1)
    parser.add_argument("--lm-term", action="store_true", help="pools phase: score with the candidate's character log-likelihood too (ChoiceConfig.lm_term)")
    parser.add_argument("--candidate-lm-weight", type=float, default=1.0, help="with --lm-term: weight of the gold candidates' conditioned LM loss")
    parser.add_argument("--z", type=float, default=0.1, help="with --lm-term: self-normalisation penalty on every LM loss")
    parser.add_argument("--lr", type=float, default=0.0)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20261031)
    parser.add_argument("--progress", help="the dashboard's progress file")
    parser.add_argument("--progress-key", default="")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = pick_device()
    vocab = json.loads(Path(args.vocab).read_text(encoding="utf-8"))["tokens"]
    tokens = np.memmap(args.tokens, dtype=np.uint16, mode="r")
    split = int(len(tokens) * 0.999)
    gen = torch.Generator().manual_seed(args.seed)

    if args.init:
        start = torch.load(args.init, map_location="cpu", weights_only=True)
        cfg = ChoiceConfig(**start["config"])
        if args.lm_term and not cfg.lm_term:
            cfg.lm_term = True
            model = ChoiceModel(cfg)
            loaded = model.load_state_dict(start["model"], strict=False)
            assert loaded.missing_keys == ["lm_scale"] and not loaded.unexpected_keys, loaded
        else:
            model = ChoiceModel(cfg)
            model.load_state_dict(start["model"])
    else:
        layers, width = (int(x) for x in args.shape.split("x"))
        cfg = ChoiceConfig(vocab=len(vocab), n_layer=layers, n_embd=width, n_head=width // 64)
        model = ChoiceModel(cfg)
    model.to(device)
    (out / "config.json").write_text(json.dumps({"choice": dataclasses.asdict(cfg), "args": vars(args)}, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "vocab.json").write_text(json.dumps({"tokens": vocab}, ensure_ascii=False), encoding="utf-8")
    print(f"{sum(p.numel() for p in model.parameters()):,} parameters ({model.non_embedding_parameters():,} outside the embeddings) on {device}", flush=True)
    log = open(out / "log.jsonl", "a", encoding="utf-8")

    def record(row):
        log.write(json.dumps({**row, "t": time.time()}) + "\n")
        log.flush()

    lm_train = lm_windows(tokens[:split], args.lm_batch if args.phase == "pools" else args.batch, args.window, device, gen)
    lm_val = lm_windows(tokens[split:], 64, args.window, device, torch.Generator().manual_seed(7))

    def lm_val_loss():
        model.eval()
        with torch.no_grad():
            value = sum(model.lm_loss(*next(lm_val)).item() for _ in range(10)) / 10
        model.train()
        return value

    if args.phase == "lm":
        lr = args.lr or 6e-4
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)
        best = float("inf")
        started = time.time()
        for step in range(args.steps):
            for group in optimizer.param_groups:
                group["lr"] = learning_rate(step, args.steps, lr, args.warmup)
            loss = model.lm_loss(*next(lm_train))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if device == "mps" and (step + 1) % 50 == 0:
                torch.mps.empty_cache()
            if (step + 1) % 100 == 0:
                record({"kind": "train", "step": step + 1, "steps": args.steps, "loss": loss.item(), "wall_s": time.time() - started})
                report(args.progress, args.progress_key, step + 1, args.steps, time.time() - started, f"{step + 1} / {args.steps} 步 · loss {loss.item():.3f}")
            if (step + 1) % args.eval_every == 0 or step + 1 == args.steps:
                val = lm_val_loss()
                print(f"step {step + 1}/{args.steps}  train {loss.item():.4f}  val {val:.4f}  {(time.time() - started) / 60:.1f} min", flush=True)
                record({"kind": "val", "step": step + 1, "lm_val": val})
                if val < best:
                    best = val
                    save(out / "checkpoint.pt", model, step + 1, {"lm_val": val})
        record({"kind": "end", "best_lm_val": best, "minutes": round((time.time() - started) / 60, 1)})
        return

    encoder = Encoder(vocab)
    pools = read_jsonl(args.pools)
    teachers = [parse_teacher(spec) for spec in args.teacher]
    for teacher in teachers:
        teacher["scores"] = {row["id"]: row["scores"] for row in read_jsonl(teacher["path"])}
    dev = read_jsonl(args.dev)
    pools = [p for p in pools if p["gold"] in p["candidates"]]
    steps = math.ceil(len(pools) * args.epochs / args.batch)
    covered = ", ".join(f"{t['name']} {sum(1 for p in pools if p['id'] in t['scores'])}" for t in teachers) or "no teacher"
    print(f"{len(pools)} pools (scored by: {covered}), {len(dev)} dev pools, {steps} steps of {args.batch}", flush=True)
    lr = args.lr or 3e-4
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)
    best = evaluate(model, encoder, dev, device)
    print(f"start: {best}/{len(dev)} dev hits", flush=True)
    record({"kind": "val", "step": 0, "hits": best, "n": len(dev)})
    save(out / "checkpoint.pt", model, 0, {"dev_hits": best})
    order = list(range(len(pools)))
    random.shuffle(order)
    cursor = 0
    started = time.time()
    for step in range(steps):
        for group in optimizer.param_groups:
            group["lr"] = learning_rate(step, steps, lr, args.warmup)
        if cursor + args.batch > len(order):
            random.shuffle(order)
            cursor = 0
        cases = [pools[i] for i in order[cursor : cursor + args.batch]]
        cursor += args.batch
        chars, lets, segs, ends, spans = batch_rows(encoder, cases, device)
        z = args.z if cfg.lm_term else 0.0
        if cfg.lm_term:
            scores, hidden = model.rows(chars, lets, segs, ends, hidden=True)
        else:
            scores = model.rows(chars, lets, segs, ends)
        gold, kd = page_loss(scores, spans, cases, teachers)
        loss = gold + sum(t["weight"] * kd[t["name"]] for t in teachers)
        cand_lm = None
        if cfg.lm_term and args.candidate_lm_weight:
            gold_rows = torch.tensor([a + case["candidates"].index(case["gold"]) for case, (a, _) in zip(cases, spans)], device=device)
            cand_lm = model.candidate_lm_loss(hidden, chars, segs, gold_rows, z)
            loss = loss + args.candidate_lm_weight * cand_lm
        lm = None
        if args.lm_weight:
            lm = model.lm_loss(*next(lm_train), z=z)
            loss = loss + args.lm_weight * lm
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if device == "mps" and (step + 1) % 50 == 0:
            torch.mps.empty_cache()
        if (step + 1) % 50 == 0:
            record({"kind": "train", "step": step + 1, "steps": steps, "loss": loss.item(), "gold": gold.item(), "kd": {name: v.item() for name, v in kd.items()}, "lm": lm.item() if lm is not None else None, "candidate_lm": cand_lm.item() if cand_lm is not None else None, "wall_s": time.time() - started})
            report(args.progress, args.progress_key, step + 1, steps, time.time() - started, f"{step + 1} / {steps} 步 · 开发集最好 {best} / {len(dev)}")
        if (step + 1) % args.eval_every == 0 or step + 1 == steps:
            hits = evaluate(model, encoder, dev, device)
            val = lm_val_loss()
            print(f"step {step + 1}/{steps}  loss {loss.item():.4f}  dev {hits}/{len(dev)}  lm {val:.4f}  {(time.time() - started) / 60:.1f} min", flush=True)
            record({"kind": "val", "step": step + 1, "hits": hits, "n": len(dev), "lm_val": val})
            if hits > best:
                best = hits
                save(out / "checkpoint.pt", model, step + 1, {"dev_hits": hits})
    print(f"best {best}/{len(dev)} dev hits", flush=True)
    record({"kind": "end", "best": best, "n": len(dev), "minutes": round((time.time() - started) / 60, 1)})


if __name__ == "__main__":
    main()
