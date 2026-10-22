from pathlib import Path
import argparse, collections, gzip, json, re
parser = argparse.ArgumentParser()
parser.add_argument('--recorded', action='store_true')
args = parser.parse_args()
summary = {}
for label in ['red', 'candidate']:
    if args.recorded:
        text = gzip.decompress((Path(__file__).resolve().parent / (label + '-dot.s.gz')).read_bytes()).decode()
    else:
        text = Path('reference/target/safe-dot-evidence/' + label + '-reproduce-dot.s').read_text()
    instructions = []
    for line in text.splitlines():
        match = re.match(r'([0-9a-f]+):\s+[0-9a-f]+\s+(\S+)\s*(.*)', line)
        if match:
            instructions.append((int(match[1], 16), match[2], match[3]))
    for end, (address, op, operands) in enumerate(instructions):
        branch = re.match(r'0x([0-9a-f]+)', operands)
        if op == 'b.ne' and branch and int(branch[1], 16) < address:
            start = next(i for i, row in enumerate(instructions) if row[0] == int(branch[1], 16))
            loop = instructions[start:end + 1]
            summary[label] = {'start': hex(loop[0][0]), 'end': hex(loop[-1][0]), 'instructions': len(loop), 'op_counts': dict(collections.Counter(row[1] for row in loop))}
            break
    assert label in summary, label
if not args.recorded:
    Path('reference/target/safe-dot-evidence/reproduce-analysis.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary, indent=2))
