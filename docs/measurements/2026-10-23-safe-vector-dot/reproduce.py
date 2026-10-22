from pathlib import Path
import argparse, gzip, hashlib, json, shutil, subprocess

parser = argparse.ArgumentParser()
parser.add_argument('--check-only', action='store_true')
args = parser.parse_args()
delivery = Path(__file__).resolve().parent
folder = Path('reference/target/safe-dot-evidence')
folder.mkdir(parents=True, exist_ok=True)
provenance = json.loads(gzip.decompress((delivery / 'provenance.json.gz').read_bytes()))
originals = {}
for name, expected in provenance['restored'].items():
    originals[name] = Path(name).read_bytes()
    assert hashlib.sha256(originals[name]).hexdigest() == expected['sha256'], name
patch = folder / 'reproduce.patch'
patch.write_bytes(gzip.decompress((delivery / 'experimental.patch.gz').read_bytes()))
subprocess.run(['git', 'apply', '--check', str(patch)], check=True)
if args.check_only:
    print('固定源码、清单、锁文件与临时补丁检查通过，未修改生产源码或构建')
    raise SystemExit(0)
try:
    subprocess.run(['git', 'apply', str(patch)], check=True)
    for label, source in [('red', 'red-probe.rs.gz'), ('candidate', 'candidate-probe.rs.gz')]:
        (folder / 'probe.rs').write_bytes(gzip.decompress((delivery / source).read_bytes()))
        command = ['cargo', 'test', '--manifest-path', 'reference/Cargo.toml', '--release', '--locked', '--lib', '--no-run', '--message-format=json']
        result = subprocess.run(command, capture_output=True, text=True)
        (folder / (label + '-reproduce-build.log')).write_text(result.stdout + result.stderr)
        result.check_returncode()
        messages = [json.loads(line) for line in result.stdout.splitlines() if line.startswith('{')]
        executables = [row['executable'] for row in messages if row.get('reason') == 'compiler-artifact' and row.get('executable')]
        assert len(executables) == 1, executables
        shutil.copy2(executables[0], folder / (label + '-binary'))
finally:
    for name, raw in originals.items():
        Path(name).write_bytes(raw)
assert all(Path(name).read_bytes() == raw for name, raw in originals.items())
for label, needle in [('red', '6dot_i8'), ('candidate', 'candidate_dot')]:
    binary = folder / (label + '-binary')
    with (folder / (label + '-reproduce-bits.log')).open('w') as output:
        subprocess.run([str(binary), 'safe_dot_probe::unchanged_bits', '--exact', '--nocapture'], stdout=output, stderr=subprocess.STDOUT, check=True)
    names = subprocess.check_output(['xcrun', 'nm', str(binary)], text=True)
    symbols = [line.split()[-1] for line in names.splitlines() if needle in line]
    assert len(symbols) == 1, symbols
    assembly = subprocess.check_output(['xcrun', 'llvm-objdump', '--disassemble-symbols=' + symbols[0], str(binary)], text=True)
    (folder / (label + '-reproduce-dot.s')).write_text(assembly)
print('两份程序和逐位对照已生成，生产源码、清单与锁文件已恢复；没有运行计时')
