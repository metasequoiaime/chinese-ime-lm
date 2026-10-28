#!/usr/bin/env bash
# Measures the plugin on Arch Linux's own librime: builds it against the same librime release, installs it where Arch loads plugins from, and runs eval/sentences-v1.tsv through luna_pinyin_simp in four configurations. Run it on the host; it rebuilds the image in Dockerfile and re-runs itself inside the container. Results go to $IME_LM_EVAL_OUT (default: integrations/rime/eval/out).
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
image=chinese-ime-lm-rime-eval
model_url=https://github.com/metasequoiaime/chinese-ime-lm/releases/download/model-v1/sentence-model.safetensors
model_sha256=86ac529510cb3b4968a5e6ade83ec8080f5b34a0a681e75c362accbbd387d1c1
schema=luna_pinyin_simp

if [[ "${1:-}" != "--inside" ]]; then
  repo="$(cd "$here/../../.." && pwd)"
  out="${IME_LM_EVAL_OUT:-$here/out}"
  mkdir -p "$out"
  docker build --platform linux/amd64 -t "$image" "$here"
  exec docker run --rm --platform linux/amd64 -v "$repo:/src:ro" -v "$out:/out" "$image" bash /src/integrations/rime/eval/run.sh --inside
fi

work=/work
mkdir -p "$work"
package=$(pacman -Q librime | awk '{print $2}')
# Arch versions read [epoch:]version-release, such as 1:1.17.0-6; the librime tag is the middle part.
version=${package#*:}
version=${version%%-*}
echo "librime $package, schema $schema"

git clone --quiet --depth 1 --branch "$version" https://github.com/rime/librime.git "$work/librime"
ln -s /src/integrations/rime "$work/librime/plugins/ime-lm"
# The options and flags of Arch's own librime build (its PKGBUILD), so the plugin matches the library that loads it.
CXXFLAGS="${CXXFLAGS:-} -DNDEBUG -DGLOG_USE_GLOG_EXPORT -DGLOG_USE_GFLAGS" \
  cmake -S "$work/librime" -B "$work/build" -GNinja -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr \
    -DBUILD_MERGED_PLUGINS=Off -DENABLE_EXTERNAL_PLUGINS=On -DBUILD_TEST=Off -Wno-dev > /dev/null
cmake --build "$work/build" --target rime-ime-lm
install -Dm755 "$work/build/lib/rime-plugins/librime-ime-lm.so" /usr/lib/rime-plugins/librime-ime-lm.so

curl -fsSL -o "$work/model.safetensors" "$model_url"
echo "$model_sha256  $work/model.safetensors" | sha256sum --check --quiet -

g++ -std=c++17 -O2 /src/integrations/rime/eval/rime_eval.cc $(pkg-config --cflags --libs rime) -o "$work/rime_eval"

# One user data directory per configuration; each deploys luna_pinyin_simp with its own patch.
configure() {
  local dir="$work/user-$1"
  mkdir -p "$dir"
  printf 'patch:\n  schema_list:\n    - schema: %s\n' "$schema" > "$dir/default.custom.yaml"
  cat > "$dir/$schema.custom.yaml"
}
configure default <<'EOF'
patch: {}
EOF
configure sentences5 <<'EOF'
patch:
  translator/max_sentences: 5
EOF
configure ime-lm-sentences1 <<EOF
patch:
  engine/filters/+:
    - ime_lm_filter
  ime_lm_filter/model: $work/model.safetensors
EOF
configure ime-lm-sentences5 <<EOF
patch:
  translator/max_sentences: 5
  engine/filters/+:
    - ime_lm_filter
  ime_lm_filter/model: $work/model.safetensors
EOF

for config in default sentences5 ime-lm-sentences1 ime-lm-sentences5; do
  "$work/rime_eval" /usr/share/rime-data "$work/user-$config" "$schema" /src/eval/sentences-v1.tsv > "/out/$config.tsv"
done
python3 /src/integrations/rime/eval/summarize.py /out/default.tsv /out/sentences5.tsv /out/ime-lm-sentences1.tsv /out/ime-lm-sentences5.tsv | tee /out/summary.txt
