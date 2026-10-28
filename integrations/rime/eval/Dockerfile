# The environment behind the numbers in integrations/rime/README.md: Arch Linux's own librime, which is built with external plugins enabled, and its luna_pinyin data. The plugin is built against the same librime release and installed where Arch's librime loads plugins from, which is also how an AUR package would ship it.
#
# Arch Linux publishes amd64 images only, so run.sh builds this with --platform linux/amd64; on an arm64 host it runs under emulation.
FROM archlinux:base-devel

# pacman's download sandbox relies on seccomp filters that emulated amd64 containers reject, which fails every download; the packages are still verified against their signatures.
RUN pacman --disable-sandbox -Syu --noconfirm --needed \
      boost capnproto cmake git google-glog leveldb lua marisa ninja opencc python rust yaml-cpp \
      librime rime-essay rime-luna-pinyin rime-prelude \
 && pacman --disable-sandbox -Scc --noconfirm
