#!/usr/bin/env bash
set -euo pipefail

REPO_URL="https://github.com/NormalFix/fix"
TARBALL_URL="https://github.com/NormalFix/fix/archive/refs/heads/main.tar.gz"
FIX_HOME="${FIX_HOME:-$HOME/.fix}"
VENV_DIR="$FIX_HOME/venv"
BIN_DIR="${HOME}/.local/bin"

printf "\033[1;36m✦ Installing Fix (v1.0.0-beta) — Autonomous AI Engineering Partner...\033[0m\n"

# 1. Locate Python >= 3.9
PYTHON_BIN=""
for candidate in python3 python3.13 python3.12 python3.11 python3.10 python3.9; do
    if command -v "$candidate" >/dev/null 2>&1; then
        if "$candidate" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 9) else 1)' >/dev/null 2>&1; then
            PYTHON_BIN="$(command -v "$candidate")"
            break
        fi
    fi
done

if [ -z "$PYTHON_BIN" ]; then
    printf "\033[1;31m✗ Error: Python 3.9+ is required but was not found.\033[0m\n"
    printf "  macOS: brew install python\n"
    printf "  Ubuntu/Debian: sudo apt update && sudo apt install -y python3 python3-venv\n"
    exit 1
fi

mkdir -p "$FIX_HOME" "$BIN_DIR"

# 2. Create isolated virtual environment (~/.fix/venv)
if [ ! -x "$VENV_DIR/bin/python" ]; then
    printf "  → Creating isolated runtime environment in %s...\n" "$VENV_DIR"
    "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

# 3. Install Fix package (from local checkout if present, otherwise from GitHub)
printf "  → Installing fix-agent package and dependencies...\n"
"$VENV_DIR/bin/python" -m pip install --quiet --upgrade pip setuptools wheel

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" 2>/dev/null && pwd || pwd)"
if [ -n "${FIX_INSTALL_SOURCE:-}" ]; then
    "$VENV_DIR/bin/pip" install --quiet --upgrade "$FIX_INSTALL_SOURCE"
elif [ -f "$SCRIPT_DIR/pyproject.toml" ] && [ -d "$SCRIPT_DIR/src/fix" ]; then
    "$VENV_DIR/bin/pip" install --quiet --upgrade "$SCRIPT_DIR"
else
    if ! "$VENV_DIR/bin/pip" install --quiet --upgrade "$TARBALL_URL" 2>/dev/null; then
        "$VENV_DIR/bin/pip" install --quiet --upgrade "git+${REPO_URL}.git"
    fi
fi

# 4. Link `fix` binary into PATH
ln -sf "$VENV_DIR/bin/fix" "$BIN_DIR/fix"

for sys_bin in /opt/homebrew/bin /usr/local/bin; do
    if [ -d "$sys_bin" ] && [ -w "$sys_bin" ]; then
        ln -sf "$VENV_DIR/bin/fix" "$sys_bin/fix" 2>/dev/null || true
    fi
done

# 5. Ensure ~/.local/bin is configured in shell profile
PATH_LINE='export PATH="$HOME/.local/bin:$PATH"'
for rc in "$HOME/.zshrc" "$HOME/.bashrc" "$HOME/.bash_profile"; do
    if [ -f "$rc" ]; then
        if ! grep -Fq ".local/bin" "$rc" 2>/dev/null; then
            printf "\n# Added by Fix installer\n%s\n" "$PATH_LINE" >> "$rc"
        fi
    fi
done
if [ ! -f "$HOME/.zshrc" ] && [ ! -f "$HOME/.bashrc" ] && [ ! -f "$HOME/.bash_profile" ]; then
    printf "%s\n" "$PATH_LINE" >> "$HOME/.zshrc"
fi

VER="$("$VENV_DIR/bin/fix" --version 2>/dev/null || echo "Fix v1.0.0-beta")"
printf "\n\033[1;32m✓ %s 安装成功！\033[0m\n" "$VER"
printf "  可执行文件位置: \033[1m%s/fix\033[0m\n" "$BIN_DIR"
if ! command -v fix >/dev/null 2>&1; then
    printf "  \033[1;33m提示:\033[0m 若当前终端提示找不到 fix，请执行: \033[1mexport PATH=\"\$HOME/.local/bin:\$PATH\"\033[0m 或重开终端窗口。\n"
fi
printf "\n\033[1;36m快速开始:\033[0m\n"
printf "  1. 进入任意代码仓库:  \033[1mcd /path/to/your-project && fix\033[0m\n"
printf "  2. 连接大模型厂商:    在交互界面输入 \033[1m/connect\033[0m（支持 DeepSeek / GLM / MiMo / MiniMax / Qwen / OpenAI / Anthropic）\n"
