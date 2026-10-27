"""Universal Polyglot Cross-File Semantic Call Graph, Type Hierarchy & Declarative Config Radar (`src/fix/lsp.py`)."""

from __future__ import annotations

import ast
import builtins
import json
import os
import re
import shutil
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterable

from fix.schema import CallHierarchyReport, DiagnosticItem, SymbolLocation

IGNORE_DIRS = {
    ".git",
    ".fix",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "dist",
    "build",
    "target",
    "out",
    ".gradle",
    ".idea",
    ".kotlin",
    ".qoder",
    ".cursor",
    "captures",
    ".next",
}

PYTHON_BUILTINS = set(dir(builtins)) | {"self", "cls", "__name__", "__file__", "__doc__", "__all__"}

CODE_EXTENSIONS = (
    ".py",
    ".ts",
    ".tsx",
    ".js",
    ".jsx",
    ".go",
    ".rs",
    ".java",
    ".kt",
    ".c",
    ".cpp",
    ".cc",
    ".h",
    ".hpp",
    ".cs",
    ".swift",
    ".rb",
    ".php",
)

DECLARATIVE_EXTENSIONS = (
    ".yaml",
    ".yml",
    ".json",
    ".toml",
    ".xml",
    ".proto",
    ".graphql",
    ".gql",
    ".sql",
    ".html",
    ".vue",
    ".svelte",
)


def _infer_module_scope(rel_path: str) -> str:
    """Infer logical module/package/workspace scope from a relative file path in any repository layout."""
    parts = rel_path.replace("\\", "/").split("/")
    if len(parts) <= 1:
        return "root"
    if parts[0] in ("packages", "crates", "services", "apps", "modules") and len(parts) >= 2:
        return f"{parts[0]}/{parts[1]}"
    if "src" in parts:
        idx = parts.index("src")
        if idx + 1 < len(parts) - 1:
            return parts[idx + 1]
    return parts[0]


def check_brace_balance_in_source(source: str) -> tuple[bool, int, str]:
    """
    Verify balanced `{}`, `()`, `[]` in C-family / braces-based languages (TS/JS/Go/Rust/C/C++/Java/Kotlin)
    while ignoring string/char literals and line/block comments.
    Returns `(ok, error_line, message)`.
    """
    stack: list[tuple[str, int]] = []
    pairs = {"}": "{", ")": "(", "]": "["}
    in_line_comment = False
    in_block_comment = False
    in_str: str | None = None
    in_triple_str = False
    line_no = 1
    i = 0
    n = len(source)

    while i < n:
        ch = source[i]
        nxt = source[i + 1] if i + 1 < n else ""
        nxt3 = source[i : i + 3]

        if ch == "\n":
            line_no += 1
            in_line_comment = False
            i += 1
            continue

        if in_line_comment:
            i += 1
            continue

        if in_block_comment:
            if ch == "*" and nxt == "/":
                in_block_comment = False
                i += 2
            else:
                i += 1
            continue

        if in_triple_str:
            if nxt3 == '"""':
                in_triple_str = False
                i += 3
            else:
                i += 1
            continue

        if in_str is not None:
            if ch == "\\":
                i += 2
                continue
            if ch == in_str:
                in_str = None
            i += 1
            continue

        if ch == "/" and nxt == "/":
            in_line_comment = True
            i += 2
            continue
        if ch == "/" and nxt == "*":
            in_block_comment = True
            i += 2
            continue
        if nxt3 == '"""':
            in_triple_str = True
            i += 3
            continue
        if ch in ('"', "'", "`"):
            in_str = ch
            i += 1
            continue

        if ch in ("{", "(", "["):
            stack.append((ch, line_no))
        elif ch in ("}", ")", "]"):
            expected = pairs[ch]
            if not stack:
                return False, line_no, f"Unmatched closing `{ch}` at line {line_no}"
            top_ch, top_ln = stack.pop()
            if top_ch != expected:
                return False, line_no, f"Mismatched `{ch}` at line {line_no} (opened `{top_ch}` at line {top_ln})"
        i += 1

    if stack:
        open_ch, open_ln = stack[-1]
        return False, open_ln, f"Unclosed `{open_ch}` opened at line {open_ln}"
    return True, 0, ""


class SemanticRadar:
    """
    Universal, domain-agnostic cross-file semantic resolver & static diagnostic engine.
    Supports Python, TypeScript/JavaScript, Go, Rust, C/C++, Java, Kotlin, C#, Swift,
    plus cross-referencing with declarative configs/schemas (`.yaml`, `.json`, `.toml`, `.proto`, `.graphql`, `.sql`, `.xml`, `.vue`).
    """

    def __init__(self, repo_root: str | Path) -> None:
        self.repo_root = Path(repo_root).resolve()

    def detect_available_lsp(self) -> str:
        """Return the name of any installed local language server / typechecker binary."""
        for binary in (
            "pyright-langserver",
            "basedpyright-langserver",
            "pyright",
            "pylsp",
            "typescript-language-server",
            "gopls",
            "rust-analyzer",
            "clangd",
            "jdtls",
            "kotlin-language-server",
        ):
            if shutil.which(binary):
                return f"{binary}+polyglot_ast_graph"
        return "polyglot_ast_graph"

    def _iter_source_files(
        self,
        extensions: tuple[str, ...] | None = None,
        containing_token: str | None = None,
    ) -> Iterable[Path]:
        exts = extensions or (CODE_EXTENSIONS + DECLARATIVE_EXTENSIONS)
        rg_bin = shutil.which("rg")
        if rg_bin:
            cmd = [
                rg_bin,
                "--files-with-matches" if containing_token else "--files",
                "--color=never",
                "--glob=!.git/**",
                "--glob=!.fix/**",
                "--glob=!.venv/**",
                "--glob=!node_modules/**",
                "--glob=!build/**",
                "--glob=!**/build/**",
                "--glob=!target/**",
                "--glob=!dist/**",
                "--glob=!.gradle/**",
            ]
            if containing_token:
                cmd.extend(["-F", containing_token, "."])
            try:
                proc = subprocess.run(cmd, cwd=self.repo_root, capture_output=True, text=True, timeout=5)
                if proc.returncode in (0, 1):
                    for rel_line in proc.stdout.splitlines():
                        rel_clean = rel_line.strip().lstrip("./")
                        if not rel_clean:
                            continue
                        fpath = self.repo_root / rel_clean
                        if fpath.suffix.lower() in exts and fpath.is_file():
                            yield fpath
                    return
            except Exception:
                pass

        for root, dirs, files in os.walk(self.repo_root):
            dirs[:] = [d for d in sorted(dirs) if d not in IGNORE_DIRS and not d.startswith(".")]
            for fname in sorted(files):
                fpath = Path(root) / fname
                if fpath.suffix.lower() in exts:
                    yield fpath

    # ------------------------------------------------------------------
    # 1. Universal Cross-File Symbol Trace (`trace_symbol`)
    # ------------------------------------------------------------------

    def trace_symbol(self, symbol_name: str, rel_path: str | None = None) -> CallHierarchyReport:
        """
        Resolve exact definition(s), supertypes/traits/interfaces, implementations/subclasses,
        decorators/attributes/wiring, declarative config/schema links, incoming callers, and outgoing callees
        across ANY programming language in the repository.
        """
        clean_sym = symbol_name.strip()
        short_name = re.split(r"[.:]+", clean_sym)[-1].strip()
        outer_qualifier = re.split(r"[.:]+", clean_sym)[0].strip() if ("." in clean_sym or "::" in clean_sym) else None

        definitions: list[SymbolLocation] = []
        supertypes: list[str] = []
        implementations: list[SymbolLocation] = []
        wiring_and_decorators: list[str] = []
        declarative_config_links: list[str] = []
        callers: list[SymbolLocation] = []
        callees_set: list[str] = []

        for fpath in self._iter_source_files(containing_token=short_name):
            rel = str(fpath.relative_to(self.repo_root))
            suffix = fpath.suffix.lower()
            try:
                source = fpath.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue

            if suffix == ".py":
                py_res = self._scan_python_file(
                    rel=rel,
                    source=source,
                    short_name=short_name,
                    class_qualifier=outer_qualifier,
                    focus_rel_path=rel_path,
                )
                definitions.extend(py_res["defs"])
                callers.extend(py_res["refs"])
                for s in py_res["supertypes"]:
                    if s not in supertypes:
                        supertypes.append(s)
                implementations.extend(py_res["implementations"])
                for w in py_res["wiring"]:
                    if w not in wiring_and_decorators:
                        wiring_and_decorators.append(w)
                for callee in py_res["callees"]:
                    if callee not in callees_set:
                        callees_set.append(callee)

            elif suffix in CODE_EXTENSIONS:
                poly_res = self._scan_polyglot_code_file(
                    rel=rel,
                    source=source,
                    full_symbol=clean_sym,
                    short_name=short_name,
                    outer_qualifier=outer_qualifier,
                )
                definitions.extend(poly_res["defs"])
                for s in poly_res["supertypes"]:
                    if s not in supertypes:
                        supertypes.append(s)
                implementations.extend(poly_res["implementations"])
                for w in poly_res["wiring"]:
                    if w not in wiring_and_decorators:
                        wiring_and_decorators.append(w)
                callers.extend(poly_res["callers"])
                for callee in poly_res["callees"]:
                    if callee not in callees_set:
                        callees_set.append(callee)

            elif suffix in DECLARATIVE_EXTENSIONS:
                links = self._scan_declarative_file(
                    rel=rel,
                    source=source,
                    short_name=short_name,
                    full_symbol=clean_sym,
                )
                for lk in links:
                    if lk not in declarative_config_links:
                        declarative_config_links.append(lk)

        if rel_path:
            definitions.sort(key=lambda d: (0 if d.rel_path == rel_path else 1, d.rel_path, d.line))

        return CallHierarchyReport(
            symbol=symbol_name,
            definitions=definitions[:15],
            supertypes=supertypes[:20],
            implementations=implementations[:25],
            wiring_and_decorators=wiring_and_decorators[:20],
            declarative_config_links=declarative_config_links[:25],
            callers=callers[:40],
            callees=callees_set[:30],
            lsp_backend_used=self.detect_available_lsp(),
        )

    # ------------------------------------------------------------------
    # Universal Polyglot Code Scanner (TS/JS, Go, Rust, C/C++, Java, Kotlin, C#, Swift)
    # ------------------------------------------------------------------

    def _scan_polyglot_code_file(
        self,
        rel: str,
        source: str,
        full_symbol: str,
        short_name: str,
        outer_qualifier: str | None,
    ) -> dict[str, list]:
        lines = source.splitlines()
        scope = _infer_module_scope(rel)
        defs: list[SymbolLocation] = []
        supertypes: list[str] = []
        implementations: list[SymbolLocation] = []
        wiring: list[str] = []
        callers: list[SymbolLocation] = []
        callees: list[str] = []

        # Universal type declaration regex across TS/JS, Go, Rust, C/C++, Java, Kotlin, C#, Swift
        type_decl_re = re.compile(
            r"^\s*(?:export\s+)?(?:default\s+)?(?:(?:public|private|protected|internal|abstract|open|final|data|sealed|inner|enum|value|pub|async)\s+)*"
            r"(?:(class|interface|object|trait|struct|enum|type)\s+([A-Za-z0-9_]+)|type\s+([A-Za-z0-9_]+)\s+(struct|interface))\b(.*)"
        )
        # Rust `impl Trait for Type` or `impl Type`
        rust_impl_re = re.compile(
            r"^\s*impl(?:<[^>]+>)?\s+([A-Za-z0-9_:]+)(?:\s+for\s+([A-Za-z0-9_:]+))?"
        )
        # Universal function/method declaration across TS/JS, Go, Rust, C/C++, Java, Kotlin
        fun_decl_re = re.compile(
            r"^\s*(?:export\s+)?(?:(override)\s+)?(?:(?:public|private|protected|internal|open|suspend|inline|async|pub|static|virtual)\s+)*"
            r"(?:(?:function|fun|fn|def)\s+(?:\([^)]+\)\s*)?|func\s+(?:\([^)]+\)\s*)?|(?:[A-Za-z0-9_<>,?\[\]:*]+\s+))"
            r"([A-Za-z0-9_]+)\s*(?:<[^>]*>)?\s*\(([^)]*)\)"
        )
        decorator_re = re.compile(r"^\s*(@[A-Za-z0-9_.]+(?:\([^)]*\))?|#\[[^\]]+\])")
        call_re = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(")

        current_outer_type: str | None = None
        def_line_numbers: set[int] = set()

        for idx, line in enumerate(lines, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("//") or stripped.startswith("*"):
                continue

            # Collect decorators / attributes (`@xxx` or `#[xxx]`) on the 4 preceding lines
            prev_decorators: list[str] = []
            for back in range(max(0, idx - 5), idx):
                dm = decorator_re.match(lines[back])
                if dm:
                    dec = dm.group(1).strip()
                    if dec not in prev_decorators:
                        prev_decorators.append(dec)

            # 1. Check Rust `impl Trait for Struct`
            m_rust_impl = rust_impl_re.match(line)
            if m_rust_impl:
                first_t, second_t = m_rust_impl.group(1), m_rust_impl.group(2)
                end_ln = self._find_block_end_line(lines, idx)
                if second_t:
                    trait_name, struct_name = first_t.split("::")[-1], second_t.split("::")[-1]
                    current_outer_type = struct_name
                    if struct_name == short_name and trait_name not in supertypes:
                        supertypes.append(first_t)
                    elif trait_name == short_name:
                        implementations.append(
                            SymbolLocation(
                                symbol=struct_name,
                                kind="struct",
                                rel_path=rel,
                                line=idx,
                                end_line=end_ln,
                                signature=stripped.rstrip("{").strip(),
                                context_line=stripped,
                                module_scope=scope,
                            )
                        )
                else:
                    current_outer_type = first_t.split("::")[-1]

            # 2. Check class / struct / interface / trait / object / type declaration
            m_type = type_decl_re.match(line)
            if m_type:
                kind_raw = m_type.group(1) or m_type.group(4) or "class"
                type_name = m_type.group(2) or m_type.group(3) or ""
                indent = len(line) - len(line.lstrip())
                if indent <= 4:
                    current_outer_type = type_name

                header_chunk = " ".join(l.strip() for l in lines[idx - 1 : min(len(lines), idx + 4)])
                parsed_supers = self._extract_polyglot_supertypes(type_name, header_chunk)
                end_ln = self._find_block_end_line(lines, idx)

                # For Go structs, also check embedded types inside the struct block
                if rel.endswith(".go") and kind_raw == "struct":
                    for s_ln in lines[idx : min(len(lines), end_ln - 1)]:
                        s_clean = s_ln.strip()
                        if re.fullmatch(r"\*?[A-Za-z0-9_.]+", s_clean):
                            emb = s_clean.lstrip("*")
                            if emb not in parsed_supers:
                                parsed_supers.append(emb)

                qual_match = (
                    type_name == short_name
                    and (outer_qualifier is None or outer_qualifier == current_outer_type or outer_qualifier == type_name)
                )
                norm_kind = (
                    "interface"
                    if kind_raw in ("interface", "trait")
                    else ("struct" if kind_raw == "struct" else ("object" if kind_raw == "object" else "class"))
                )

                if qual_match:
                    def_line_numbers.add(idx)
                    defs.append(
                        SymbolLocation(
                            symbol=f"{current_outer_type}.{type_name}" if current_outer_type and current_outer_type != type_name else type_name,
                            kind=norm_kind,  # type: ignore[arg-type]
                            rel_path=rel,
                            line=idx,
                            end_line=end_ln,
                            signature=stripped.rstrip("{").strip(),
                            context_line=stripped,
                            module_scope=scope,
                        )
                    )
                    supertypes.extend(parsed_supers)
                    if prev_decorators:
                        wiring.append(f"[{rel}:{idx}] `{type_name}` decorated/wired with {', '.join(prev_decorators)}")
                    for body_ln in lines[idx : min(len(lines), end_ln)]:
                        for cm in call_re.finditer(body_ln):
                            cname = cm.group(1)
                            if cname not in ("if", "for", "while", "when", "switch", "catch", "return", "super", "this", "sizeof", short_name):
                                if cname not in callees:
                                    callees.append(cname)

                target_matches_super = any(
                    s == full_symbol or s == short_name or s.endswith(f".{short_name}") or s.endswith(f"::{short_name}")
                    for s in parsed_supers
                )
                if target_matches_super and not qual_match:
                    implementations.append(
                        SymbolLocation(
                            symbol=type_name,
                            kind=norm_kind,  # type: ignore[arg-type]
                            rel_path=rel,
                            line=idx,
                            end_line=end_ln,
                            signature=stripped.rstrip("{").strip(),
                            context_line=stripped,
                            module_scope=scope,
                        )
                    )

            # 3. Check function / method declaration
            if short_name in line and "(" in line and idx not in def_line_numbers:
                m_fun = fun_decl_re.match(line)
                if m_fun and m_fun.group(2) == short_name:
                    # Ensure it's not a control flow statement
                    if short_name not in ("if", "for", "while", "switch", "catch", "when"):
                        is_override = bool(m_fun.group(1)) or any("@Override" in d or "override" in d for d in prev_decorators)
                        end_ln = self._find_block_end_line(lines, idx)
                        def_line_numbers.add(idx)
                        loc = SymbolLocation(
                            symbol=f"{current_outer_type}.{short_name}" if current_outer_type else short_name,
                            kind="method" if current_outer_type else "function",
                            rel_path=rel,
                            line=idx,
                            end_line=end_ln,
                            signature=stripped.rstrip("{").strip(),
                            context_line=stripped,
                            module_scope=scope,
                        )
                        if is_override and outer_qualifier and current_outer_type != outer_qualifier:
                            implementations.append(loc)
                        else:
                            defs.append(loc)
                            if is_override:
                                implementations.append(loc)
                        if prev_decorators:
                            wiring.append(f"[{rel}:{idx}] `{loc.symbol}` decorated with {', '.join(prev_decorators)}")
                        for body_ln in lines[idx : min(len(lines), end_ln)]:
                            for cm in call_re.finditer(body_ln):
                                cname = cm.group(1)
                                if cname not in ("if", "for", "while", "when", "switch", "catch", "return", "super", "this", short_name):
                                    if cname not in callees:
                                        callees.append(cname)

            # 4. Check wiring/annotations & general references/callers
            if short_name in line and idx not in def_line_numbers:
                if prev_decorators or re.search(r"@[A-Za-z0-9_]+", line):
                    inline_decs = re.findall(r"@[A-Za-z0-9_]+", line) + prev_decorators
                    if inline_decs:
                        wiring.append(f"[{rel}:{idx}] Wiring/Annotation ({', '.join(dict.fromkeys(inline_decs))}): `{stripped}`")

                if re.search(rf"\b{re.escape(short_name)}\b", line) and not stripped.startswith(("import ", "package ", "use ", "#include")):
                    callers.append(
                        SymbolLocation(
                            symbol=short_name,
                            kind="reference",
                            rel_path=rel,
                            line=idx,
                            end_line=idx,
                            context_line=stripped,
                            module_scope=scope,
                        )
                    )

        return {
            "defs": defs,
            "supertypes": supertypes,
            "implementations": implementations,
            "wiring": wiring,
            "callers": callers,
            "callees": callees,
        }

    @staticmethod
    def _extract_polyglot_supertypes(type_name: str, header_chunk: str) -> list[str]:
        """Extract supertypes from `extends X implements Y` (TS/JS/Java) or `: Base, Trait` (Kotlin/C++/Rust/C#/Swift)."""
        before_brace = header_chunk.split("{")[0]
        supers: list[str] = []

        m_ext = re.search(r"\bextends\s+([A-Za-z0-9_.,\s<>]+?)(?:\bimplements\b|$)", before_brace)
        if m_ext:
            for item in m_ext.group(1).split(","):
                clean = re.sub(r"<.*?>", "", item).strip()
                if clean:
                    supers.append(clean)
        m_imp = re.search(r"\bimplements\s+([A-Za-z0-9_.,\s<>]+)$", before_brace)
        if m_imp:
            for item in m_imp.group(1).split(","):
                clean = re.sub(r"<.*?>", "", item).strip()
                if clean:
                    supers.append(clean)
        if supers:
            return supers

        idx_name = before_brace.find(type_name)
        if idx_name == -1:
            return []
        tail = before_brace[idx_name + len(type_name) :].strip()
        # Strip generic parameters `<T>`
        if tail.startswith("<"):
            depth = 0
            for i, ch in enumerate(tail):
                if ch == "<":
                    depth += 1
                elif ch == ">":
                    depth -= 1
                    if depth == 0:
                        tail = tail[i + 1 :].strip()
                        break
        # Strip Kotlin `constructor(...)` or primary constructor `(...)`
        for kw in ("constructor", "@Inject constructor", "public constructor", "internal constructor"):
            if tail.startswith(kw):
                tail = tail[len(kw) :].strip()
        if tail.startswith("("):
            depth = 0
            for i, ch in enumerate(tail):
                if ch == "(":
                    depth += 1
                elif ch == ")":
                    depth -= 1
                    if depth == 0:
                        tail = tail[i + 1 :].strip()
                        break
        if tail.startswith(":"):
            sup_str = tail[1:].strip()
            parts: list[str] = []
            curr: list[str] = []
            depth = 0
            for ch in sup_str:
                if ch in ("(", "<"):
                    depth += 1
                elif ch in (")", ">"):
                    depth -= 1
                if ch in (",", "+") and depth == 0:
                    parts.append("".join(curr).strip())
                    curr = []
                else:
                    curr.append(ch)
            if curr:
                parts.append("".join(curr).strip())

            for p in parts:
                p_clean = re.sub(r"^(?:public|private|protected|virtual)\s+", "", p.split(" by ")[0].strip())
                p_clean = re.sub(r"\(.*\)$", "", p_clean).strip()
                p_clean = re.sub(r"<.*?>", "", p_clean).strip()
                if p_clean and re.match(r"^[A-Za-z0-9_.:]+$", p_clean):
                    supers.append(p_clean)
        return supers

    @staticmethod
    def _find_block_end_line(lines: list[str], start_line: int) -> int:
        depth = 0
        opened = False
        for idx in range(start_line - 1, min(len(lines), start_line + 600)):
            for ch in lines[idx]:
                if ch == "{":
                    depth += 1
                    opened = True
                elif ch == "}":
                    depth -= 1
                    if opened and depth <= 0:
                        return idx + 1
        return start_line

    # ------------------------------------------------------------------
    # Universal Declarative Config, Schema & Template Scanner
    # ------------------------------------------------------------------

    @staticmethod
    def _scan_declarative_file(
        rel: str,
        source: str,
        short_name: str,
        full_symbol: str,
    ) -> list[str]:
        """
        Scan any declarative file (`.yaml`, `.json`, `.toml`, `.xml`, `.proto`, `.graphql`, `.sql`, `.html`, `.vue`)
        that references `short_name` or `full_symbol`.
        """
        if short_name not in source and full_symbol not in source:
            return []

        scope = _infer_module_scope(rel)
        scope_tag = f" (scope={scope})" if scope and scope != "root" else ""
        links: list[str] = []
        lines = source.splitlines()

        # If XML contains `<declare-styleable>` or custom elements, extract structured attributes too
        if rel.endswith(".xml") and "declare-styleable" in source:
            try:
                root = ET.fromstring(source)
                for ds in root.findall(".//declare-styleable"):
                    if ds.attrib.get("name") == short_name:
                        attrs = [a.attrib.get("name", "") for a in ds.findall("attr") if a.attrib.get("name")]
                        links.append(
                            f"[{rel}]{scope_tag} XML <declare-styleable name=\"{short_name}\">: "
                            + (", ".join(attrs) if attrs else "(no child attrs)")
                        )
            except ET.ParseError:
                pass

        for idx, line in enumerate(lines, start=1):
            if re.search(rf"\b{re.escape(short_name)}\b", line):
                chunk = " ".join(l.strip() for l in lines[idx - 1 : min(len(lines), idx + 5)])
                m_id = re.search(r'(?:id|name|key)\s*[:=]\s*["\'](@\+id/[A-Za-z0-9_]+|[A-Za-z0-9_.-]+)["\']', chunk)
                id_str = f" id={m_id.group(1)}" if m_id else ""
                links.append(f"[{rel}:{idx}]{scope_tag}{id_str} — `{line.strip()[:120]}`")
                if len(links) >= 12:
                    break

        return links

    # ------------------------------------------------------------------
    # Python AST Scanner
    # ------------------------------------------------------------------

    def _scan_python_file(
        self,
        rel: str,
        source: str,
        short_name: str,
        class_qualifier: str | None,
        focus_rel_path: str | None,
    ) -> dict[str, list]:
        try:
            tree = ast.parse(source, filename=rel)
        except SyntaxError:
            return {"defs": [], "refs": [], "callees": [], "supertypes": [], "implementations": [], "wiring": []}

        lines = source.splitlines()
        scope = _infer_module_scope(rel)
        defs: list[SymbolLocation] = []
        refs: list[SymbolLocation] = []
        callees: list[str] = []
        supertypes: list[str] = []
        implementations: list[SymbolLocation] = []
        wiring: list[str] = []

        def _get_line(lineno: int) -> str:
            return lines[lineno - 1].strip() if 1 <= lineno <= len(lines) else ""

        def _extract_calls_from_node(body_node: ast.AST) -> list[str]:
            found: list[str] = []
            for child in ast.walk(body_node):
                if isinstance(child, ast.Call):
                    cname = self._extract_call_name(child.func)
                    if cname and cname not in found:
                        found.append(cname)
            return found

        def _extract_decorators(node: ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
            decs: list[str] = []
            for d in node.decorator_list:
                if hasattr(ast, "unparse"):
                    try:
                        decs.append(f"@{ast.unparse(d)}")
                    except Exception:
                        pass
            return decs

        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                bases = [ast.unparse(b) if hasattr(ast, "unparse") else "" for b in node.bases]
                clean_bases = [b for b in bases if b]
                end_ln = getattr(node, "end_lineno", node.lineno)
                sig = f"class {node.name}({', '.join(clean_bases)})" if clean_bases else f"class {node.name}"
                decs = _extract_decorators(node)

                if node.name == short_name and (class_qualifier is None or class_qualifier == short_name):
                    defs.append(
                        SymbolLocation(
                            symbol=node.name,
                            kind="class",
                            rel_path=rel,
                            line=node.lineno,
                            end_line=end_ln,
                            signature=sig,
                            docstring=(ast.get_docstring(node) or "")[:240],
                            context_line=_get_line(node.lineno),
                            module_scope=scope,
                        )
                    )
                    supertypes.extend(clean_bases)
                    if decs:
                        wiring.append(f"[{rel}:{node.lineno}] `{node.name}` decorated with {', '.join(decs)}")
                    if focus_rel_path is None or rel == focus_rel_path:
                        callees.extend(_extract_calls_from_node(node))
                elif any(b == short_name or b.endswith(f".{short_name}") for b in clean_bases):
                    implementations.append(
                        SymbolLocation(
                            symbol=node.name,
                            kind="class",
                            rel_path=rel,
                            line=node.lineno,
                            end_line=end_ln,
                            signature=sig,
                            context_line=_get_line(node.lineno),
                            module_scope=scope,
                        )
                    )

                for sub in node.body:
                    if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)) and sub.name == short_name:
                        if class_qualifier and class_qualifier != node.name:
                            continue
                        s_end = getattr(sub, "end_lineno", sub.lineno)
                        msig = self._format_py_signature(sub, class_name=node.name)
                        s_decs = _extract_decorators(sub)
                        defs.append(
                            SymbolLocation(
                                symbol=f"{node.name}.{sub.name}",
                                kind="method",
                                rel_path=rel,
                                line=sub.lineno,
                                end_line=s_end,
                                signature=msig,
                                docstring=(ast.get_docstring(sub) or "")[:240],
                                context_line=_get_line(sub.lineno),
                                module_scope=scope,
                            )
                        )
                        if s_decs:
                            wiring.append(f"[{rel}:{sub.lineno}] `{node.name}.{sub.name}` decorated with {', '.join(s_decs)}")
                        if focus_rel_path is None or rel == focus_rel_path:
                            callees.extend(_extract_calls_from_node(sub))

            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == short_name:
                if class_qualifier is not None:
                    continue
                end_ln = getattr(node, "end_lineno", node.lineno)
                sig = self._format_py_signature(node)
                decs = _extract_decorators(node)
                defs.append(
                    SymbolLocation(
                        symbol=node.name,
                        kind="async_function" if isinstance(node, ast.AsyncFunctionDef) else "function",
                        rel_path=rel,
                        line=node.lineno,
                        end_line=end_ln,
                        signature=sig,
                        docstring=(ast.get_docstring(node) or "")[:240],
                        context_line=_get_line(node.lineno),
                        module_scope=scope,
                    )
                )
                if decs:
                    wiring.append(f"[{rel}:{node.lineno}] `{node.name}` decorated with {', '.join(decs)}")
                if focus_rel_path is None or rel == focus_rel_path:
                    callees.extend(_extract_calls_from_node(node))

        def_lines = {d.line for d in defs}
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                called = self._extract_call_name(node.func)
                if not called:
                    continue
                called_tail = called.split(".")[-1]
                if called_tail == short_name and node.lineno not in def_lines:
                    refs.append(
                        SymbolLocation(
                            symbol=called,
                            kind="reference",
                            rel_path=rel,
                            line=node.lineno,
                            end_line=getattr(node, "end_lineno", node.lineno),
                            context_line=_get_line(node.lineno),
                            module_scope=scope,
                        )
                    )
            elif isinstance(node, (ast.ImportFrom, ast.Import)):
                for alias in node.names:
                    if alias.name.split(".")[-1] == short_name or alias.asname == short_name:
                        refs.append(
                            SymbolLocation(
                                symbol=alias.name,
                                kind="reference",
                                rel_path=rel,
                                line=node.lineno,
                                end_line=getattr(node, "end_lineno", node.lineno),
                                context_line=_get_line(node.lineno),
                                module_scope=scope,
                            )
                        )

        return {
            "defs": defs,
            "refs": refs,
            "callees": callees,
            "supertypes": supertypes,
            "implementations": implementations,
            "wiring": wiring,
        }

    @staticmethod
    def _extract_call_name(func_node: ast.AST) -> str:
        if isinstance(func_node, ast.Name):
            return func_node.id
        if isinstance(func_node, ast.Attribute):
            if isinstance(func_node.value, ast.Name):
                return f"{func_node.value.id}.{func_node.attr}"
            return func_node.attr
        return ""

    @staticmethod
    def _format_py_signature(node: ast.FunctionDef | ast.AsyncFunctionDef, class_name: str | None = None) -> str:
        prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
        qual = f"{class_name}.{node.name}" if class_name else node.name
        args = [a.arg for a in node.args.args]
        ret = ""
        if node.returns is not None and hasattr(ast, "unparse"):
            try:
                ret = f" -> {ast.unparse(node.returns)}"
            except Exception:
                ret = ""
        return f"{prefix} {qual}({', '.join(args)}){ret}"

    def format_trace_report(self, report: CallHierarchyReport) -> str:
        lines = [
            f"=== Cross-File Semantic Trace: `{report.symbol}` (backend: {report.lsp_backend_used}) ==="
        ]
        if report.definitions:
            lines.append("\n### Definitions")
            for d in report.definitions:
                scp = f" [scope={d.module_scope}]" if d.module_scope and d.module_scope != "root" else ""
                lines.append(f"  • [{d.rel_path}:{d.line}-{d.end_line}]{scp} ({d.kind}) {d.signature or d.context_line}")
                if d.docstring:
                    lines.append(f"    Doc: {d.docstring.splitlines()[0]}")
        else:
            lines.append("\n### Definitions\n  (No definition found in repository)")

        if report.supertypes:
            lines.append("\n### Supertypes / Implemented Interfaces / Traits")
            lines.append("  • " + ", ".join(report.supertypes))

        if report.implementations:
            lines.append(f"\n### Subclasses / Interface & Trait Implementations ({len(report.implementations)})")
            for imp in report.implementations:
                scp = f" [scope={imp.module_scope}]" if imp.module_scope and imp.module_scope != "root" else ""
                lines.append(f"  • [{imp.rel_path}:{imp.line}-{imp.end_line}]{scp} `{imp.symbol}` — {imp.signature or imp.context_line}")

        if report.wiring_and_decorators:
            lines.append(f"\n### Decorators / Attributes / Dependency Wiring ({len(report.wiring_and_decorators)})")
            for b in report.wiring_and_decorators:
                lines.append(f"  • {b}")

        if report.declarative_config_links:
            lines.append(f"\n### Declarative Config / Schema / Template References ({len(report.declarative_config_links)})")
            for x in report.declarative_config_links:
                lines.append(f"  • {x}")

        if report.callers:
            lines.append(f"\n### Incoming Callers / References ({len(report.callers)})")
            for c in report.callers:
                lines.append(f"  • [{c.rel_path}:{c.line}] {c.context_line}")
        else:
            lines.append("\n### Incoming Callers / References\n  (No external callers found)")

        if report.callees:
            lines.append(f"\n### Outgoing Calls Inside `{report.symbol}`")
            lines.append("  • " + ", ".join(report.callees))

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # 2. Universal Static & Structural Diagnostics (`check_diagnostics`)
    # ------------------------------------------------------------------

    def check_diagnostics(self, rel_path: str | None = None) -> list[DiagnosticItem]:
        """
        Run fast static diagnostics across Python (AST & imports), C-family/Polyglot languages
        (brace/delimiter balance), and declarative configs (JSON/XML well-formedness).
        """
        items: list[DiagnosticItem] = []

        if rel_path:
            target = (self.repo_root / rel_path).resolve()
            files = [target] if target.exists() and target.is_file() else []
        else:
            files = list(self._iter_source_files(extensions=(".py", ".ts", ".js", ".go", ".rs", ".java", ".kt", ".c", ".cpp", ".xml", ".json")))

        for fpath in files:
            rel = str(fpath.relative_to(self.repo_root))
            suffix = fpath.suffix.lower()
            try:
                source = fpath.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue

            if suffix == ".py":
                items.extend(self._diagnose_python_file(rel, source))
            elif suffix in (".ts", ".tsx", ".js", ".jsx", ".go", ".rs", ".java", ".kt", ".c", ".cpp", ".cs", ".swift"):
                ok, err_ln, msg = check_brace_balance_in_source(source)
                if not ok:
                    items.append(
                        DiagnosticItem(
                            rel_path=rel,
                            line=err_ln,
                            col=1,
                            severity="ERROR",
                            code="UnbalancedDelimiters",
                            message=msg,
                        )
                    )
            elif suffix == ".xml":
                try:
                    ET.fromstring(source)
                except ET.ParseError as exc:
                    ln, col = getattr(exc, "position", (1, 1))
                    items.append(
                        DiagnosticItem(
                            rel_path=rel,
                            line=ln,
                            col=col,
                            severity="ERROR",
                            code="XMLParseError",
                            message=str(exc),
                        )
                    )
            elif suffix == ".json":
                try:
                    json.loads(source)
                except json.JSONDecodeError as exc:
                    items.append(
                        DiagnosticItem(
                            rel_path=rel,
                            line=exc.lineno,
                            col=exc.colno,
                            severity="ERROR",
                            code="JSONSyntaxError",
                            message=exc.msg,
                        )
                    )

        return items

    def _diagnose_python_file(self, rel: str, source: str) -> list[DiagnosticItem]:
        diags: list[DiagnosticItem] = []
        try:
            tree = ast.parse(source, filename=rel)
        except SyntaxError as exc:
            diags.append(
                DiagnosticItem(
                    rel_path=rel,
                    line=exc.lineno or 1,
                    col=exc.offset or 1,
                    severity="ERROR",
                    code="SyntaxError",
                    message=exc.msg,
                )
            )
            return diags

        defined_names: set[str] = set(PYTHON_BUILTINS)
        has_star_import = False

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                defined_names.add(node.name)
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    for arg in node.args.args + node.args.posonlyargs + node.args.kwonlyargs:
                        defined_names.add(arg.arg)
                    if node.args.vararg:
                        defined_names.add(node.args.vararg.arg)
                    if node.args.kwarg:
                        defined_names.add(node.args.kwarg.arg)
            elif isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Param)):
                defined_names.add(node.id)
            elif isinstance(node, ast.ExceptHandler) and node.name:
                defined_names.add(node.name)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    bound = alias.asname or alias.name.split(".")[0]
                    defined_names.add(bound)
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    if alias.name == "*":
                        has_star_import = True
                    else:
                        defined_names.add(alias.asname or alias.name)
                if node.module and node.level == 0:
                    rel_mod = node.module.replace(".", "/")
                    candidates = [
                        self.repo_root / f"{rel_mod}.py",
                        self.repo_root / "src" / f"{rel_mod}.py",
                        self.repo_root / rel_mod / "__init__.py",
                        self.repo_root / "src" / rel_mod / "__init__.py",
                    ]
                    for cand in candidates:
                        if cand.exists():
                            diags.extend(self._verify_imported_symbols(rel, node, cand))
                            break

        if not has_star_import:
            for node in ast.walk(tree):
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                    if node.id not in defined_names:
                        diags.append(
                            DiagnosticItem(
                                rel_path=rel,
                                line=node.lineno,
                                col=node.col_offset + 1,
                                severity="ERROR",
                                code="UndefinedName",
                                message=f"Name `{node.id}` is not defined or imported in `{rel}`",
                            )
                        )
        return diags

    def _verify_imported_symbols(
        self,
        importer_rel: str,
        node: ast.ImportFrom,
        target_file: Path,
    ) -> list[DiagnosticItem]:
        diags: list[DiagnosticItem] = []
        try:
            target_tree = ast.parse(target_file.read_text(encoding="utf-8", errors="replace"))
        except Exception:
            return diags

        exported: set[str] = set()
        for tnode in target_tree.body:
            if isinstance(tnode, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                exported.add(tnode.name)
            elif isinstance(tnode, ast.Assign):
                for t in tnode.targets:
                    if isinstance(t, ast.Name):
                        exported.add(t.id)
            elif isinstance(tnode, ast.AnnAssign) and isinstance(tnode.target, ast.Name):
                exported.add(tnode.target.id)
            elif isinstance(tnode, (ast.Import, ast.ImportFrom)):
                for alias in tnode.names:
                    exported.add(alias.asname or alias.name.split(".")[0])

        if not exported:
            return diags

        for alias in node.names:
            if alias.name != "*" and alias.name not in exported:
                if (target_file.parent / f"{alias.name}.py").exists():
                    continue
                diags.append(
                    DiagnosticItem(
                        rel_path=importer_rel,
                        line=node.lineno,
                        col=node.col_offset + 1,
                        severity="ERROR",
                        code="UnresolvedImportSymbol",
                        message=(
                            f"Cannot import `{alias.name}` from `{node.module}` "
                            f"({target_file.relative_to(self.repo_root)} does not define `{alias.name}`)"
                        ),
                    )
                )
        return diags

    @staticmethod
    def format_diagnostics(items: list[DiagnosticItem]) -> str:
        if not items:
            return "No static syntax, import, declarative, or type errors detected (0 diagnostics)."
        lines = [f"Found {len(items)} diagnostic issue(s):"]
        for d in items:
            lines.append(f"  • [{d.rel_path}:{d.line}:{d.col}] {d.severity} ({d.code}): {d.message}")
        return "\n".join(lines)
