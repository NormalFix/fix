"""Fix: Autonomous AI Engineering Partner — modular Codex-grade package architecture."""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.machinery
import sys
from types import ModuleType

__version__ = "1.0.0-beta"

_LEGACY_MODULE_MAP: dict[str, str] = {
    "fix.schema": "fix.protocol.schema",
    "fix.runtime": "fix.core.runtime",
    "fix.context": "fix.core.context",
    "fix.looper": "fix.core.looper",
    "fix.autonomy": "fix.core.autonomy",
    "fix.evolution": "fix.core.evolution",
    "fix.lsp": "fix.tools.lsp",
    "fix.profiler": "fix.tools.profiler",
}


class _LegacyAliasLoader(importlib.abc.Loader):
    """Zero-overhead loader that aliases legacy flat module paths (`fix.runtime`, `fix.schema`, etc.) to their domain packages."""

    def __init__(self, alias_name: str, target_name: str) -> None:
        self.alias_name = alias_name
        self.target_name = target_name

    def create_module(self, spec: importlib.machinery.ModuleSpec) -> ModuleType:
        target_mod = importlib.import_module(self.target_name)
        sys.modules[self.alias_name] = target_mod
        short_attr = self.alias_name.split(".", 1)[1]
        pkg = sys.modules.get("fix")
        if pkg is not None:
            setattr(pkg, short_attr, target_mod)
        return target_mod

    def exec_module(self, module: ModuleType) -> None:
        pass


class _LegacyAliasFinder(importlib.abc.MetaPathFinder):
    def find_spec(
        self,
        fullname: str,
        path: object = None,
        target: object = None,
    ) -> importlib.machinery.ModuleSpec | None:
        target_name = _LEGACY_MODULE_MAP.get(fullname)
        if target_name is None:
            return None
        return importlib.machinery.ModuleSpec(
            name=fullname,
            loader=_LegacyAliasLoader(fullname, target_name),
        )


if not any(isinstance(f, _LegacyAliasFinder) for f in sys.meta_path):
    sys.meta_path.insert(0, _LegacyAliasFinder())
