#!/usr/bin/env python3
"""
check_model.py — Validador exaustivo de modelos MiiaModel para AsaMiia.

Verifica toda a cadeia que o PythonBackend C++ percorre:
  importação → classe → instanciação → load → get_schema → predict → outputs → unload

Uso:
    python check_model.py <modelo.py> [opções]

Opções:
    --models-dir DIR     Diretório com miia_model.py  (default: dir do modelo)
    --predict            Executa predict() com inputs sintéticos
    --predict-n N        Repete predict() N vezes     (default: 1)
    --no-color           Desativa cores ANSI
    --verbose / -v       Exibe detalhes extras
    --json               Saída em JSON (para CI)
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import inspect
import io
import json
import math
import re
import sys
import textwrap
import time
import traceback
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ─────────────────────────────────────────────────────────────────────────────
# ANSI helpers
# ─────────────────────────────────────────────────────────────────────────────

USE_COLOR = True


def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if USE_COLOR else text


def _ok(msg):   return f"  {_c('32', '✓')} {msg}"
def _err(msg):  return f"  {_c('31', '✗')} {_c('31', msg)}"
def _warn(msg): return f"  {_c('33', '⚠')} {_c('33', msg)}"
def _info(msg): return f"  {_c('2',  '→')} {_c('2',  msg)}"
def _note(msg): return f"    {_c('2', msg)}"


def _section(title: str) -> str:
    bar = "─" * (len(title) + 4)
    return f"\n{_c('36;1', bar)}\n  {_c('36;1', title)}\n{_c('36;1', bar)}"


# ─────────────────────────────────────────────────────────────────────────────
# Resultado acumulado
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Finding:
    level: str   # "error" | "warning" | "info"
    phase: str
    message: str


@dataclass
class Report:
    findings: list[Finding] = field(default_factory=list)

    def error(self, phase: str, msg: str):
        self.findings.append(Finding("error", phase, msg))
        print(_err(msg))

    def warn(self, phase: str, msg: str):
        self.findings.append(Finding("warning", phase, msg))
        print(_warn(msg))

    def ok(self, msg: str):
        print(_ok(msg))

    def info(self, msg: str):
        print(_info(msg))

    def note(self, msg: str):
        print(_note(msg))

    @property
    def errors(self):   return [f for f in self.findings if f.level == "error"]
    @property
    def warnings(self): return [f for f in self.findings if f.level == "warning"]

    def to_dict(self) -> dict:
        return {
            "valid": len(self.errors) == 0,
            "errors":   [{"phase": f.phase, "message": f.message} for f in self.errors],
            "warnings": [{"phase": f.phase, "message": f.message} for f in self.warnings],
        }


# ─────────────────────────────────────────────────────────────────────────────
# Context manager: captura stdout/stderr de chamadas Python
# ─────────────────────────────────────────────────────────────────────────────

@contextmanager
def _capture():
    buf_out, buf_err = io.StringIO(), io.StringIO()
    with redirect_stdout(buf_out), redirect_stderr(buf_err):
        yield buf_out, buf_err


# ─────────────────────────────────────────────────────────────────────────────
# FASE 0 — Análise estática do código-fonte (sem importar)
# ─────────────────────────────────────────────────────────────────────────────

def _collect_all_bases_ast(cls_node: ast.ClassDef) -> list[str]:
    """Extrai nomes de bases de uma ClassDef (Name e Attribute)."""
    bases = []
    for b in cls_node.bases:
        if isinstance(b, ast.Name):
            bases.append(b.id)
        elif isinstance(b, ast.Attribute):
            bases.append(b.attr)
    return bases


def _build_inheritance_map_ast(tree: ast.AST) -> dict[str, list[str]]:
    """Mapeia nome_da_classe → lista de bases (nível AST, sem resolver imports)."""
    return {
        n.name: _collect_all_bases_ast(n)
        for n in ast.walk(tree)
        if isinstance(n, ast.ClassDef)
    }


def _is_miia_subclass_ast(class_name: str,
                           inheritance_map: dict[str, list[str]],
                           visited: set[str] | None = None) -> bool:
    """
    Verifica recursivamente se class_name é subclasse de MiiaModel no AST.
    Suporta herança indireta (ex.: PilotBT → PilotBTModel → MiiaModel).
    """
    if visited is None:
        visited = set()
    if class_name in visited:
        return False
    visited.add(class_name)

    bases = inheritance_map.get(class_name, [])
    if "MiiaModel" in bases:
        return True
    return any(
        _is_miia_subclass_ast(b, inheritance_map, visited)
        for b in bases
    )


def phase_static_analysis(path: Path, rep: Report, verbose: bool):
    print(_section("0. Análise Estática (AST)"))

    src = path.read_text(encoding="utf-8")

    try:
        tree = ast.parse(src, filename=str(path))
    except SyntaxError as e:
        rep.error("static", f"SyntaxError: {e}")
        return None

    rep.ok("Sintaxe Python válida")

    # Imports presentes
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module or "")

    if "miia_model" not in " ".join(imports):
        rep.error("static", "miia_model não é importado — backend não conseguirá encontrar subclasse de MiiaModel")
    else:
        rep.ok("miia_model importado")

    if "numpy" not in " ".join(imports) and "np" not in src:
        rep.warn("static", "numpy não parece importado — predict() deve retornar np.ndarray")

    # Classes no módulo
    class_defs = [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
    if not class_defs:
        rep.error("static", "Nenhuma classe definida no arquivo")
        return tree

    rep.ok(f"{len(class_defs)} classe(s) encontrada(s): {[c.name for c in class_defs]}")

    # Mapa de herança para resolução recursiva
    inheritance_map = _build_inheritance_map_ast(tree)

    # Subclasses de MiiaModel (diretas OU indiretas) no AST
    miia_subclasses = [
        c.name for c in class_defs
        if _is_miia_subclass_ast(c.name, inheritance_map)
    ]

    # Separa diretas de indiretas para informação
    direct    = [c.name for c in class_defs if "MiiaModel" in _collect_all_bases_ast(c)]
    indirect  = [n for n in miia_subclasses if n not in direct]

    # Nomes importados de miia_model (ex.: PilotBTModel) — bases externas válidas
    miia_module_imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "miia_model":
            for alias in node.names:
                miia_module_imports.add(alias.asname or alias.name)

    # Classes cujas bases diretas incluem algo importado de miia_model
    # (herança indireta via intermediária externa — não rastreável no AST local)
    external_chain = [
        c.name for c in class_defs
        if c.name not in miia_subclasses
        and any(b in miia_module_imports for b in _collect_all_bases_ast(c))
    ]
    if external_chain:
        rep.ok(f"Subclasse(s) via intermediária externa de miia_model: {external_chain}")
        miia_subclasses = miia_subclasses + external_chain

    if not miia_subclasses:
        rep.error("static", "Nenhuma classe herda de MiiaModel (direta ou indiretamente)")
    else:
        if direct:
            rep.ok(f"Subclasse(s) diretas de MiiaModel: {direct}")
        if indirect:
            rep.ok(f"Subclasse(s) indiretas de MiiaModel: {indirect} (herança via intermediária)")
        if len(miia_subclasses) > 1:
            rep.warn("static", (
                f"Múltiplas subclasses de MiiaModel — backend C++ usa a primeira "
                f"CONCRETA (sem __abstractmethods__) encontrada no dict do módulo"
            ))

    # Métodos obrigatórios — verifica apenas os definidos localmente no arquivo.
    # Se a classe herda diretamente de uma base importada de miia_model, métodos
    # ausentes no AST local podem estar na cadeia externa — a fase 3 (runtime)
    # resolve isso com precisão via MRO real, sem falsos positivos.
    REQUIRED = {"load", "predict", "get_schema"}
    for cls_name in miia_subclasses:
        cls_node = next((n for n in class_defs if n.name == cls_name), None)
        if cls_node is None:
            continue  # definida em outro módulo (importada)

        defined = {n.name for n in ast.walk(cls_node) if isinstance(n, ast.FunctionDef)}
        missing = REQUIRED - defined

        # Bases diretas desta classe que foram importadas de miia_model
        bases_from_miia = [
            b for b in _collect_all_bases_ast(cls_node)
            if b in miia_module_imports
        ]

        if missing and bases_from_miia:
            # Métodos ausentes podem estar na base externa — não reportar aqui
            rep.ok(f"{cls_name}: métodos locais verificados (herança via {bases_from_miia} validada em runtime)")
        elif missing:
            rep.error("static", f"{cls_name}: métodos obrigatórios ausentes: {sorted(missing)}")
        else:
            rep.ok(f"{cls_name}: métodos load/predict/get_schema presentes")

        # __init__ com parâmetros além de self
        for node in ast.walk(cls_node):
            if isinstance(node, ast.FunctionDef) and node.name == "__init__":
                non_self = [a.arg for a in node.args.args if a.arg != "self"]
                req_args = non_self[len(node.args.defaults):]
                if req_args:
                    rep.error("static", (
                        f"{cls_name}.__init__ tem parâmetros obrigatórios {req_args} "
                        f"— backend instancia sem argumentos"
                    ))
                else:
                    rep.ok(f"{cls_name}.__init__ sem parâmetros obrigatórios")

    # Heurística: uso de print()
    print_calls = [
        node.lineno for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(getattr(node, "func", None), ast.Name)
        and node.func.id == "print"
    ]
    if print_calls:
        lines = ", ".join(str(l) for l in print_calls[:8])
        suffix = f" (e mais {len(print_calls)-8})" if len(print_calls) > 8 else ""
        rep.warn("static", f"print() nas linhas {lines}{suffix} — usa asalog em vez de print()")

    # sys.exit / os._exit
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            fn = node.func
            if isinstance(fn, ast.Attribute) and fn.attr in ("exit", "_exit"):
                rep.error("static", f"Linha {node.lineno}: sys.exit/os._exit em modelo causa crash do worker")

    return tree


# ─────────────────────────────────────────────────────────────────────────────
# FASE 1 — Importação do módulo
# ─────────────────────────────────────────────────────────────────────────────

def phase_import(path: Path, models_dir: Path, rep: Report):
    print(_section("1. Importação"))

    for p in [str(models_dir), str(path.parent)]:
        if p not in sys.path:
            sys.path.insert(0, p)

    try:
        import miia_model as _mm
        rep.ok(f"miia_model importado de: {_mm.__file__}")
        MiiaModel   = _mm.MiiaModel
        ModelSchema = _mm.ModelSchema
        TensorSpec  = _mm.TensorSpec
    except ImportError as e:
        rep.error("import", f"Falha ao importar miia_model: {e}")
        rep.info("Use --models-dir para apontar o diretório que contém miia_model.py")
        return None, None, None, None

    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod  = importlib.util.module_from_spec(spec)

    with _capture() as (out, err_buf):
        try:
            spec.loader.exec_module(mod)
        except Exception:
            rep.error("import", f"Exceção ao importar módulo '{path.stem}':")
            for line in traceback.format_exc().splitlines():
                print(_note(line))
            return None, MiiaModel, ModelSchema, TensorSpec

    # rep.ok fora do _capture para não aparecer como stdout capturado
    rep.ok(f"Módulo '{path.stem}' importado sem exceção")
    captured_out = out.getvalue().strip()
    if captured_out:
        rep.warn("import", f"Módulo emitiu stdout durante import: {captured_out[:200]!r}")

    return mod, MiiaModel, ModelSchema, TensorSpec


# ─────────────────────────────────────────────────────────────────────────────
# FASE 2 — Descoberta da classe
#
# Replica EXATAMENTE o comportamento do C++ após o fix de find_model_class():
#   - Itera module.__dict__ em ordem de inserção (Python 3.7+)
#   - Pula MiiaModel em si
#   - Pula classes com __abstractmethods__ não-vazio (classes abstratas)
#   - Retorna a PRIMEIRA classe concreta encontrada
# ─────────────────────────────────────────────────────────────────────────────

def phase_find_class(mod, MiiaModel, rep: Report):
    print(_section("2. Descoberta da Classe (replica find_model_class + fix abstrata)"))

    all_subclasses    = []   # todas as subclasses (inclusive abstratas)
    concrete_classes  = []   # apenas as concretas (sem __abstractmethods__)
    abstract_classes  = []   # abstratas ignoradas pelo backend

    for name, obj in vars(mod).items():
        if not inspect.isclass(obj):
            continue
        if obj is MiiaModel:
            continue
        try:
            if not issubclass(obj, MiiaModel):
                continue
        except TypeError:
            continue

        all_subclasses.append((name, obj))

        abstract_methods = getattr(obj, "__abstractmethods__", frozenset())
        if abstract_methods:
            abstract_classes.append((name, obj, set(abstract_methods)))
        else:
            concrete_classes.append((name, obj))

    if not all_subclasses:
        rep.error("class", "Nenhuma subclasse de MiiaModel encontrada no módulo")
        return None

    # Informa classes abstratas ignoradas
    for name, _, abst in abstract_classes:
        rep.info(
            f"Classe abstrata ignorada pelo backend: '{name}' "
            f"(__abstractmethods__={sorted(abst)})"
        )

    if not concrete_classes:
        rep.error("class", (
            f"Nenhuma classe CONCRETA encontrada — todas as subclasses ainda têm "
            f"métodos abstratos pendentes: "
            + ", ".join(f"'{n}' {sorted(a)}" for n, _, a in abstract_classes)
        ))
        return None

    # Backend usa a primeira concreta (ordem de inserção no dict do módulo)
    name, cls = concrete_classes[0]
    rep.ok(f"Classe concreta selecionada pelo backend: '{name}'")

    if len(concrete_classes) > 1:
        extra = [n for n, _ in concrete_classes[1:]]
        rep.warn("class", (
            f"Múltiplas classes concretas: {[n for n, _ in concrete_classes]} "
            f"— backend usará '{name}', ignorará {extra}"
        ))

    # Verifica se a classe é do próprio módulo ou importada
    if cls.__module__ != mod.__name__:
        rep.warn("class", (
            f"'{name}' é definida em '{cls.__module__}', não em '{mod.__name__}' "
            f"— pode ser importada de outro módulo"
        ))
    else:
        rep.ok(f"'{name}' definida no próprio módulo (✓)")

    # MRO completo
    mro_names = [c.__name__ for c in cls.__mro__]
    rep.ok(f"MRO: {' → '.join(mro_names)}")
    if "MiiaModel" not in mro_names:
        rep.error("class", "MiiaModel ausente no MRO — herança incorreta")

    # Destaca a cadeia de herança intermediária
    intermediaries = [n for n in mro_names if n not in (name, "MiiaModel", "ABC", "object")]
    if intermediaries:
        rep.info(f"Classes intermediárias na cadeia: {intermediaries}")

    return cls


# ─────────────────────────────────────────────────────────────────────────────
# FASE 3 — Inspeção da interface (sem instanciar)
# ─────────────────────────────────────────────────────────────────────────────

REQUIRED_METHODS = ["load", "predict", "get_schema"]
OPTIONAL_METHODS = ["unload", "validate_inputs", "warmup", "memory_usage_bytes"]

VALID_DTYPES = {
    "float32", "float64", "float16",
    "int8", "int16", "int32", "int64",
    "uint8", "uint16", "uint32", "uint64",
    "bool",
}


def phase_interface(cls, rep: Report):
    print(_section("3. Interface"))

    for m in REQUIRED_METHODS:
        attr = getattr(cls, m, None)
        if attr is None or not callable(attr):
            rep.error("interface", f"Método obrigatório ausente: {m}()")
        else:
            # Verifica se está resolvido na MRO (não apenas herdado como abstrato)
            abstract = getattr(cls, "__abstractmethods__", frozenset())
            if m in abstract:
                rep.error("interface", f"{m}() ainda abstrato — não implementado em nenhuma classe da MRO")
            else:
                # Informa onde na MRO está implementado
                for klass in cls.__mro__:
                    if m in klass.__dict__:
                        source = klass.__name__
                        break
                else:
                    source = "?"
                rep.ok(f"{m}() presente (implementado em '{source}')")

    for m in OPTIONAL_METHODS:
        if callable(getattr(cls, m, None)):
            for klass in cls.__mro__:
                if m in klass.__dict__:
                    source = klass.__name__
                    break
            else:
                source = "?"
            rep.ok(f"{m}() implementado em '{source}' (opcional)")
        else:
            rep.info(f"{m}() não implementado (opcional, usa default da base)")

    # Assinatura de predict
    try:
        sig    = inspect.signature(cls.predict)
        params = list(sig.parameters.keys())
        if "self" not in params:
            rep.error("interface", "predict() não tem parâmetro 'self'")
        if "inputs" not in params:
            rep.warn("interface", f"predict() parâmetro esperado 'inputs' não encontrado; parâmetros: {params}")
        else:
            rep.ok("predict(self, inputs) — assinatura correta")
    except Exception as e:
        rep.warn("interface", f"Não foi possível inspecionar assinatura de predict(): {e}")

    # __init__ sem parâmetros obrigatórios
    try:
        sig    = inspect.signature(cls.__init__)
        params = {k: v for k, v in sig.parameters.items() if k != "self"}
        req    = [
            k for k, v in params.items()
            if v.default is inspect.Parameter.empty
            and v.kind not in (v.VAR_POSITIONAL, v.VAR_KEYWORD)
        ]
        if req:
            rep.error("interface", f"__init__ tem parâmetros obrigatórios: {req} — backend instancia sem args")
        else:
            rep.ok("__init__ sem parâmetros obrigatórios")
    except Exception as e:
        rep.warn("interface", f"Não foi possível inspecionar __init__: {e}")

    # __abstractmethods__ — confirmação final
    abstract = getattr(cls, "__abstractmethods__", frozenset())
    if abstract:
        rep.error("interface", (
            f"Métodos abstratos não implementados: {sorted(abstract)} "
            f"— cls() vai lançar TypeError"
        ))
    else:
        rep.ok("Nenhum método abstrato pendente — classe instanciável")


# ─────────────────────────────────────────────────────────────────────────────
# FASE 4 — Instanciação
# ─────────────────────────────────────────────────────────────────────────────

def phase_instantiate(cls, rep: Report):
    print(_section("4. Instanciação  [cls()]"))
    try:
        with _capture() as (out, _):
            instance = cls()
        rep.ok(f"{cls.__name__}() instanciado com sucesso")
        captured = out.getvalue().strip()
        if captured:
            rep.warn("instantiate", f"__init__ emitiu stdout: {captured[:200]!r}")
        return instance
    except TypeError as e:
        rep.error("instantiate", f"TypeError: {e}")
        rep.note("Provável: __init__ com parâmetros obrigatórios ou abstract method não implementado")
    except Exception:
        rep.error("instantiate", "Exceção inesperada em __init__:")
        for line in traceback.format_exc().splitlines():
            rep.note(line)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# FASE 5 — load()
# ─────────────────────────────────────────────────────────────────────────────

def phase_load(instance, rep: Report):
    print(_section("5. load()"))
    t0 = time.perf_counter()
    try:
        with _capture() as (out, err_buf):
            instance.load()
        ms = (time.perf_counter() - t0) * 1000
        rep.ok(f"load() concluiu em {ms:.2f} ms")

        if out.getvalue().strip():
            rep.warn("load", f"load() emitiu stdout: {out.getvalue().strip()[:200]!r}")
        if err_buf.getvalue().strip():
            rep.warn("load", f"load() emitiu stderr: {err_buf.getvalue().strip()[:200]!r}")

        return True, ms
    except Exception:
        rep.error("load", "Exceção em load():")
        for line in traceback.format_exc().splitlines():
            rep.note(line)
        return False, 0.0


# ─────────────────────────────────────────────────────────────────────────────
# FASE 6 — get_schema() — validação exaustiva
# ─────────────────────────────────────────────────────────────────────────────

def phase_schema(instance, ModelSchema, TensorSpec, rep: Report, verbose: bool):
    print(_section("6. get_schema()"))

    try:
        with _capture():
            schema = instance.get_schema()
    except Exception:
        rep.error("schema", "get_schema() lançou exceção:")
        for line in traceback.format_exc().splitlines():
            rep.note(line)
        return None

    if not isinstance(schema, ModelSchema):
        rep.error("schema", f"Retornou {type(schema).__name__} em vez de ModelSchema")
        return None
    rep.ok("Retornou ModelSchema")

    if schema.description:
        rep.ok(f"description: '{schema.description}'")
    else:
        rep.warn("schema", "description vazia — boa prática preencher")

    if schema.author:
        rep.ok(f"author: '{schema.author}'")

    _validate_spec_list("inputs",  schema.inputs,  TensorSpec, rep, verbose)
    _validate_spec_list("outputs", schema.outputs, TensorSpec, rep, verbose)

    input_names  = {ts.name for ts in schema.inputs}
    output_names = {ts.name for ts in schema.outputs}
    overlap = input_names & output_names
    if overlap:
        rep.warn("schema", f"Nomes compartilhados entre inputs e outputs: {overlap}")

    for label, specs in [("inputs", schema.inputs), ("outputs", schema.outputs)]:
        names = [ts.name for ts in specs]
        seen  = set()
        for n in names:
            if n in seen:
                rep.error("schema", f"Nome duplicado em {label}: '{n}'")
            seen.add(n)

    return schema


def _validate_spec_list(label: str, specs, TensorSpec, rep: Report, verbose: bool):
    if not specs:
        rep.warn("schema", f"schema.{label} está vazio")
        return

    rep.ok(f"schema.{label}: {len(specs)} TensorSpec(s)")

    for i, ts in enumerate(specs):
        prefix = f"{label}[{i}] '{ts.name}'"

        if not isinstance(ts, TensorSpec):
            rep.error("schema", f"{prefix}: não é TensorSpec, é {type(ts).__name__}")
            continue

        if not ts.name or not ts.name.strip():
            rep.error("schema", f"{label}[{i}]: name vazio")
        elif not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', ts.name):
            rep.warn("schema", f"{prefix}: nome com caracteres não-identificador")
        else:
            rep.ok(f"{prefix}: nome válido")

        if not ts.shape:
            rep.warn("schema", f"{prefix}: shape vazio — considere [-1] para dinâmico")
        else:
            for j, d in enumerate(ts.shape):
                if not isinstance(d, int):
                    rep.error("schema", f"{prefix}: shape[{j}]={d!r} não é int")
                elif d == 0:
                    rep.error("schema", f"{prefix}: shape[{j}]=0 — dimensão zero inválida")
                elif d < -1:
                    rep.warn("schema", f"{prefix}: shape[{j}]={d} — valor negativo diferente de -1")
            rep.ok(f"{prefix}: shape={ts.shape}")

        if ts.dtype not in VALID_DTYPES:
            rep.warn("schema", f"{prefix}: dtype='{ts.dtype}' não é dtype numpy padrão")
        else:
            rep.ok(f"{prefix}: dtype={ts.dtype}")

        if ts.structured:
            rep.ok(f"{prefix}: structured=True (input dict/list aninhado)")
            if label == "outputs":
                rep.warn("schema", f"{prefix}: output com structured=True — py_dict_to_outputs espera np.ndarray")
            if ts.shape and ts.shape != [-1]:
                rep.warn("schema", f"{prefix}: structured=True mas shape={ts.shape} — use [-1]")
        else:
            if label == "inputs" and (ts.shape == [-1] or not ts.shape):
                rep.warn("schema", f"{prefix}: structured=False mas shape={ts.shape} — se for dict/list, declare structured=True")

        if ts.min_value is not None and ts.max_value is not None:
            if ts.min_value >= ts.max_value:
                rep.error("schema", f"{prefix}: min_value={ts.min_value} >= max_value={ts.max_value}")
            else:
                rep.ok(f"{prefix}: range=[{ts.min_value}, {ts.max_value}]")


# ─────────────────────────────────────────────────────────────────────────────
# FASE 7 — predict()
# ─────────────────────────────────────────────────────────────────────────────

def phase_predict(instance, schema, rep: Report, n_runs: int, verbose: bool):
    print(_section("7. predict()"))

    if schema is None:
        rep.warn("predict", "Schema inválido — predict() não será executado")
        return

    import numpy as np

    inputs = _build_inputs(schema, rep)
    if verbose:
        rep.info(f"Inputs sintéticos: {_summarize_value(inputs)}")

    times    = []
    results  = []
    first_tb = None

    for i in range(n_runs):
        t0 = time.perf_counter()
        try:
            with _capture() as (out, err_buf):
                result = instance.predict(inputs)
            ms = (time.perf_counter() - t0) * 1000
            times.append(ms)
            results.append(result)

            if out.getvalue().strip() and i == 0:
                rep.warn("predict", f"predict() emitiu stdout (run 0): {out.getvalue().strip()[:200]!r}")
        except Exception:
            ms = (time.perf_counter() - t0) * 1000
            if first_tb is None:
                first_tb = traceback.format_exc()
            rep.error("predict", f"predict() lançou exceção (run {i}, {ms:.2f} ms):")
            for line in first_tb.splitlines():
                rep.note(line)
            return

    if n_runs > 0:
        rep.ok(f"predict() executou {n_runs}x sem exceção")
        if n_runs > 1:
            avg = sum(times) / len(times)
            rep.ok(f"Latência — avg={avg:.2f} ms  min={min(times):.2f} ms  max={max(times):.2f} ms")

    result = results[0] if results else None
    if result is None:
        return

    if not isinstance(result, dict):
        rep.error("predict", f"predict() retornou {type(result).__name__} em vez de dict")
        return
    rep.ok("predict() retornou dict")

    declared_out = {ts.name for ts in schema.outputs}
    returned     = set(result.keys())
    missing = declared_out - returned
    extra   = returned   - declared_out

    if missing:
        rep.error("predict", f"Outputs declarados no schema mas AUSENTES no resultado: {sorted(missing)}")
    if extra:
        rep.warn("predict", f"Outputs presentes no resultado mas NÃO declarados no schema: {sorted(extra)}")

    for key, val in result.items():
        _validate_output_value(key, val, schema, rep, verbose)

    if n_runs > 1:
        _check_determinism(results, schema, rep)

    for k in result:
        if not isinstance(k, str):
            rep.error("predict", f"Chave de output não é string: {k!r} — py_dict_to_outputs vai falhar")


def _validate_output_value(key: str, val: Any, schema, rep: Report, verbose: bool):
    import numpy as np

    prefix = f"output '{key}'"

    try:
        arr = np.asarray(val, dtype=np.float32)
    except Exception as e:
        rep.error("predict", f"{prefix}: não conversível para float32 array: {e}")
        return

    rep.ok(f"{prefix}: conversível para float32 array")

    shape_str = str(list(arr.shape))
    total     = arr.size
    rep.ok(f"{prefix}: shape={shape_str}  total={total}")

    if total == 0:
        rep.error("predict", f"{prefix}: array VAZIO (total=0)")
    elif total == 1:
        v = float(arr.flat[0])
        rep.ok(f"{prefix}: escalar {v:.6g} (shape={shape_str}, C++ entrega como is_number())")
        if math.isnan(v):
            rep.error("predict", f"{prefix}: valor é NaN")
        elif math.isinf(v):
            rep.error("predict", f"{prefix}: valor é Inf")
    else:
        rep.ok(f"{prefix}: total={total} → C++ produz Value{{Array}}")

    if total > 0:
        if np.any(np.isnan(arr)):
            rep.error("predict", f"{prefix}: array contém NaN")
        if np.any(np.isinf(arr)):
            rep.error("predict", f"{prefix}: array contém Inf")

    schema_spec = next((ts for ts in schema.outputs if ts.name == key), None)
    if schema_spec and not schema_spec.structured and schema_spec.shape:
        expected = schema_spec.shape
        actual   = list(arr.shape)
        if len(actual) != len(expected):
            rep.warn("predict", (
                f"{prefix}: schema declara {len(expected)}D shape={expected}, "
                f"resultado tem {len(actual)}D shape={actual}"
            ))
        else:
            for i, (e, a) in enumerate(zip(expected, actual)):
                if e != -1 and e != a:
                    rep.warn("predict", f"{prefix}: dim[{i}] schema={e} real={a}")

    if verbose:
        rep.info(f"{prefix}: tipo original Python = {type(val).__name__}")


def _check_determinism(results: list, schema, rep: Report):
    import numpy as np
    rep.ok("Verificando determinismo entre runs...")
    for ts in schema.outputs:
        key  = ts.name
        vals = [results[i].get(key) for i in range(len(results))]
        if any(v is None for v in vals):
            continue
        try:
            arrs = [np.asarray(v, dtype=np.float32) for v in vals]
            ref  = arrs[0]
            for i, a in enumerate(arrs[1:], start=1):
                if ref.shape != a.shape:
                    rep.warn("predict", f"output '{key}': shape divergiu entre runs")
                elif not np.allclose(ref, a, atol=1e-5, rtol=1e-5, equal_nan=True):
                    diff = np.max(np.abs(a - ref))
                    rep.warn("predict", f"output '{key}': resultado divergiu entre runs (max_diff={diff:.2e})")
            else:
                rep.ok(f"output '{key}': determinístico entre {len(results)} runs")
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# FASE 8 — validate_inputs()
# ─────────────────────────────────────────────────────────────────────────────

def phase_validate_inputs(instance, schema, rep: Report, verbose: bool):
    if not callable(getattr(instance, "validate_inputs", None)):
        return
    print(_section("8. validate_inputs()"))

    inputs = _build_inputs(schema, rep)

    try:
        vr = instance.validate_inputs(inputs)
        if vr.valid:
            rep.ok("validate_inputs(inputs_sintéticos): valid=True")
        else:
            rep.warn("validate_inputs", f"validate_inputs retornou valid=False: {vr.errors}")
    except Exception:
        rep.error("validate_inputs", "validate_inputs() lançou exceção:")
        for line in traceback.format_exc().splitlines():
            rep.note(line)


# ─────────────────────────────────────────────────────────────────────────────
# FASE 9 — unload()
# ─────────────────────────────────────────────────────────────────────────────

def phase_unload(instance, rep: Report):
    print(_section("9. unload()"))
    if not callable(getattr(instance, "unload", None)):
        rep.info("unload() não implementado (usa no-op da base)")
        return

    try:
        with _capture():
            instance.unload()
        rep.ok("unload() executou sem exceção")
    except Exception:
        rep.error("unload", "unload() lançou exceção:")
        for line in traceback.format_exc().splitlines():
            rep.note(line)

    try:
        with _capture():
            instance.unload()
        rep.ok("Segunda chamada a unload() segura (idempotente)")
    except Exception:
        rep.warn("unload", "Segunda chamada a unload() lançou exceção — não é idempotente")


# ─────────────────────────────────────────────────────────────────────────────
# FASE 10 — get_schema() após unload
# ─────────────────────────────────────────────────────────────────────────────

def phase_schema_after_unload(instance, ModelSchema, rep: Report):
    print(_section("10. get_schema() após unload()"))
    try:
        with _capture():
            schema2 = instance.get_schema()
        if isinstance(schema2, ModelSchema):
            rep.ok("get_schema() acessível após unload() (✓)")
        else:
            rep.warn("schema_after_unload", f"get_schema() após unload retornou {type(schema2).__name__}")
    except Exception:
        rep.warn("schema_after_unload", "get_schema() lançou exceção após unload()")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers de construção de inputs sintéticos
# ─────────────────────────────────────────────────────────────────────────────

def _build_inputs(schema, rep: Report) -> dict:
    import numpy as np
    inputs = {}
    for ts in schema.inputs:
        if ts.structured:
            inputs[ts.name] = {}
            rep.info(f"Input '{ts.name}' é structured=True — usando {{}} como input sintético")
        else:
            shape = [max(1, d) for d in (ts.shape or [1])]
            try:
                arr = np.ones(shape, dtype=ts.dtype)
                if ts.min_value is not None:
                    arr = np.clip(arr, ts.min_value, None)
                if ts.max_value is not None:
                    arr = np.clip(arr, None, ts.max_value)
                inputs[ts.name] = arr.tolist()
            except Exception as e:
                inputs[ts.name] = [1.0]
                rep.warn("predict", f"Não foi possível construir input '{ts.name}': {e} — usando [1.0]")
    return inputs


def _summarize_value(v: Any, depth: int = 0) -> str:
    if isinstance(v, dict):
        inner = ", ".join(f"{k}: {_summarize_value(vv, depth+1)}" for k, vv in list(v.items())[:4])
        return "{" + inner + ("..." if len(v) > 4 else "") + "}"
    if isinstance(v, list):
        if not v: return "[]"
        return f"[{_summarize_value(v[0])}×{len(v)}]"
    return repr(v)[:40]


# ─────────────────────────────────────────────────────────────────────────────
# Resumo final
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(rep: Report):
    print(_section("Resumo"))
    n_err  = len(rep.errors)
    n_warn = len(rep.warnings)

    if n_err == 0 and n_warn == 0:
        print(f"  {_c('32;1', '✓ VÁLIDO')} — nenhum erro ou aviso encontrado.\n")
        return True

    if n_err == 0:
        print(f"  {_c('33;1', f'✓ VÁLIDO com {n_warn} aviso(s)')}\n")
    else:
        print(f"  {_c('31;1', f'✗ INVÁLIDO — {n_err} erro(s), {n_warn} aviso(s)')}\n")

    by_phase: dict[str, list] = {}
    for f in rep.findings:
        by_phase.setdefault(f.phase, []).append(f)

    for phase, findings in by_phase.items():
        errs  = [f for f in findings if f.level == "error"]
        warns = [f for f in findings if f.level == "warning"]
        if errs:
            print(f"  {_c('31', f'[{phase}] {len(errs)} erro(s):')}")
            for f in errs:
                print(f"    {_c('31', '•')} {f.message}")
        if warns:
            print(f"  {_c('33', f'[{phase}] {len(warns)} aviso(s):')}")
            for f in warns:
                print(f"    {_c('33', '•')} {f.message}")

    print()
    return n_err == 0


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    global USE_COLOR

    parser = argparse.ArgumentParser(
        description="Validador exaustivo de modelos MiiaModel para AsaMiia.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("model", help="Caminho para o arquivo .py do modelo")
    parser.add_argument("--models-dir", default=None,
                        help="Diretório com miia_model.py (default: mesmo dir do modelo)")
    parser.add_argument("--predict", action="store_true",
                        help="Executa predict() com inputs sintéticos")
    parser.add_argument("--predict-n", type=int, default=1, metavar="N",
                        help="Número de execuções de predict() (default: 1)")
    parser.add_argument("--no-color", action="store_true",
                        help="Desativa cores ANSI")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Exibe detalhes extras")
    parser.add_argument("--json", action="store_true",
                        help="Saída em JSON para CI")
    args = parser.parse_args()

    if args.no_color or args.json:
        USE_COLOR = False

    model_path = Path(args.model).resolve()
    models_dir = Path(args.models_dir).resolve() if args.models_dir else model_path.parent

    run_predict = args.predict or args.predict_n > 1
    n_runs      = max(1, args.predict_n) if run_predict else 0

    if not args.json:
        print(f"\n{_c('1', 'AsaMiia — check_model')}  {_c('2', 'v2.1')}")
        print(_c("2", f"Modelo    : {model_path}"))
        print(_c("2", f"miia_model: {models_dir}"))
        if run_predict:
            print(_c("2", f"predict() : {n_runs}x"))

    rep = Report()

    if not model_path.exists():
        rep.error("file", f"Arquivo não encontrado: {model_path}")
        _finish(rep, args.json)
        return

    if model_path.suffix != ".py":
        rep.error("file", f"Extensão esperada .py, encontrada: '{model_path.suffix}'")

    tree = phase_static_analysis(model_path, rep, args.verbose)
    if tree is None:
        _finish(rep, args.json)
        return

    mod, MiiaModel, ModelSchema, TensorSpec = phase_import(model_path, models_dir, rep)
    if mod is None:
        _finish(rep, args.json)
        return

    cls = phase_find_class(mod, MiiaModel, rep)
    if cls is None:
        _finish(rep, args.json)
        return

    phase_interface(cls, rep)

    instance = phase_instantiate(cls, rep)
    if instance is None:
        _finish(rep, args.json)
        return

    loaded, _ = phase_load(instance, rep)

    schema = phase_schema(instance, ModelSchema, TensorSpec, rep, args.verbose)

    if loaded and run_predict:
        phase_predict(instance, schema, rep, n_runs=n_runs, verbose=args.verbose)
    elif not run_predict:
        print(_section("7. predict()"))
        rep.info("Use --predict para executar predict() com inputs sintéticos")

    if loaded:
        phase_validate_inputs(instance, schema, rep, args.verbose)

    if loaded:
        phase_unload(instance, rep)

    phase_schema_after_unload(instance, ModelSchema, rep)

    _finish(rep, args.json)


def _finish(rep: Report, as_json: bool):
    if as_json:
        print(json.dumps(rep.to_dict(), indent=2, ensure_ascii=False))
        sys.exit(0 if not rep.errors else 1)
    else:
        ok = print_summary(rep)
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()