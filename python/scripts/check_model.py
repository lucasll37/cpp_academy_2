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

def phase_static_analysis(path: Path, rep: Report, verbose: bool):
    print(_section("0. Análise Estática (AST)"))

    src = path.read_text(encoding="utf-8")

    # Parseia AST — erro aqui significa arquivo com syntax error
    try:
        tree = ast.parse(src, filename=str(path))
    except SyntaxError as e:
        rep.error("static", f"SyntaxError: {e}")
        return None  # sem AST, fases seguintes não fazem sentido

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

    # Verifica herança de MiiaModel em nível de AST
    miia_subclasses = []
    for cls in class_defs:
        bases = []
        for b in cls.bases:
            if isinstance(b, ast.Name):
                bases.append(b.id)
            elif isinstance(b, ast.Attribute):
                bases.append(b.attr)
        if "MiiaModel" in bases:
            miia_subclasses.append(cls.name)

    if not miia_subclasses:
        rep.error("static", "Nenhuma classe herda de MiiaModel (verifique: class MinhaClasse(MiiaModel):)")
    else:
        rep.ok(f"Subclasse(s) de MiiaModel no AST: {miia_subclasses}")
        if len(miia_subclasses) > 1:
            rep.warn("static", f"Múltiplas subclasses — backend usa a PRIMEIRA encontrada no dict do módulo (ordem pode variar)")

    # Métodos obrigatórios definidos em cada subclasse
    REQUIRED = {"load", "predict", "get_schema"}
    for cls_name in miia_subclasses:
        cls_node = next(n for n in class_defs if n.name == cls_name)
        defined  = {n.name for n in ast.walk(cls_node) if isinstance(n, ast.FunctionDef)}
        missing  = REQUIRED - defined
        if missing:
            rep.error("static", f"{cls_name}: métodos obrigatórios ausentes: {sorted(missing)}")
        else:
            rep.ok(f"{cls_name}: métodos load/predict/get_schema presentes")

        # __init__ com parâmetros além de self
        for node in ast.walk(cls_node):
            if isinstance(node, ast.FunctionDef) and node.name == "__init__":
                non_self = [a.arg for a in node.args.args if a.arg != "self"]
                req_args = non_self[len(node.args.defaults):]  # sem default = obrigatório
                if req_args:
                    rep.error("static", f"{cls_name}.__init__ tem parâmetros obrigatórios {req_args} — backend instancia sem argumentos")
                else:
                    rep.ok(f"{cls_name}.__init__ sem parâmetros obrigatórios")

    # Heurística: uso de print() — polui stdout do servidor
    print_calls = [
        node.lineno for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(getattr(node, "func", None), ast.Name)
        and node.func.id == "print"
    ]
    if print_calls:
        lines = ", ".join(str(l) for l in print_calls[:8])
        suffix = f" (e mais {len(print_calls)-8})" if len(print_calls) > 8 else ""
        rep.warn("static", f"print() nas linhas {lines}{suffix} — usa asalog em vez de print() para não poluir stdout do servidor")

    # Heurística: sys.exit / os._exit em código de modelo
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

    # Injeta paths
    for p in [str(models_dir), str(path.parent)]:
        if p not in sys.path:
            sys.path.insert(0, p)

    # miia_model
    try:
        import miia_model as _mm
        rep.ok(f"miia_model importado de: {_mm.__file__}")
        MiiaModel  = _mm.MiiaModel
        ModelSchema = _mm.ModelSchema
        TensorSpec = _mm.TensorSpec
    except ImportError as e:
        rep.error("import", f"Falha ao importar miia_model: {e}")
        rep.info("Use --models-dir para apontar o diretório que contém miia_model.py")
        return None, None, None, None

    # Módulo do usuário — captura stdout/stderr para não poluir saída do checker
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod  = importlib.util.module_from_spec(spec)

    with _capture() as (out, err_buf):
        try:
            spec.loader.exec_module(mod)
            rep.ok(f"Módulo '{path.stem}' importado sem exceção")
            captured_out = out.getvalue().strip()
            if captured_out:
                rep.warn("import", f"Módulo emitiu stdout durante import: {captured_out[:200]!r}")
        except Exception:
            rep.error("import", f"Exceção ao importar módulo '{path.stem}':")
            for line in traceback.format_exc().splitlines():
                print(_note(line))
            return None, MiiaModel, ModelSchema, TensorSpec

    return mod, MiiaModel, ModelSchema, TensorSpec


# ─────────────────────────────────────────────────────────────────────────────
# FASE 2 — Descoberta da classe (replica lógica de find_model_class do C++)
# ─────────────────────────────────────────────────────────────────────────────

def phase_find_class(mod, MiiaModel, rep: Report):
    print(_section("2. Descoberta da Classe (replica find_model_class)"))

    # O backend itera module.__dict__ e pega a PRIMEIRA subclasse encontrada
    # (PyDict_Next — ordem de inserção, Python 3.7+)
    candidates = []
    for name, obj in vars(mod).items():
        if not inspect.isclass(obj):
            continue
        if obj is MiiaModel:
            continue
        try:
            if issubclass(obj, MiiaModel):
                candidates.append((name, obj))
        except TypeError:
            pass

    if not candidates:
        rep.error("class", "Nenhuma subclasse de MiiaModel encontrada — backend retornará nullptr em find_model_class()")
        return None

    name, cls = candidates[0]
    rep.ok(f"Classe encontrada: {name}")

    if len(candidates) > 1:
        extra = [n for n, _ in candidates[1:]]
        rep.warn("class", f"Múltiplas subclasses: {[n for n, _ in candidates]} — backend usará '{name}'; ignorará {extra}")

    # Verifica que é do próprio módulo (não importada de outro lugar)
    if cls.__module__ != mod.__name__:
        rep.warn("class", f"{name} é definida em '{cls.__module__}', não em '{mod.__name__}' — pode ser importada de outro módulo")
    else:
        rep.ok(f"{name} definida no próprio módulo (✓)")

    # MRO — verifica que MiiaModel está na cadeia
    mro_names = [c.__name__ for c in cls.__mro__]
    rep.ok(f"MRO: {' → '.join(mro_names)}")
    if "MiiaModel" not in mro_names:
        rep.error("class", "MiiaModel ausente no MRO — herança incorreta")

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

    # Métodos obrigatórios
    for m in REQUIRED_METHODS:
        attr = getattr(cls, m, None)
        if attr is None or not callable(attr):
            rep.error("interface", f"Método obrigatório ausente: {m}()")
        else:
            rep.ok(f"{m}() presente")

    for m in OPTIONAL_METHODS:
        if callable(getattr(cls, m, None)):
            rep.ok(f"{m}() implementado (opcional)")
        else:
            rep.info(f"{m}() não implementado (opcional, usa default da base)")

    # Assinatura de predict
    try:
        sig = inspect.signature(cls.predict)
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
        sig   = inspect.signature(cls.__init__)
        params = {k: v for k, v in sig.parameters.items() if k != "self"}
        req   = [k for k, v in params.items() if v.default is inspect.Parameter.empty
                 and v.kind not in (v.VAR_POSITIONAL, v.VAR_KEYWORD)]
        if req:
            rep.error("interface", f"__init__ tem parâmetros obrigatórios: {req} — backend instancia sem args")
        else:
            rep.ok("__init__ sem parâmetros obrigatórios")
    except Exception as e:
        rep.warn("interface", f"Não foi possível inspecionar __init__: {e}")

    # Verifica abstract methods — se ainda há algum não implementado, instanciação vai falhar
    abstract = getattr(cls, "__abstractmethods__", frozenset())
    if abstract:
        rep.error("interface", f"Métodos abstratos não implementados: {sorted(abstract)} — cls() vai lançar TypeError")
    else:
        rep.ok("Nenhum método abstrato pendente")


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

    # get_schema chamado antes e depois de load() — deve ser idempotente
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

    # ── Metadados ──────────────────────────────────────────────────────
    if schema.description:
        rep.ok(f"description: '{schema.description}'")
    else:
        rep.warn("schema", "description vazia — boa prática preencher")

    if schema.author:
        rep.ok(f"author: '{schema.author}'")
    else:
        rep.warn("schema", "author não preenchido")

    # ── Inputs ────────────────────────────────────────────────────────
    _validate_spec_list("inputs", schema.inputs, TensorSpec, rep, verbose)

    # ── Outputs ───────────────────────────────────────────────────────
    _validate_spec_list("outputs", schema.outputs, TensorSpec, rep, verbose)

    # ── Consistência inputs ↔ outputs ─────────────────────────────────
    input_names  = {ts.name for ts in schema.inputs}
    output_names = {ts.name for ts in schema.outputs}
    overlap = input_names & output_names
    if overlap:
        rep.warn("schema", f"Nomes compartilhados entre inputs e outputs: {overlap} — pode gerar confusão na leitura do schema")

    # Nomes duplicados dentro do mesmo grupo
    for label, specs in [("inputs", schema.inputs), ("outputs", schema.outputs)]:
        names = [ts.name for ts in specs]
        seen  = set()
        for n in names:
            if n in seen:
                rep.error("schema", f"Nome duplicado em {label}: '{n}' — o C++ usa dict, segundo vai sobrescrever o primeiro")
            seen.add(n)

    return schema


def _validate_spec_list(label: str, specs, TensorSpec, rep: Report, verbose: bool):
    if not specs:
        rep.warn("schema", f"schema.{label} está vazio — backend pode ter comportamento inesperado")
        return

    rep.ok(f"schema.{label}: {len(specs)} TensorSpec(s)")

    for i, ts in enumerate(specs):
        prefix = f"{label}[{i}] '{ts.name}'"

        if not isinstance(ts, TensorSpec):
            rep.error("schema", f"{prefix}: não é TensorSpec, é {type(ts).__name__}")
            continue

        # Nome
        if not ts.name or not ts.name.strip():
            rep.error("schema", f"{label}[{i}]: name vazio — backend usa como chave no dict")
        elif not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', ts.name):
            rep.warn("schema", f"{prefix}: nome com caracteres não-identificador — pode causar problemas")
        else:
            rep.ok(f"{prefix}: nome válido")

        # Shape
        if not ts.shape:
            rep.warn("schema", f"{prefix}: shape vazio — considere [-1] para dinâmico")
        else:
            for j, d in enumerate(ts.shape):
                if not isinstance(d, int):
                    rep.error("schema", f"{prefix}: shape[{j}]={d!r} não é int")
                elif d == 0:
                    rep.error("schema", f"{prefix}: shape[{j}]=0 — dimensão zero inválida")
                elif d < -1:
                    rep.warn("schema", f"{prefix}: shape[{j}]={d} — valor negativo diferente de -1 (dinâmico)")
            rep.ok(f"{prefix}: shape={ts.shape}")

        # dtype
        if ts.dtype not in VALID_DTYPES:
            rep.warn("schema", f"{prefix}: dtype='{ts.dtype}' não é dtype numpy padrão (esperado: {sorted(VALID_DTYPES)})")
        else:
            rep.ok(f"{prefix}: dtype={ts.dtype}")

        # structured
        if ts.structured:
            rep.ok(f"{prefix}: structured=True (input dict/list aninhado)")
            if label == "outputs":
                rep.warn("schema", f"{prefix}: output com structured=True — py_dict_to_outputs espera np.ndarray, não dict")
            if ts.shape and ts.shape != [-1]:
                rep.warn("schema", f"{prefix}: structured=True mas shape={ts.shape} — use [-1] para não induzir validação de shape")
        else:
            if label == "inputs":
                # Heurística: shape vazia ou [-1] para input não-structured é suspeito
                if ts.shape == [-1] or not ts.shape:
                    rep.warn("schema", f"{prefix}: structured=False mas shape={ts.shape} — se o input for dict/list, declare structured=True")

        # min/max_value
        if ts.min_value is not None and ts.max_value is not None:
            if ts.min_value >= ts.max_value:
                rep.error("schema", f"{prefix}: min_value={ts.min_value} >= max_value={ts.max_value}")
            else:
                rep.ok(f"{prefix}: range=[{ts.min_value}, {ts.max_value}]")

        # ── Aviso de colapso para escalar (comportamento C++) ────────
        if label == "outputs" and not ts.structured:
            total = 1
            for d in ts.shape:
                if d > 0:
                    total *= d
            if total == 1:
                rep.warn("schema", (
                    f"{prefix}: shape={ts.shape} → total=1 → "
                    "py_dict_to_outputs COLAPSA para Value escalar (is_number()). "
                    "Se o caller espera is_array(), use shape com total>1"
                ))


# ─────────────────────────────────────────────────────────────────────────────
# FASE 7 — predict() — validação exaustiva de entrada e saída
# ─────────────────────────────────────────────────────────────────────────────

def phase_predict(instance, schema, rep: Report, n_runs: int, verbose: bool):
    print(_section("7. predict()"))

    if schema is None:
        rep.warn("predict", "Schema inválido — predict() não será executado")
        return

    import numpy as np  # já sabemos que importa; se não, o erro foi em fase anterior

    # ── Constrói inputs ───────────────────────────────────────────────
    inputs = _build_inputs(schema, rep)
    if verbose:
        rep.info(f"Inputs sintéticos: {_summarize_value(inputs)}")

    # ── Executa N vezes ───────────────────────────────────────────────
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

    # ── Verifica tipo de retorno ──────────────────────────────────────
    if not isinstance(result, dict):
        rep.error("predict", f"predict() retornou {type(result).__name__} em vez de dict — py_dict_to_outputs falhará com 'predict() did not return a dict'")
        return
    rep.ok("predict() retornou dict")

    # ── Chaves do resultado ───────────────────────────────────────────
    declared_out = {ts.name for ts in schema.outputs}
    returned     = set(result.keys())

    missing = declared_out - returned
    extra   = returned   - declared_out

    if missing:
        rep.error("predict", f"Outputs declarados no schema mas AUSENTES no resultado: {sorted(missing)}")
    if extra:
        rep.warn("predict", f"Outputs presentes no resultado mas NÃO declarados no schema: {sorted(extra)}")

    # ── Valida cada output ────────────────────────────────────────────
    for key, val in result.items():
        _validate_output_value(key, val, schema, rep, verbose)

    # ── Determinismo (só se n_runs > 1) ─────────────────────────────
    if n_runs > 1:
        _check_determinism(results, schema, rep)

    # ── Verifica keys não-string ──────────────────────────────────────
    for k in result:
        if not isinstance(k, str):
            rep.error("predict", f"Chave de output não é string: {k!r} (tipo {type(k).__name__}) — py_dict_to_outputs vai falhar")


def _validate_output_value(key: str, val: Any, schema, rep: Report, verbose: bool):
    import numpy as np

    prefix = f"output '{key}'"

    # Tenta converter para np.ndarray float32 (exatamente como o C++ faz)
    try:
        arr = np.asarray(val, dtype=np.float32)
    except Exception as e:
        rep.error("predict", f"{prefix}: PyArray_FROMANY falharia — não conversível para float32 array: {e}")
        return

    rep.ok(f"{prefix}: conversível para float32 array")

    # Shape
    shape_str = str(list(arr.shape))
    total     = arr.size
    ndim      = arr.ndim
    rep.ok(f"{prefix}: shape={shape_str}  ndim={ndim}  total={total}")

    # Colapso escalar (comportamento C++ exato)
    if total == 0:
        rep.error("predict", f"{prefix}: array VAZIO (total=0) — py_dict_to_outputs produz Value{{Array{{}}}} vazio, caller receberá array vazio")
    elif total == 1:
        v = float(arr.flat[0])
        rep.warn("predict", (
            f"{prefix}: total=1 → C++ COLAPSA para Value escalar (is_number()={v:.6g}). "
            "Caller deve usar result.outputs.at('{key}').as_number(), NÃO .as_array()"
        ))
        if math.isnan(v):
            rep.error("predict", f"{prefix}: valor é NaN — indica erro de cálculo")
        elif math.isinf(v):
            rep.error("predict", f"{prefix}: valor é Inf — indica divisão por zero ou overflow")
    else:
        rep.ok(f"{prefix}: total={total} → C++ produz Value{{Array}} com {total} elementos float64")

    # NaN/Inf em qualquer elemento
    if total > 0:
        if np.any(np.isnan(arr)):
            rep.error("predict", f"{prefix}: array contém NaN")
        if np.any(np.isinf(arr)):
            rep.error("predict", f"{prefix}: array contém Inf")

    # Verifica shape contra schema declarado
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

    # Tipo original retornado pelo modelo
    if verbose:
        rep.info(f"{prefix}: tipo original Python = {type(val).__name__}")


def _check_determinism(results: list, schema, rep: Report):
    import numpy as np
    rep.ok("Verificando determinismo entre runs...")
    for ts in schema.outputs:
        key = ts.name
        vals = [results[i].get(key) for i in range(len(results))]
        if any(v is None for v in vals):
            continue
        try:
            arrs = [np.asarray(v, dtype=np.float32) for v in vals]
            ref  = arrs[0]
            for i, a in enumerate(arrs[1:], start=1):
                if ref.shape != a.shape:
                    rep.warn("predict", f"output '{key}': shape divergiu entre runs (run 0={list(ref.shape)}, run {i}={list(a.shape)})")
                elif not np.allclose(ref, a, atol=1e-5, rtol=1e-5, equal_nan=True):
                    diff = np.max(np.abs(a - ref))
                    rep.warn("predict", f"output '{key}': resultado divergiu entre runs (max_diff={diff:.2e}) — modelo não-determinístico?")
            else:
                rep.ok(f"output '{key}': determinístico entre {len(results)} runs")
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# FASE 8 — validate_inputs() (se implementado)
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
            rep.warn("validate_inputs", f"validate_inputs retornou valid=False para inputs sintéticos: {vr.errors}")
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

    # Segunda chamada de unload deve ser segura (idempotência)
    try:
        with _capture():
            instance.unload()
        rep.ok("Segunda chamada a unload() segura (idempotente)")
    except Exception:
        rep.warn("unload", "Segunda chamada a unload() lançou exceção — unload() não é idempotente")


# ─────────────────────────────────────────────────────────────────────────────
# FASE 10 — get_schema() após unload (idempotência)
# ─────────────────────────────────────────────────────────────────────────────

def phase_schema_after_unload(instance, ModelSchema, rep: Report):
    print(_section("10. get_schema() após unload()"))
    try:
        with _capture():
            schema2 = instance.get_schema()
        if isinstance(schema2, ModelSchema):
            rep.ok("get_schema() acessível após unload() (conforme docstring)")
        else:
            rep.warn("schema_after_unload", f"get_schema() após unload retornou {type(schema2).__name__}")
    except Exception:
        rep.warn("schema_after_unload", (
            "get_schema() lançou exceção após unload() — "
            "docstring diz que deve ser chamável antes e depois de load()"
        ))


# ─────────────────────────────────────────────────────────────────────────────
# Helpers de construção de inputs sintéticos
# ─────────────────────────────────────────────────────────────────────────────

def _build_inputs(schema, rep: Report) -> dict:
    """Constrói inputs sintéticos compatíveis com o schema."""
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
                # Aplica min/max se declarados
                if ts.min_value is not None:
                    arr = np.clip(arr, ts.min_value, None)
                if ts.max_value is not None:
                    arr = np.clip(arr, None, ts.max_value)
                inputs[ts.name] = arr.tolist()
            except Exception as e:
                inputs[ts.name] = [1.0]
                rep.warn("predict", f"Não foi possível construir input '{ts.name}' com shape={shape} dtype={ts.dtype}: {e} — usando [1.0]")
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
                        help="Saída em JSON para CI (suprime saída formatada)")
    args = parser.parse_args()

    if args.no_color or args.json:
        USE_COLOR = False

    model_path = Path(args.model).resolve()
    models_dir = Path(args.models_dir).resolve() if args.models_dir else model_path.parent

    run_predict = args.predict or args.predict_n > 1
    n_runs      = max(1, args.predict_n) if run_predict else 0

    if not args.json:
        print(f"\n{_c('1', 'AsaMiia — check_model')}  {_c('2', 'v2.0')}")
        print(_c("2", f"Modelo    : {model_path}"))
        print(_c("2", f"miia_model: {models_dir}"))
        if run_predict:
            print(_c("2", f"predict() : {n_runs}x"))

    rep = Report()

    # ── Verificações básicas de arquivo ──────────────────────────────
    if not model_path.exists():
        rep.error("file", f"Arquivo não encontrado: {model_path}")
        _finish(rep, args.json)
        return

    if model_path.suffix != ".py":
        rep.error("file", f"Extensão esperada .py, encontrada: '{model_path.suffix}'")

    # ── Fase 0: AST ───────────────────────────────────────────────────
    tree = phase_static_analysis(model_path, rep, args.verbose)
    if tree is None:
        _finish(rep, args.json)
        return

    # ── Fase 1: Importação ────────────────────────────────────────────
    mod, MiiaModel, ModelSchema, TensorSpec = phase_import(model_path, models_dir, rep)
    if mod is None:
        _finish(rep, args.json)
        return

    # ── Fase 2: Descoberta da classe ──────────────────────────────────
    cls = phase_find_class(mod, MiiaModel, rep)
    if cls is None:
        _finish(rep, args.json)
        return

    # ── Fase 3: Interface ─────────────────────────────────────────────
    phase_interface(cls, rep)

    # ── Fase 4: Instanciação ──────────────────────────────────────────
    instance = phase_instantiate(cls, rep)
    if instance is None:
        _finish(rep, args.json)
        return

    # ── Fase 5: load() ────────────────────────────────────────────────
    loaded, _ = phase_load(instance, rep)

    # ── Fase 6: get_schema() ──────────────────────────────────────────
    schema = phase_schema(instance, ModelSchema, TensorSpec, rep, args.verbose)

    # ── Fase 7: predict() ─────────────────────────────────────────────
    if loaded and run_predict:
        phase_predict(instance, schema, rep, n_runs=n_runs, verbose=args.verbose)
    elif not run_predict:
        print(_section("7. predict()"))
        rep.info("Use --predict para executar predict() com inputs sintéticos")

    # ── Fase 8: validate_inputs() ─────────────────────────────────────
    if loaded:
        phase_validate_inputs(instance, schema, rep, args.verbose)

    # ── Fase 9: unload() ──────────────────────────────────────────────
    if loaded:
        phase_unload(instance, rep)

    # ── Fase 10: schema após unload ───────────────────────────────────
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