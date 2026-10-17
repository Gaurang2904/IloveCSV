"""Safe, vectorised expressions for cross-column rules.

``check: "ship_date >= order_date"`` is parsed with Python's ``ast`` module and
every node is checked against a whitelist: column names, literals, arithmetic,
comparisons, ``and/or/not`` and a few named functions. Attribute access,
subscripts, lambdas, comprehensions and any other call are rejected at load time.
The tree is then evaluated against pandas Series; ``eval`` and ``DataFrame.eval``
are never called on user text.

Null semantics follow SQL: a comparison that involves a null is *unknown* (pd.NA),
and an unknown check is not counted as a violation. Use ``x is null`` or
``not_null(x)`` to test for missing values explicitly.
"""

from __future__ import annotations

import ast
import io
import re
import tokenize
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

import numpy as np
import pandas as pd

MAX_LEN = 1000
MAX_NODES = 300


class ExprError(Exception):
    pass


_BIN_OPS = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod)
_CMP_OPS = (ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.In, ast.NotIn, ast.Is, ast.IsNot)
_ALLOWED = (
    ast.Expression, ast.BoolOp, ast.And, ast.Or, ast.UnaryOp, ast.Not, ast.USub, ast.UAdd,
    ast.BinOp, ast.Compare, ast.Name, ast.Load, ast.Constant, ast.List, ast.Tuple, ast.Call,
    *_BIN_OPS, *_CMP_OPS,
)  # fmt: skip
# name -> (min args, max args)
FUNCTIONS: dict[str, tuple[int, int]] = {
    "abs": (1, 1),
    "round": (1, 2),
    "len": (1, 1),
    "lower": (1, 1),
    "upper": (1, 1),
    "strip": (1, 1),
    "is_null": (1, 1),
    "not_null": (1, 1),
    "today": (0, 0),
    "date": (1, 1),
    "days_between": (2, 2),
    "coalesce": (2, 2),
}
_BACKTICK = re.compile(r"`([^`]+)`")


@dataclass
class CompiledExpr:
    text: str
    tree: ast.Expression
    columns: list[str]
    _names: dict[str, str]  # identifier in tree -> real column name

    def evaluate(self, resolve: Callable[[str], pd.Series], index: pd.Index) -> pd.Series:
        """Evaluate against typed columns; returns a nullable boolean/number Series."""
        ev = _Evaluator(resolve, index, self._names)
        out = ev.visit(self.tree.body)
        return ev.as_series(out)


def compile_expr(text: str) -> CompiledExpr:
    if not isinstance(text, str) or not text.strip():
        raise ExprError("expression is empty")
    if len(text) > MAX_LEN:
        raise ExprError(f"expression longer than {MAX_LEN} characters")

    names: dict[str, str] = {}

    def _placeholder(m: re.Match) -> str:
        ident = f"__col{len(names)}__"
        names[ident] = m.group(1)
        return ident

    src = _BACKTICK.sub(_placeholder, text)
    src = _null_to_none(src)
    try:
        tree = ast.parse(src.strip(), mode="eval")
    except SyntaxError as e:
        raise ExprError(f"syntax error in '{text}': {e.msg}") from e

    nodes = list(ast.walk(tree))
    if len(nodes) > MAX_NODES:
        raise ExprError("expression is too complex")
    columns: list[str] = []
    func_nodes = {id(n.func) for n in nodes if isinstance(n, ast.Call)}
    for node in nodes:
        if not isinstance(node, _ALLOWED):
            raise ExprError(f"'{type(node).__name__}' is not allowed in rule expressions")
        if isinstance(node, ast.Constant) and not (
            node.value is None or isinstance(node.value, (int, float, str, bool))
        ):
            raise ExprError("only numbers, strings, booleans and null are allowed as literals")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in FUNCTIONS:
                fname = getattr(node.func, "id", type(node.func).__name__)
                raise ExprError(f"function '{fname}' is not allowed")
            if node.keywords:
                raise ExprError("keyword arguments are not allowed")
            lo, hi = FUNCTIONS[node.func.id]
            if not lo <= len(node.args) <= hi:
                raise ExprError(f"{node.func.id}() takes {lo}-{hi} arguments")
        if isinstance(node, ast.Compare):
            for op, right in zip(node.ops, node.comparators, strict=True):
                if isinstance(op, (ast.Is, ast.IsNot)) and not (
                    isinstance(right, ast.Constant) and right.value is None
                ):
                    raise ExprError("'is' / 'is not' can only be used with null")
                if isinstance(op, (ast.In, ast.NotIn)) and not isinstance(right, (ast.List, ast.Tuple)):
                    raise ExprError("'in' needs a literal list, e.g. status in ['A', 'B']")
        if isinstance(node, (ast.List, ast.Tuple)):
            if not all(isinstance(e, ast.Constant) for e in node.elts):
                raise ExprError("lists may only contain literals")
        if isinstance(node, ast.Name) and id(node) not in func_nodes:
            col = names.get(node.id, node.id)
            if node.id.startswith("__") and node.id not in names:
                raise ExprError("names starting with '__' are not allowed")
            if col not in columns:
                columns.append(col)
    return CompiledExpr(text=text, tree=tree, columns=columns, _names=names)


def _null_to_none(src: str) -> str:
    """Let rule authors write `null` (YAML habit) for Python's None."""
    try:
        toks = list(tokenize.generate_tokens(io.StringIO(src).readline))
    except (tokenize.TokenError, IndentationError) as e:
        raise ExprError(f"could not tokenise expression: {e}") from e
    out = []
    for t in toks:
        if t.type == tokenize.NAME and t.string in ("null", "NULL", "Null"):
            t = t._replace(string="None")
        out.append(t)
    return tokenize.untokenize(out)


# ----------------------------------------------------------------------------- evaluation


def _is_series(x: Any) -> bool:
    return isinstance(x, pd.Series)


def _isna(x: Any) -> Any:
    if _is_series(x):
        return x.isna()
    return x is None or (isinstance(x, float) and np.isnan(x)) or x is pd.NaT or x is pd.NA


def _is_datetime(x: Any) -> bool:
    if _is_series(x):
        return pd.api.types.is_datetime64_any_dtype(x)
    return isinstance(x, (pd.Timestamp, datetime, date))


def _to_ts(x: Any) -> Any:
    if isinstance(x, str):
        try:
            return pd.Timestamp(x)
        except (ValueError, TypeError) as e:
            raise ExprError(f"'{x}' is not a date") from e
    if isinstance(x, (datetime, date)) and not isinstance(x, pd.Timestamp):
        return pd.Timestamp(x)
    return x


class _Evaluator:
    def __init__(self, resolve: Callable[[str], pd.Series], index: pd.Index, names: dict[str, str]):
        self.resolve = resolve
        self.index = index
        self.names = names

    def as_series(self, x: Any) -> pd.Series:
        if _is_series(x):
            return x
        return pd.Series([x] * len(self.index), index=self.index, dtype=object)

    def as_bool(self, x: Any) -> pd.Series:
        s = self.as_series(x)
        if s.dtype == "boolean":
            return s
        if pd.api.types.is_bool_dtype(s):
            return s.astype("boolean")
        # Truthiness of non-boolean values: null stays unknown.
        vals = [pd.NA if _isna(v) else bool(v) for v in s.tolist()]
        return pd.Series(vals, index=s.index, dtype="boolean")

    def visit(self, node: ast.AST) -> Any:
        method = getattr(self, f"_{type(node).__name__}")
        return method(node)

    def _Constant(self, node: ast.Constant) -> Any:
        return node.value

    def _List(self, node: ast.List) -> list:
        return [e.value for e in node.elts]  # type: ignore[attr-defined]

    _Tuple = _List

    def _Name(self, node: ast.Name) -> pd.Series:
        col = self.names.get(node.id, node.id)
        try:
            return self.resolve(col)
        except KeyError as e:
            raise ExprError(f"unknown column '{col}'") from e

    def _UnaryOp(self, node: ast.UnaryOp) -> Any:
        v = self.visit(node.operand)
        if isinstance(node.op, ast.Not):
            return ~self.as_bool(v)
        if isinstance(node.op, ast.USub):
            return -v
        return v

    def _BoolOp(self, node: ast.BoolOp) -> pd.Series:
        vals = [self.as_bool(self.visit(v)) for v in node.values]
        out = vals[0]
        for v in vals[1:]:
            out = (out & v) if isinstance(node.op, ast.And) else (out | v)
        return out

    def _BinOp(self, node: ast.BinOp) -> Any:
        a, b = self.visit(node.left), self.visit(node.right)
        if _is_datetime(a) or _is_datetime(b):
            return self._date_arith(node.op, a, b)
        a, b = self._numeric(a), self._numeric(b)
        try:
            if isinstance(node.op, ast.Add):
                return a + b
            if isinstance(node.op, ast.Sub):
                return a - b
            if isinstance(node.op, ast.Mult):
                return a * b
            if isinstance(node.op, ast.Div):
                if _is_series(b):
                    return a / b.replace(0, np.nan)
                return a / b if b not in (0, 0.0) else np.nan
            if isinstance(node.op, ast.Mod):
                return a % b
        except TypeError as e:
            raise ExprError(f"arithmetic on non-numeric values: {e}") from e
        raise ExprError("unsupported operator")  # pragma: no cover

    def _numeric(self, x: Any) -> Any:
        if _is_series(x) and x.dtype == object:
            return pd.to_numeric(x, errors="coerce")
        if x is None:
            return np.nan
        if isinstance(x, str):
            raise ExprError(f"cannot do arithmetic with text '{x}'")
        return x

    def _date_arith(self, op: ast.operator, a: Any, b: Any) -> Any:
        a, b = _to_ts(a), _to_ts(b)
        if isinstance(op, ast.Sub) and _is_datetime(a) and _is_datetime(b):
            delta = a - b
            if _is_series(delta):
                return delta.dt.total_seconds() / 86400.0
            return delta.total_seconds() / 86400.0
        if isinstance(op, (ast.Add, ast.Sub)) and _is_datetime(a) and not _is_datetime(b):
            days = pd.to_timedelta(self._numeric(b), unit="D")
            return a + days if isinstance(op, ast.Add) else a - days
        raise ExprError("only date - date and date +/- days are supported")

    def _Compare(self, node: ast.Compare) -> pd.Series:
        left = self.visit(node.left)
        result: pd.Series | None = None
        for op, right_node in zip(node.ops, node.comparators, strict=True):
            right = self.visit(right_node)
            part = self._compare_pair(op, left, right)
            result = part if result is None else (result & part)
            left = right
        assert result is not None
        return result

    def _compare_pair(self, op: ast.cmpop, a: Any, b: Any) -> pd.Series:
        if isinstance(op, (ast.Is, ast.IsNot)):
            na = self.as_series(a).isna() if _is_series(a) else self.as_series(_isna(a))
            out = na if isinstance(op, ast.Is) else ~na
            return out.astype("boolean")
        if isinstance(op, (ast.In, ast.NotIn)):
            s = self.as_series(a)
            options = list(b)
            if _is_datetime(s):
                options = [_to_ts(o) for o in options]
            hit = s.isin(options).astype("boolean")
            out = hit if isinstance(op, ast.In) else ~hit
            out[s.isna()] = pd.NA
            return out

        if _is_datetime(a) or _is_datetime(b):
            a, b = _to_ts(a), _to_ts(b)
            if not ((_is_datetime(a) or _isna_scalar(a)) and (_is_datetime(b) or _isna_scalar(b))):
                raise ExprError("cannot compare a date with a non-date value")
        sa, sb = self.as_series(a), self.as_series(b)
        if not _is_series(b) and not _isna_scalar(b) and sa.dtype == object:
            # object column vs numeric literal: compare numerically where possible
            if isinstance(b, (int, float)) and not isinstance(b, bool):
                sa = pd.to_numeric(sa, errors="coerce")
        nulls = sa.isna() | sb.isna()
        fn = {
            ast.Eq: lambda x, y: x == y,
            ast.NotEq: lambda x, y: x != y,
            ast.Lt: lambda x, y: x < y,
            ast.LtE: lambda x, y: x <= y,
            ast.Gt: lambda x, y: x > y,
            ast.GtE: lambda x, y: x >= y,
        }[type(op)]
        try:
            with np.errstate(invalid="ignore"):
                raw = fn(sa, sb)
            out = pd.Series(raw, index=self.index).astype("boolean")
        except TypeError:
            vals = []
            for x, y, n in zip(sa.tolist(), sb.tolist(), nulls.tolist(), strict=True):
                if n:
                    vals.append(pd.NA)
                    continue
                try:
                    vals.append(bool(fn(x, y)))
                except TypeError:
                    vals.append(pd.NA)
            out = pd.Series(vals, index=self.index, dtype="boolean")
        out[nulls] = pd.NA
        return out

    def _Call(self, node: ast.Call) -> Any:
        name = node.func.id  # type: ignore[attr-defined]
        args = [self.visit(a) for a in node.args]
        if name == "today":
            return pd.Timestamp.today().normalize()
        if name == "date":
            return _to_ts(args[0])
        if name == "is_null":
            return self._compare_pair(ast.Is(), args[0], None)
        if name == "not_null":
            return self._compare_pair(ast.IsNot(), args[0], None)
        if name == "abs":
            return abs(self._numeric(args[0]))
        if name == "round":
            digits = int(args[1]) if len(args) > 1 else 0
            x = self._numeric(args[0])
            return x.round(digits) if _is_series(x) else round(x, digits)
        if name == "days_between":
            return self._date_arith(ast.Sub(), args[1], args[0])
        if name == "coalesce":
            a = self.as_series(args[0])
            return a.where(a.notna(), args[1] if not _is_series(args[1]) else args[1])
        # string functions
        s = self.as_series(args[0]).astype(object)
        mask = s.notna()
        out = pd.Series([None] * len(s), index=s.index, dtype=object)
        strs = s[mask].astype(str)
        if name == "len":
            out[mask] = strs.str.len()
            return pd.to_numeric(out, errors="coerce")
        if name == "lower":
            out[mask] = strs.str.lower()
        elif name == "upper":
            out[mask] = strs.str.upper()
        elif name == "strip":
            out[mask] = strs.str.strip()
        return out


def _isna_scalar(x: Any) -> bool:
    return not _is_series(x) and bool(_isna(x))
