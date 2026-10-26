"""Read-only Analog-Agent import and a deliberately bounded SPICE parser.

This module never runs a simulator, netlist, Python source, or saved SQL. Only a
DUT subcircuit, MOS instances, independent DC sources, and ideal R/C elements
are accepted. Unsupported DUT syntax is an error, not a missing graph node.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import json
import math
import re
import sqlite3
from contextlib import contextmanager
from itertools import groupby
from pathlib import Path
from typing import Any, Mapping

from .data import TERMINAL_NAMES, validate_dataset


class AdapterError(ValueError):
    """The source cannot be represented without guessing or losing information."""


PARAMETER_UNITS = {
    "W_m": "m", "L_m": "m", "m": "dimensionless", "dc_current_A": "A",
    "dc_voltage_V": "V", "R_ohm": "ohm", "C_F": "F",
}
DEFAULT_MAX_TABLE_ROWS = 1_000_000
_SUFFIXES = {"t": 1e12, "g": 1e9, "meg": 1e6, "k": 1e3, "m": 1e-3,
             "u": 1e-6, "n": 1e-9, "p": 1e-12, "f": 1e-15, "a": 1e-18,
             "mil": 25.4e-6}
_NUMBER = re.compile(r"(?<![\w.])((?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)([a-zA-Z]+)?")
_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_ASSIGN = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*=\s*")
_OP_METADATA = {"sample_id", "device_name", "device_type", "model_name", "primitive_path"}
_KNOWN_MOS_MODELS = {
    "nmos": "NMOS", "pmos": "PMOS",
    **{f"sky130_fd_pr__nfet_{suffix}": "NMOS" for suffix in
       ("01v8", "01v8_lvt", "03v3_nvt", "05v0_nvt", "05v0_nvt__model", "g5v0d10v5")},
    **{f"sky130_fd_pr__pfet_{suffix}": "PMOS" for suffix in
       ("01v8", "01v8_lvt", "01v8_hvt", "g5v0d10v5")},
}


def _number(value: Any, location: str, *, missing: bool = True) -> float | None:
    if value is None and missing:
        return None
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise AdapterError(f"{location}: expected a numeric value" + (" or null" if missing else ""))
    try:
        result = float(value)
    except OverflowError as exc:
        raise AdapterError(f"{location}: numeric overflow") from exc
    if not math.isfinite(result):
        if missing:
            return None
        raise AdapterError(f"{location}: nonfinite value")
    return result


def _json(raw: str | None, location: str, default: Any = None) -> Any:
    if raw is None:
        return copy.deepcopy(default)
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise AdapterError(f"{location}: duplicate JSON key {key!r}")
            result[key] = value
        return result
    try:
        value = json.loads(raw, object_pairs_hook=unique)
    except (ValueError, TypeError, RecursionError) as exc:
        raise AdapterError(f"{location}: invalid JSON: {exc}") from exc
    def finite(item):
        if isinstance(item, float) and not math.isfinite(item):
            return None
        if isinstance(item, list):
            return [finite(v) for v in item]
        if isinstance(item, dict):
            return {k: finite(v) for k, v in item.items()}
        return item
    return finite(value)


def evaluate_spice_expression(expression: str, parameters: Mapping[str, float] | None = None) -> float:
    """Evaluate bounded numeric SPICE arithmetic without eval/exec or functions.

    Names are case-insensitive. Supported suffixes are T/G/MEG/K/M/U/N/P/F/A/MIL;
    M means milli, including upper-case M. Exponent magnitude is limited to 64,
    AST depth to 32, node count to 256, and intermediate magnitude to 1e100.
    """
    if not isinstance(expression, str) or len(expression) > 4096:
        raise AdapterError("SPICE expression must be a string of at most 4096 characters")
    source = expression.strip()
    if source.startswith("{") and source.endswith("}"):
        source = source[1:-1].strip()
    if not source or "{" in source or "}" in source:
        raise AdapterError(f"Unsupported SPICE expression: {expression!r}")
    # Explicit character allowlist also prevents Python numeric extensions.
    if not re.fullmatch(r"[A-Za-z0-9_\s.+*/()\-]+", source):
        raise AdapterError(f"Unsupported SPICE expression: {expression!r}")
    def suffix(match):
        value, unit = match.groups()
        if unit:
            factor = _SUFFIXES.get(unit.lower())
            if factor is None:
                raise AdapterError(f"Unknown SPICE numeric suffix: {unit}")
            return f"({value}*{factor!r})"
        return value
    source = _NUMBER.sub(suffix, source)
    try:
        tree = ast.parse(source, mode="eval")
    except (SyntaxError, RecursionError) as exc:
        raise AdapterError(f"Unsupported SPICE expression: {expression!r}") from exc
    if sum(1 for _ in ast.walk(tree)) > 256:
        raise AdapterError("SPICE expression exceeds node limit")
    environment = {}
    for name, value in (parameters or {}).items():
        if not isinstance(name, str) or not _NAME.fullmatch(name):
            raise AdapterError(f"Invalid parameter identifier: {name!r}")
        lowered = name.lower()
        if lowered in environment:
            raise AdapterError(f"Duplicate case-insensitive parameter: {name}")
        environment[lowered] = value
    def visit(node, depth=0):
        if depth > 32:
            raise AdapterError("SPICE expression exceeds depth limit")
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            result = _number(node.value, "SPICE constant", missing=False)
        elif isinstance(node, ast.Name):
            if node.id.lower() not in environment:
                raise AdapterError(f"Unknown SPICE parameter: {node.id}")
            result = _number(environment[node.id.lower()], f"SPICE parameter {node.id}", missing=False)
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = visit(node.operand, depth + 1)
            result = value if isinstance(node.op, ast.UAdd) else -value
        elif isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)):
            left, right = visit(node.left, depth + 1), visit(node.right, depth + 1)
            try:
                if isinstance(node.op, ast.Add): result = left + right
                elif isinstance(node.op, ast.Sub): result = left - right
                elif isinstance(node.op, ast.Mult): result = left * right
                elif isinstance(node.op, ast.Div): result = left / right
                else:
                    if abs(right) > 64:
                        raise AdapterError("SPICE exponent exceeds magnitude limit")
                    result = left ** right
            except (OverflowError, ZeroDivisionError, ValueError) as exc:
                raise AdapterError(f"Invalid SPICE arithmetic: {exc}") from exc
        else:
            raise AdapterError(f"Unsupported SPICE syntax: {type(node).__name__}")
        if not isinstance(result, (int, float)) or not math.isfinite(result) or abs(result) > 1e100:
            raise AdapterError("SPICE arithmetic is nonfinite, complex, or exceeds magnitude limit")
        return float(result)
    return visit(tree.body)


def _inside(path: Path, root: Path) -> Path:
    resolved = path.resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise AdapterError(f"Source path escapes the input root: {path.name}")
    if not resolved.is_file():
        raise AdapterError(f"Source file is missing: {path.name}")
    return resolved


def _lines(path: Path):
    if path.stat().st_size > 2 * 1024 * 1024:
        raise AdapterError(f"Netlist/parameter file exceeds 2 MiB: {path.name}")
    result = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("*"):
            continue
        line = re.split(r"[;$]", line, maxsplit=1)[0].strip()
        if not line:
            continue
        if line.startswith("+"):
            if not result:
                raise AdapterError(f"{path.name}:{number}: orphan continuation")
            result[-1] = (result[-1][0], result[-1][1] + " " + line[1:].strip())
        else:
            result.append((number, line))
    return result


def _assignments(source: str, location: str) -> dict[str, str]:
    matches = list(_ASSIGN.finditer(source))
    if not matches or source[:matches[0].start()].strip():
        raise AdapterError(f"{location}: expected named parameter assignments")
    output = {}
    for i, match in enumerate(matches):
        key = match.group(1).lower()
        value = source[match.end():matches[i + 1].start() if i + 1 < len(matches) else len(source)].strip()
        if key in output or not value:
            raise AdapterError(f"{location}: duplicate or empty parameter {key}")
        output[key] = value
    return output


def parse_spice_netlist(path: str | Path, *, root: str | Path | None = None,
                        subckt: str = "DUT", parameter_file: str | Path | None = None) -> dict[str, Any]:
    """Parse structure only. Includes are not followed except one local params file.

    The default permitted parameter file is <netlist-stem>_params.sp. It may
    contain .param lines only; recursive includes and executable directives fail.
    All other .include/.lib statements are retained as unexecuted metadata.
    Geometry is multiplied by the numeric .option scale (default 1) to get m.
    """
    path = Path(path)
    if root is None:
        path = path.resolve()
        root = path.parent
    else:
        root = Path(root).resolve()
        path = path if path.is_absolute() else root / path
    path = _inside(path, root)
    permitted = Path(parameter_file) if parameter_file is not None else path.with_name(path.stem + "_params.sp")
    if not permitted.is_absolute():
        permitted = root / permitted
    permitted = permitted.resolve()
    if not permitted.is_relative_to(root):
        raise AdapterError("Parameter file must be inside the input root")
    params, devices, ports, includes, source_files = {}, [], {}, [], [path]
    models = dict(_KNOWN_MOS_MODELS)
    scale, in_dut, found, closed, ended = 1.0, False, False, False, False
    seen_devices, seen_nets = set(), {}
    def net(name):
        return seen_nets.setdefault(name.lower(), name)
    def add_params(content, location):
        for key, value in _assignments(content, location).items():
            if key in params:
                raise AdapterError(f"{location}: duplicate .param definition {key}")
            params[key] = value
    for line_number, line in _lines(path):
        location = f"{path.name}:{line_number}"
        if ended:
            raise AdapterError(f"{location}: content after .end is unsupported")
        head, _, rest = line.partition(" ")
        # partition must also support tabs between the directive and operands.
        pieces = line.split(None, 1)
        head, rest = pieces[0].lower(), pieces[1] if len(pieces) > 1 else ""
        if head in (".include", ".inc", ".lib"):
            if in_dut:
                raise AdapterError(f"{location}: includes inside DUT are unsupported")
            target = rest.strip().strip('"\'')
            local = (path.parent / target).resolve()
            loaded = head != ".lib" and local == permitted
            includes.append({"directive": head, "target": rest, "parameter_file_read": loaded,
                             "executed": False})
            if loaded:
                local = _inside(local, root)
                if local in source_files:
                    raise AdapterError(f"{location}: duplicate parameter include")
                source_files.append(local)
                for number, param_line in _lines(local):
                    part = param_line.split(None, 1)
                    if len(part) != 2 or part[0].lower() != ".param":
                        raise AdapterError(f"{local.name}:{number}: only .param declarations are allowed")
                    add_params(part[1], f"{local.name}:{number}")
            continue
        if head == ".param":
            add_params(rest, location)
            continue
        if head in (".option", ".options"):
            options = _assignments(rest, location)
            if set(options) != {"scale"}:
                raise AdapterError(f"{location}: only .option scale is supported")
            scale = evaluate_spice_expression(options["scale"])
            if scale <= 0:
                raise AdapterError(f"{location}: scale must be positive")
            continue
        if head == ".model" and not in_dut:
            definition = rest.split(None, 2)
            if len(definition) < 2 or definition[1].split("(")[0].upper() not in ("NMOS", "PMOS"):
                raise AdapterError(f"{location}: only NMOS/PMOS model declarations are supported")
            model_name, device_type = definition[0].lower(), definition[1].split("(")[0].upper()
            if model_name in models and models[model_name] != device_type:
                raise AdapterError(f"{location}: conflicting model polarity")
            models[model_name] = device_type
            continue
        if head == ".subckt":
            tokens = rest.split()
            if in_dut or found or not tokens or tokens[0].lower() != subckt.lower():
                raise AdapterError(f"{location}: expected one .subckt {subckt}; nested/other subcircuits unsupported")
            if len(tokens) < 2 or any("=" in token for token in tokens):
                raise AdapterError(f"{location}: DUT ports required; subcircuit parameters unsupported")
            if len({port.lower() for port in tokens[1:]}) != len(tokens[1:]):
                raise AdapterError(f"{location}: duplicate ports")
            ports = {port: net(port) for port in tokens[1:]}
            in_dut, found = True, True
            continue
        if head == ".ends":
            if not in_dut or (rest and rest.lower() != subckt.lower()):
                raise AdapterError(f"{location}: mismatched .ends")
            in_dut, closed = False, True
            continue
        if head == ".end" and not in_dut:
            if rest:
                raise AdapterError(f"{location}: unexpected .end operands")
            if not found or not closed:
                raise AdapterError(f"{location}: .end requires a completed DUT")
            ended = True
            continue
        if head.startswith("."):
            raise AdapterError(f"{location}: unsupported directive {head}")
        if not in_dut:
            raise AdapterError(f"{location}: device outside DUT is unsupported")
        tokens = line.split(None, 6)
        name, kind = tokens[0], tokens[0][0].upper()
        if name.lower() in seen_devices:
            raise AdapterError(f"{location}: duplicate device name {name}")
        seen_devices.add(name.lower())
        model_name = None
        if kind in ("M", "X"):
            if len(tokens) != 7:
                raise AdapterError(f"{location}: MOS requires D G S B model and W/L")
            model_name = tokens[5]
            if model_name.lower() not in models:
                raise AdapterError(f"{location}: unknown MOS model or unsupported subcircuit {model_name}")
            device_type = models[model_name.lower()]
            expressions = _assignments(tokens[6], location)
            if set(expressions) - {"w", "l", "m"} or not {"w", "l"} <= set(expressions):
                raise AdapterError(f"{location}: MOS supports W/L/m only and requires W/L")
            expressions.setdefault("m", "1")
            terminals = dict(zip(("d", "g", "s", "b"), map(net, tokens[1:5])))
        elif kind in ("I", "V", "R", "C"):
            tokens = line.split(None, 3)
            if len(tokens) != 4:
                raise AdapterError(f"{location}: {kind} requires two terminals and a value")
            value = tokens[3]
            if kind in ("I", "V") and re.match(r"(?i)^DC(?:\s|=)", value):
                value = re.sub(r"(?i)^DC\s*=?\s*", "", value)
            feature = {"I": "dc_current_A", "V": "dc_voltage_V", "R": "R_ohm", "C": "C_F"}[kind]
            device_type = {"I": "independent_current_source", "V": "independent_voltage_source",
                           "R": "resistor", "C": "capacitor"}[kind]
            expressions = {feature: value}
            terminals = {"p": net(tokens[1]), "n": net(tokens[2])}
        else:
            raise AdapterError(f"{location}: unsupported DUT device {name}")
        devices.append({"device_id": name, "device_type": device_type, "model_name": model_name,
                        "terminals": terminals, "parameter_expressions": expressions,
                        "netlist_line": line, "netlist_line_number": line_number,
                        "op_target": kind in ("M", "X")})
    if not found or not closed or in_dut or not devices:
        raise AdapterError(f"Netlist requires a nonempty, closed .subckt {subckt}")
    return {"devices": devices, "ports": ports, "parameter_defaults": params,
            "geometry_scale_to_m": scale, "includes": includes,
            "source_files": [str(p) for p in source_files], "subckt": subckt}


def _resolve_parameters(defaults: Mapping[str, str], sample_values: Mapping[str, float]) -> dict[str, float]:
    result = {key.lower(): _number(value, f"design parameter {key}", missing=False)
              for key, value in sample_values.items()}
    pending = {key: value for key, value in defaults.items() if key.lower() not in result}
    while pending:
        progress = False
        for key, expression in list(pending.items()):
            try:
                value = evaluate_spice_expression(expression, result)
            except AdapterError as exc:
                if str(exc).startswith("Unknown SPICE parameter:"):
                    continue
                raise
            result[key.lower()] = value
            del pending[key]
            progress = True
        if not progress:
            raise AdapterError(f"Cyclic or unresolved .param definitions: {sorted(pending)}")
    return result


def _unit(name: str) -> str:
    if name in ("multiplicity", "intrinsic_gain_v_v"):
        return "dimensionless"
    for suffix, unit in (("_v_inv", "1/V"), ("_V_us", "V/us"), ("_uW", "uW"),
                         ("_um", "um"), ("_a", "A"), ("_s", "S"), ("_v", "V"),
                         ("_f", "F"), ("_Hz", "Hz"), ("_dB", "dB"), ("_deg", "deg")):
        if name.endswith(suffix):
            return unit
    return "unspecified_in_source"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _names(value: Any, location: str) -> list[str]:
    if not isinstance(value, list) or not value or any(not isinstance(x, str) or not x for x in value):
        raise AdapterError(f"{location}: expected a nonempty list of names")
    if len({name.lower() for name in value}) != len(value):
        raise AdapterError(f"{location}: duplicate case-insensitive names")
    return value


def _vector(raw: str | None, names: list[str], location: str, *, missing: bool) -> dict[str, float | None]:
    values = _json(raw, location, None)
    if values is None and missing:
        values = [None] * len(names)
    if not isinstance(values, list) or len(values) != len(names):
        raise AdapterError(f"{location}: expected {len(names)} values in dataset_schema order")
    return {key: _number(value, f"{location}.{key}", missing=missing) for key, value in zip(names, values)}


@contextmanager
def _read_database(path: Path, *, max_table_rows: int | None = DEFAULT_MAX_TABLE_ROWS):
    """Preflight counts, then stream fixed SELECTs in one read-only snapshot.

    The safety bound is per table, not per sample. Raising it or passing None
    explicitly permits larger imports. Iterators are valid only inside this
    context. Raw OP/sample tables are never materialized as complete lists;
    callers still decide whether to retain their resulting canonical dataset.
    SQLite may need temporary space for ORDER BY when source indexes are absent.
    """
    if max_table_rows is not None and (type(max_table_rows) is not int or max_table_rows < 1):
        raise AdapterError("max_table_rows must be a positive integer or None")
    connection = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        connection.enable_load_extension(False)
        connection.execute("PRAGMA query_only = ON")
        connection.execute("PRAGMA trusted_schema = OFF")
        connection.execute("BEGIN")
        tables = {row["name"]: row for row in connection.execute(
            "SELECT name, type, sql FROM sqlite_master WHERE name IN "
            "('dataset_schema','samples','dc_device_operating_points','dc_operating_points','sampling_runs')")}
        for required in ("dataset_schema", "samples", "dc_device_operating_points"):
            if required not in tables:
                raise AdapterError(f"Missing source table: {required}")
        for name, row in tables.items():
            if row["type"] != "table" or not re.match(r"\s*CREATE\s+TABLE\b", row["sql"] or "", re.I):
                raise AdapterError(f"Only ordinary SQLite tables are allowed: {name}")
        # No SQL from any source cell or sqlite_master is ever executed.
        info = [dict(row) for row in connection.execute("PRAGMA table_info(dc_device_operating_points)")]
        numeric = [row["name"] for row in info if row["name"] not in _OP_METADATA]
        if not numeric or not _OP_METADATA <= {row["name"] for row in info}:
            raise AdapterError("dc_device_operating_points lacks required identity/numeric columns")
        for column in info:
            if column["name"] in numeric and column["type"].upper() not in ("REAL", "FLOAT", "DOUBLE", "INTEGER", "NUMERIC"):
                raise AdapterError(f"Unexpected nonnumeric OP column: {column['name']}")
        statements = {
            "dataset_schema": "SELECT * FROM dataset_schema ORDER BY id LIMIT 2",
            "samples": "SELECT * FROM samples ORDER BY sample_id",
            "dc_device_operating_points": "SELECT * FROM dc_device_operating_points ORDER BY sample_id,device_name",
            "dc_operating_points": "SELECT * FROM dc_operating_points ORDER BY sample_id",
            "sampling_runs": "SELECT * FROM sampling_runs ORDER BY run_id",
        }
        # Table names come only from this fixed allowlist, never from source SQL.
        counts = {name: connection.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0]
                  if name in tables else 0 for name in statements}
        for name, count in counts.items():
            if max_table_rows is not None and count > max_table_rows:
                raise AdapterError(
                    f"Source table {name} has {count} rows, exceeding max_table_rows={max_table_rows}; "
                    "set an explicit larger limit or None (--max-table-rows none) after checking memory/disk capacity"
                )
        output = {"op_names": numeric, "table_counts": counts}
        for name, statement in statements.items():
            rows = map(dict, connection.execute(statement)) if name in tables else iter(())
            output[name] = list(rows) if name == "dataset_schema" else rows
        yield output
    except sqlite3.Error as exc:
        raise AdapterError(f"Cannot read Analog-Agent SQLite schema: {exc}") from exc
    finally:
        connection.close()


def _observation(row: dict, source_sample: dict) -> dict:
    # State metadata is target-only. Missing reasons do NOT imply device passes.
    original = _json(source_sample.get("observation_json"), "observation_json", {}) or {}
    if not isinstance(original, dict):
        raise AdapterError("observation_json must be an object")
    result = {key: copy.deepcopy(original[key]) for key in
              ("schema_version", "status", "converged", "feasible", "reasons", "missing_fields",
               "bias_topology", "checked_devices", "signal_devices", "thresholds", "constraints",
               "node_voltages_v", "follower_error_v", "vout_v") if key in original}
    def comparable(key, value):
        # SQLite boolean columns use 0/1; JSON uses false/true. These are the
        # same evidence, unlike a missing value or an explicit empty list.
        if key in {"converged", "feasible"} and value is not None:
            if type(value) not in (bool, int) or value not in (0, 1):
                raise AdapterError(f"observation {key}: expected boolean, 0/1, or null")
            return bool(value)
        if key == "reasons" and isinstance(value, list) and all(isinstance(item, str) for item in value):
            return sorted(value)
        return value
    def merge(key, value):
        old = result.get(key)
        normalized_old, normalized_new = comparable(key, old), comparable(key, value)
        if old is not None and value is not None and normalized_old != normalized_new:
            raise AdapterError(
                f"sample {source_sample.get('sample_id')}: conflicting observation {key} "
                "between samples.observation_json and dc_operating_points"
            )
        if value is not None or key not in result:
            result[key] = value
    for key in ("schema_version", "status", "converged", "feasible", "bias_topology", "follower_error_v", "vout_v"):
        if key in row:
            value = row[key]
            merge(key, _number(value, key) if key in ("follower_error_v", "vout_v") else value)
    for column, key in (("reasons_json", "reasons"), ("missing_fields_json", "missing_fields"),
                        ("node_voltages_json", "node_voltages_v")):
        if column in row:
            merge(key, _json(row[column], column))
    result["target_only"] = True
    return result


def adapt_analog_agent(root: str | Path, netlist: str | Path | None = None,
                       strict: bool = True, *,
                       max_table_rows: int | None = DEFAULT_MAX_TABLE_ROWS,
                       progress: bool = False) -> dict[str, Any]:
    """Import every sample, including infeasible and performance-failed samples.

    strict=False relaxes only input-geometry vs measured-geometry consistency;
    unsupported syntax, unknown models, ambiguous identities, and malformed
    vectors always fail. Missing OP rows/features and metrics remain explicit
    null values, never zero-imputed. Source status labels are not model inputs.
    The default per-table safety bound can be raised explicitly or disabled with
    max_table_rows=None. Source rows are streamed, but the returned canonical
    object remains in memory; this is not an out-of-core training data format.
    """
    root = Path(root).resolve()
    if not root.is_dir():
        raise AdapterError(f"Input root is not a directory: {root}")
    database = _inside(root / "sampling_history.sqlite3", root)
    if progress:
        print(f"Import: opening read-only SQLite source {root.name}", flush=True)
    with _read_database(database, max_table_rows=max_table_rows) as source:
        return _adapt_source(root, netlist, strict, database, source, progress=progress)


def _adapt_source(root: Path, netlist: str | Path | None, strict: bool,
                  database: Path, source: dict[str, Any], progress=False) -> dict[str, Any]:
    if len(source["dataset_schema"]) != 1:
        raise AdapterError("Expected exactly one dataset_schema row")
    schema = source["dataset_schema"][0]
    circuit = schema["circuit_name"]
    if not isinstance(circuit, str) or not circuit.strip():
        raise AdapterError("dataset_schema.circuit_name must be nonempty")
    metadata_path = root / "circuit_metadata.json"
    metadata = _json(_inside(metadata_path, root).read_text(encoding="utf-8"), "circuit_metadata.json", {}) if metadata_path.exists() else {}
    if not isinstance(metadata, dict):
        raise AdapterError("circuit_metadata.json must be an object")
    if metadata and metadata.get("circuit_name", circuit) != circuit:
        raise AdapterError("Circuit identity differs between SQLite and circuit_metadata.json")
    netlist_path = Path(netlist) if netlist is not None else Path(circuit + ".sp")
    parsed = parse_spice_netlist(netlist_path, root=root)
    design_names = _names(_json(schema["parameter_names_json"], "parameter_names_json"), "parameter names")
    metric_names = _names(_json(schema["metric_names_json"], "metric_names_json"), "metric names")
    op_names = source["op_names"]
    if not source["table_counts"]["samples"]:
        raise AdapterError("Samples must be nonempty and have unique sample_id values")
    inventory = {device["device_id"].lower(): device for device in parsed["devices"]}
    op_groups = iter(groupby(source["dc_device_operating_points"], key=lambda row: row["sample_id"]))
    observation_groups = iter(groupby(source["dc_operating_points"], key=lambda row: row["sample_id"]))
    pending_ops, pending_observation = next(op_groups, None), next(observation_groups, None)
    sample_ids = set()
    samples, parameter_names, warnings = [], [], []
    for row in source["samples"]:
        sample_id = row["sample_id"]
        if sample_id in sample_ids:
            raise AdapterError("Samples must be nonempty and have unique sample_id values")
        sample_ids.add(sample_id)
        # Keep only this sample's raw OP rows while building its canonical form.
        op_rows = {}
        if pending_ops is not None and pending_ops[0] == sample_id:
            for op_row in pending_ops[1]:
                name = op_row["device_name"]
                if not isinstance(name, str) or name.lower() not in inventory:
                    raise AdapterError(f"OP device is absent from DUT: {name}")
                device = inventory[name.lower()]
                if not device["op_target"]:
                    raise AdapterError(f"Unexpected OP row for unsupported target device: {name}")
                if str(op_row["device_type"]).upper() != device["device_type"] or str(op_row["model_name"]).lower() != device["model_name"].lower():
                    raise AdapterError(f"OP model/type mismatch for {name}")
                if name.lower() in op_rows:
                    raise AdapterError(f"Duplicate OP identity: {(sample_id, name.lower())}")
                op_rows[name.lower()] = op_row
            pending_ops = next(op_groups, None)
        observation = {}
        if pending_observation is not None and pending_observation[0] == sample_id:
            observation = next(pending_observation[1])
            if next(pending_observation[1], None) is not None:
                raise AdapterError("Duplicate or orphan dc_operating_points sample_id")
            pending_observation = next(observation_groups, None)
        design = _vector(row["design_values_json"], design_names, f"sample {sample_id} design", missing=False)
        environment = _resolve_parameters(parsed["parameter_defaults"], design)
        devices = []
        for template in parsed["devices"]:
            device = copy.deepcopy(template)
            expressions = device["parameter_expressions"]
            if device["op_target"]:
                parameters = {"W_m": evaluate_spice_expression(expressions["w"], environment) * parsed["geometry_scale_to_m"],
                              "L_m": evaluate_spice_expression(expressions["l"], environment) * parsed["geometry_scale_to_m"],
                              "m": evaluate_spice_expression(expressions["m"], environment)}
                if any(not math.isfinite(v) or v <= 0 for v in parameters.values()):
                    raise AdapterError(f"sample {sample_id} {device['device_id']}: W/L/m must be finite and positive")
            else:
                parameters = {key: evaluate_spice_expression(value, environment) for key, value in expressions.items()}
                if device["device_type"] in ("resistor", "capacitor") and any(v <= 0 for v in parameters.values()):
                    raise AdapterError(f"sample {sample_id} {device['device_id']}: R/C must be positive")
            for key in parameters:
                if key not in parameter_names:
                    parameter_names.append(key)
            device["parameters"] = parameters
            op = op_rows.get(device["device_id"].lower())
            if not device["op_target"]:
                device["op"], device["op_status"] = {}, "not_supported"
            else:
                device["op"] = {key: _number(op[key], f"sample {sample_id} {device['device_id']}.{key}") if op else None for key in op_names}
                count = sum(value is not None for value in device["op"].values())
                device["op_status"] = "complete" if count == len(op_names) else "partial" if count else "missing"
                device["op_source"] = {"sample_id": sample_id, "device_name": op["device_name"] if op else device["device_id"],
                                       "model_name": op["model_name"] if op else None,
                                       "primitive_path": op.get("primitive_path") if op else None}
                for raw_key, input_key, factor in (("w_um", "W_m", 1e-6), ("l_um", "L_m", 1e-6), ("multiplicity", "m", 1.0)):
                    measured = device["op"].get(raw_key)
                    if measured is not None and not math.isclose(measured * factor, parameters[input_key], rel_tol=1e-7, abs_tol=1e-18):
                        message = f"sample {sample_id} {device['device_id']}: input {input_key} disagrees with source OP {raw_key}"
                        if strict:
                            raise AdapterError(message)
                        warnings.append(message)
            devices.append(device)
        merged_observation = _observation(observation, row)
        observation_json = _json(row.get("observation_json"), "observation_json", {}) or {}
        def condition_context(value):
            if value is None:
                return None
            if not isinstance(value, dict):
                raise AdapterError("OP condition must be an object")
            # Host paths are not operating conditions and are not propagated.
            return {key: item for key, item in value.items() if not key.upper().endswith("PATH")}
        original_context = condition_context(observation_json.get("condition"))
        table_context = condition_context(_json(observation.get("condition_json"), "condition_json"))
        if original_context is not None and table_context is not None and original_context != table_context:
            raise AdapterError(f"sample {sample_id}: conflicting observation condition between samples.observation_json and dc_operating_points")
        context = table_context if table_context is not None else original_context or {}
        bias = merged_observation.get("bias_topology")
        if bias is not None:
            context["bias_topology"] = bias
        design_key = row["design_key"]
        if not isinstance(design_key, str) or not design_key:
            raise AdapterError(f"sample {sample_id}: design_key must be nonempty")
        statuses = [device["op_status"] for device in devices if device["op_target"]]
        samples.append({"sample_id": f"{circuit}:{sample_id}", "source_sample_id": sample_id,
                        "circuit_id": circuit, "design_id": design_key, "design_key": design_key,
                        "raw_design_parameters": design, "devices": devices,
                        "performance": _vector(row.get("metric_values_json"), metric_names, f"sample {sample_id} metrics", missing=True),
                        "ports": copy.deepcopy(parsed["ports"]), "context": context,
                        "op_status": "complete" if statuses and all(s == "complete" for s in statuses) else "partial" if any(s != "missing" for s in statuses) else "failed",
                        "source_observation": merged_observation,
                        "source_sampling": {key: row[key] for key in ("run_id", "sample_index", "sample_method", "success", "error_type", "access_level", "is_real_spice_call", "cache_hit_sample_id") if key in row}})
        if progress and (len(samples) == 1 or len(samples) % 5000 == 0
                         or len(samples) == source['table_counts']['samples']):
            print(f"Import {circuit}: {len(samples)}/{source['table_counts']['samples']} samples", flush=True)
    if pending_ops is not None:
        raise AdapterError(f"Orphan OP sample_id: {pending_ops[0]}")
    if pending_observation is not None:
        raise AdapterError("Duplicate or orphan dc_operating_points sample_id")
    files = [database] + [Path(p) for p in parsed["source_files"]]
    if metadata_path.exists():
        files.append(_inside(metadata_path, root))
    if database.with_name(database.name + "-wal").exists():
        files.append(_inside(database.with_name(database.name + "-wal"), root))
    runs = []
    for row in source["sampling_runs"]:
        config = _json(row.get("config_json"), "config_json", {}) or {}
        # Narrow allowlist: keep useful source provenance, not arbitrary credentials/paths.
        runs.append({"run_id": row["run_id"], "constraints": config.get("sampling", {}).get("constraints", {}),
                     "checked_devices": config.get("checked_devices"), "signal_devices": config.get("signal_devices"),
                     "context_fingerprint": config.get("context_fingerprint"), "implementation": config.get("implementation", {})})
    dataset = {"schema_version": 1,
               "feature_spec": {"parameter_names": parameter_names, "op_names": op_names,
                                "metric_names": metric_names, "terminal_names": list(TERMINAL_NAMES),
                                "parameter_units": {key: PARAMETER_UNITS[key] for key in parameter_names},
                                "op_units": {key: _unit(key) for key in op_names},
                                "metric_units": {key: _unit(key) for key in metric_names}},
               "samples": samples,
               "metadata": {"synthetic": False, "adapter": "analog_agent_sqlite_v1", "circuit_type": schema.get("circuit_type"),
                            "source_observation_role": "target-only metadata; absent reason never implies checked/pass"},
               "provenance": {"source_format": "Analog-Agent", "sqlite_open_mode": "read-only",
                              "join_key": ["sample_id", "device_name (SPICE case-insensitive, exact full name)"],
                              "source_hashes_sha256": {str(p.relative_to(root)): _sha256(p) for p in files},
                              "source_counts": {"samples": len(samples), "design_parameters": len(design_names),
                                                "devices_per_sample": len(parsed["devices"]),
                                                "mos_devices_per_sample": sum(d["op_target"] for d in parsed["devices"]),
                                                "device_op_rows": source["table_counts"]["dc_device_operating_points"], "op_fields": len(op_names), "metrics": len(metric_names)},
                              "raw_design_parameter_order": design_names,
                              "geometry_scale_to_m": parsed["geometry_scale_to_m"],
                              "parameter_defaults": parsed["parameter_defaults"], "includes": parsed["includes"],
                              "runs": runs, "warnings": warnings,
                              "filtering": "none; infeasible samples and missing metrics retained",
                              "op_semantics": "source numeric columns preserved without abs, normalization, or recomputation; some columns are upstream-derived"}}
    if progress:
        print(f"Import {circuit}: validating complete canonical dataset", flush=True)
    return validate_dataset(dataset)


def _combined_source_scope(dataset: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return explicit provenance leaves and samples, without guessing ancestry.

    Scope is added on combination rather than import, so existing single-source
    import fingerprints remain unchanged. A run_id is only local to its source.
    """
    provenance = copy.deepcopy(dataset.get("provenance", {}))
    samples = copy.deepcopy(dataset["samples"])
    leaves: dict[str, dict[str, Any]] = {}

    def add_leaf(leaf: dict[str, Any]) -> None:
        source_id, circuits = leaf.get("source_id"), leaf.get("circuit_ids")
        if not isinstance(source_id, str) or not source_id:
            raise AdapterError("Combined source provenance requires an explicit source_id; legacy unscoped ancestry cannot be guessed")
        if (not isinstance(circuits, list) or not circuits
                or any(not isinstance(item, str) or not item for item in circuits)
                or len(circuits) != len(set(circuits))):
            raise AdapterError(f"Source {source_id}: circuit_ids must be a nonempty unique list")
        if source_id in leaves and leaves[source_id] != leaf:
            raise AdapterError(f"Conflicting provenance for source_id {source_id}")
        leaves[source_id] = leaf

    def flatten(node: dict[str, Any]) -> None:
        if not isinstance(node, dict):
            raise AdapterError("Combined source provenance must be an object")
        if "sources" in node:
            children = node["sources"]
            if not isinstance(children, list) or not children:
                raise AdapterError("Combined provenance.sources must be a nonempty list")
            for child in children:
                flatten(child)
        else:
            add_leaf(node)

    if "sources" in provenance:
        # Do not promote an old unscoped combined dataset into a new single
        # source: doing so would make otherwise unrelated run IDs collide again.
        flatten(provenance)
    else:
        if str(dataset.get("metadata", {}).get("adapter", "")).startswith("analog_agent_combined"):
            raise AdapterError("Combined dataset is missing scoped provenance.sources")
        actual_circuits = sorted({sample["circuit_id"] for sample in samples})
        if "source_id" not in provenance:
            if any("source_id" in sample for sample in samples):
                raise AdapterError("Sample source_id has no matching source provenance")
            scope = {"circuit_ids": actual_circuits, "provenance": provenance}
            digest = hashlib.sha256(json.dumps(scope, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()
            provenance["source_id"] = "analog-agent:" + digest
            provenance["circuit_ids"] = actual_circuits
        add_leaf(provenance)
        for sample in samples:
            sample.setdefault("source_id", provenance["source_id"])

    for sample in samples:
        source_id = sample.get("source_id")
        if not isinstance(source_id, str) or source_id not in leaves:
            raise AdapterError(f"Sample {sample['sample_id']}: source_id has no matching scoped provenance")
        if sample["circuit_id"] not in leaves[source_id]["circuit_ids"]:
            raise AdapterError(f"Sample {sample['sample_id']}: circuit_id is outside its source scope")
    return list(leaves.values()), samples


def combine_datasets(datasets: list[dict[str, Any]]) -> dict[str, Any]:
    """Combine imports with stable feature unions, units, and source scopes.

    Each sample gets a deterministic source_id linked to one provenance leaf,
    whose circuit_ids bound that source. Run IDs are not globally unique. Nested
    combinations preserve/flatten those scopes. Legacy unscoped combined inputs
    are rejected instead of assigning their run metadata to unrelated samples.

    Existing schema rules still require one fixed operating context. This helper
    does not fabricate circuits or reconcile incompatible measurement units.
    """
    if not datasets:
        raise AdapterError("At least one dataset is required")
    combined = {"schema_version": 1, "samples": [], "feature_spec": {"terminal_names": list(TERMINAL_NAMES)},
                "metadata": {"synthetic": False, "adapter": "analog_agent_combined_v1"},
                "provenance": {"source_scope_version": 1, "sources": []}}
    for kind in ("parameter", "op", "metric"):
        names, units = [], {}
        for dataset in datasets:
            validate_dataset(dataset)
            spec = dataset["feature_spec"]
            for name in spec[kind + "_names"]:
                unit = spec.get(kind + "_units", {}).get(name)
                if unit is None:
                    raise AdapterError(f"Missing {kind} units for {name}")
                if name in units and units[name] != unit:
                    raise AdapterError(f"Incompatible {kind} units for {name}: {units[name]} vs {unit}")
                if name not in names:
                    names.append(name)
                units[name] = unit
        combined["feature_spec"][kind + "_names"] = names
        combined["feature_spec"][kind + "_units"] = units
    sources: dict[str, dict[str, Any]] = {}
    for dataset in datasets:
        leaves, samples = _combined_source_scope(dataset)
        combined["samples"].extend(samples)
        for leaf in leaves:
            source_id = leaf["source_id"]
            if source_id in sources and sources[source_id] != leaf:
                raise AdapterError(f"Conflicting provenance for source_id {source_id}")
            sources[source_id] = leaf
    combined["provenance"]["sources"] = list(sources.values())
    return validate_dataset(combined)


def inspect_analog_agent(root: str | Path, netlist: str | Path | None = None, *,
                         max_table_rows: int | None = DEFAULT_MAX_TABLE_ROWS) -> dict[str, Any]:
    """Validated import summary suitable for a CLI inspection command."""
    dataset = adapt_analog_agent(root, netlist, max_table_rows=max_table_rows)
    return {"circuit_ids": sorted({s["circuit_id"] for s in dataset["samples"]}),
            **dataset["provenance"]["source_counts"], "feature_spec": dataset["feature_spec"],
            "missing_metric_values": sum(value is None for s in dataset["samples"] for value in s["performance"].values()),
            "source_hashes_sha256": dataset["provenance"]["source_hashes_sha256"]}
