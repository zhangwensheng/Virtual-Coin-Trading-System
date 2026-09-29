from __future__ import annotations

import json
import random
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from typing import Any

import pandas as pd
import requests

from futures_strategy.config import AppConfig

DEFAULT_OPENAI_MODEL = "gpt-5.4"
DEFAULT_MAJORS_SYMBOLS = [
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "XRPUSDT",
]


@dataclass(frozen=True)
class ParameterSpec:
    kind: str
    minimum: float
    maximum: float
    step: float


DEFAULT_MAJORS_SEARCH_SPACE = {
    "htf_adx_threshold": ParameterSpec(kind="float", minimum=12.0, maximum=26.0, step=1.0),
    "atr_stop_mult": ParameterSpec(kind="float", minimum=0.8, maximum=1.8, step=0.1),
    "trail_atr_mult": ParameterSpec(kind="float", minimum=1.0, maximum=2.8, step=0.1),
    "partial_rr": ParameterSpec(kind="float", minimum=0.8, maximum=1.8, step=0.1),
    "majors_impulse_threshold": ParameterSpec(kind="float", minimum=0.004, maximum=0.018, step=0.001),
    "majors_pullback_atr_tolerance": ParameterSpec(kind="float", minimum=0.2, maximum=0.7, step=0.05),
    "majors_min_retrace_from_extreme": ParameterSpec(kind="float", minimum=0.001, maximum=0.005, step=0.001),
    "majors_max_retrace_from_extreme": ParameterSpec(kind="float", minimum=0.008, maximum=0.025, step=0.001),
    "majors_vwap_extension_cap": ParameterSpec(kind="float", minimum=0.006, maximum=0.02, step=0.001),
    "majors_volume_ratio_threshold": ParameterSpec(kind="float", minimum=0.8, maximum=1.4, step=0.05),
    "majors_entry_rsi_low": ParameterSpec(kind="float", minimum=44.0, maximum=54.0, step=1.0),
    "majors_entry_rsi_high": ParameterSpec(kind="float", minimum=56.0, maximum=72.0, step=1.0),
    "majors_exit_rsi": ParameterSpec(kind="float", minimum=40.0, maximum=52.0, step=1.0),
}


def strategy_param_snapshot(config: AppConfig, search_space: dict[str, ParameterSpec]) -> dict[str, float]:
    return {key: float(getattr(config.strategy, key)) for key in search_space}


def decimal_places(step: float) -> int:
    step_decimal = Decimal(str(step)).normalize()
    exponent = step_decimal.as_tuple().exponent
    return max(0, -exponent)


def quantize_value(value: float, step: float) -> float:
    if step <= 0:
        return value
    step_decimal = Decimal(str(step))
    value_decimal = Decimal(str(value))
    quantized = (value_decimal / step_decimal).quantize(Decimal("1"), rounding=ROUND_HALF_UP) * step_decimal
    return float(quantized)


def clamp_and_coerce_value(raw_value: Any, spec: ParameterSpec, fallback: float) -> float:
    try:
        numeric = float(raw_value)
    except (TypeError, ValueError):
        numeric = fallback
    numeric = min(spec.maximum, max(spec.minimum, numeric))
    numeric = quantize_value(numeric, spec.step)
    numeric = min(spec.maximum, max(spec.minimum, numeric))
    if spec.kind == "int":
        return int(round(numeric))
    return round(numeric, decimal_places(spec.step))


def coerce_candidate_params(
    raw_params: dict[str, Any] | None,
    base_params: dict[str, float],
    search_space: dict[str, ParameterSpec],
) -> dict[str, float]:
    raw_params = raw_params or {}
    params = {
        key: clamp_and_coerce_value(raw_params.get(key, base_params[key]), spec, base_params[key])
        for key, spec in search_space.items()
    }

    if params["trail_atr_mult"] < params["atr_stop_mult"]:
        params["trail_atr_mult"] = round(params["atr_stop_mult"], decimal_places(search_space["trail_atr_mult"].step))
    if params["majors_entry_rsi_high"] <= params["majors_entry_rsi_low"]:
        params["majors_entry_rsi_high"] = min(
            search_space["majors_entry_rsi_high"].maximum,
            params["majors_entry_rsi_low"] + search_space["majors_entry_rsi_high"].step,
        )
        params["majors_entry_rsi_high"] = round(
            params["majors_entry_rsi_high"],
            decimal_places(search_space["majors_entry_rsi_high"].step),
        )
    if params["majors_max_retrace_from_extreme"] <= params["majors_min_retrace_from_extreme"]:
        params["majors_max_retrace_from_extreme"] = min(
            search_space["majors_max_retrace_from_extreme"].maximum,
            params["majors_min_retrace_from_extreme"] + search_space["majors_max_retrace_from_extreme"].step,
        )
        params["majors_max_retrace_from_extreme"] = round(
            params["majors_max_retrace_from_extreme"],
            decimal_places(search_space["majors_max_retrace_from_extreme"].step),
        )
    return params


def candidate_signature(params: dict[str, float]) -> str:
    return json.dumps(params, sort_keys=True, ensure_ascii=False)


def space_for_prompt(search_space: dict[str, ParameterSpec]) -> dict[str, dict[str, float | str]]:
    return {
        key: {
            "type": spec.kind,
            "min": spec.minimum,
            "max": spec.maximum,
            "step": spec.step,
        }
        for key, spec in search_space.items()
    }


def best_split_metrics(aggregate_results: pd.DataFrame, combo_id: int) -> dict[str, dict[str, float]]:
    subset = aggregate_results[aggregate_results["combo_id"] == combo_id]
    metrics: dict[str, dict[str, float]] = {}
    for split_name in subset["split"].drop_duplicates():
        row = subset[subset["split"] == split_name].iloc[0]
        metrics[split_name] = {
            "robustness_score": round(float(row["robustness_score"]), 4),
            "mean_return_pct": round(float(row["mean_return_pct"]), 4),
            "median_return_pct": round(float(row["median_return_pct"]), 4),
            "min_return_pct": round(float(row["min_return_pct"]), 4),
            "avg_max_drawdown_pct": round(float(row["avg_max_drawdown_pct"]), 4),
            "avg_profit_factor": round(float(row["avg_profit_factor"]), 4),
            "profitable_ratio": round(float(row["profitable_ratio"]), 4),
            "mean_total_trades": round(float(row["mean_total_trades"]), 4),
        }
    return metrics


def per_symbol_snapshot(symbol_results: pd.DataFrame, combo_id: int, split_name: str) -> list[dict[str, float | str]]:
    subset = symbol_results[(symbol_results["combo_id"] == combo_id) & (symbol_results["split"] == split_name)]
    if subset.empty:
        return []
    ordered = subset.sort_values("return_pct", ascending=False)
    rows: list[dict[str, float | str]] = []
    for _, row in ordered.iterrows():
        rows.append(
            {
                "symbol": str(row["symbol"]),
                "return_pct": round(float(row["return_pct"]), 4),
                "max_drawdown_pct": round(float(row["max_drawdown_pct"]), 4),
                "profit_factor": round(float(row["profit_factor"]), 4) if row["profit_factor"] != "inf" else "inf",
                "total_trades": int(row["total_trades"]),
                "win_rate_pct": round(float(row["win_rate_pct"]), 4),
            }
        )
    return rows


def summarize_round_history(
    round_index: int,
    selection_table: pd.DataFrame,
    aggregate_results: pd.DataFrame,
    symbol_results: pd.DataFrame,
    limit: int = 3,
) -> dict[str, Any]:
    summary = {
        "round": round_index,
        "top_candidates": [],
    }
    if selection_table.empty:
        return summary

    for _, row in selection_table.head(limit).iterrows():
        combo_id = int(row["combo_id"])
        valid_symbols = per_symbol_snapshot(symbol_results, combo_id, "valid")
        if not valid_symbols:
            valid_symbols = per_symbol_snapshot(symbol_results, combo_id, "train")
        summary["top_candidates"].append(
            {
                "combo_id": combo_id,
                "selection_score": round(float(row["selection_score"]), 4),
                "params": {
                    column.replace("param_", ""): row[column]
                    for column in selection_table.columns
                    if column.startswith("param_")
                },
                "split_metrics": best_split_metrics(aggregate_results, combo_id),
                "per_symbol": valid_symbols,
            }
        )
    return summary


def build_structured_output_schema(search_space: dict[str, ParameterSpec], candidates_per_round: int) -> dict[str, Any]:
    param_properties = {
        key: {
            "type": "number" if spec.kind == "float" else "integer",
            "minimum": spec.minimum,
            "maximum": spec.maximum,
        }
        for key, spec in search_space.items()
    }
    return {
        "type": "object",
        "properties": {
            "model_thesis": {"type": "string"},
            "candidates": {
                "type": "array",
                "minItems": candidates_per_round,
                "maxItems": candidates_per_round,
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "rationale": {"type": "string"},
                        "params": {
                            "type": "object",
                            "properties": param_properties,
                            "required": list(search_space.keys()),
                            "additionalProperties": False,
                        },
                    },
                    "required": ["name", "rationale", "params"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["model_thesis", "candidates"],
        "additionalProperties": False,
    }


def build_openai_messages(
    base_params: dict[str, float],
    split_months: dict[str, list[str]],
    symbols: list[str],
    history: list[dict[str, Any]],
    search_space: dict[str, ParameterSpec],
    candidates_per_round: int,
) -> list[dict[str, str]]:
    system_prompt = (
        "You are a quantitative trading research assistant. "
        "Propose the next parameter candidates for a multi-symbol futures backtest. "
        "Prioritize validation robustness over raw return, avoid overfitting, and keep drawdowns controlled."
    )
    user_payload = {
        "objective": {
            "priority_order": [
                "maximize validation robustness_score",
                "keep profitable_ratio broad across symbols",
                "limit avg_max_drawdown_pct and worst single-symbol losses",
                "avoid parameter collapse into a single-symbol solution",
            ],
            "strategy_context": "Major-coin 5m low-timeframe trend continuation with 1h trend filter.",
            "candidates_per_round": candidates_per_round,
        },
        "splits": split_months,
        "symbols": symbols,
        "base_params": base_params,
        "search_space": space_for_prompt(search_space),
        "history": history[-3:],
        "requirements": [
            "Return exactly the requested number of candidates.",
            "Stay inside the search space bounds and respect the step size conceptually.",
            "Diversify the proposals: at least one conservative variant and one activity-restoring variant.",
            "Do not simply repeat the current best candidate unless you materially adjust it.",
        ],
    }
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False, indent=2)},
    ]


def extract_response_text(payload: dict[str, Any]) -> str:
    if isinstance(payload.get("output_text"), str) and payload["output_text"].strip():
        return payload["output_text"]

    for item in payload.get("output", []):
        if not isinstance(item, dict):
            continue
        for content in item.get("content", []):
            if not isinstance(content, dict):
                continue
            if isinstance(content.get("text"), str) and content["text"].strip():
                return content["text"]
    raise ValueError("OpenAI response did not contain output text.")


def request_openai_candidates(
    api_key: str,
    model: str,
    base_params: dict[str, float],
    split_months: dict[str, list[str]],
    symbols: list[str],
    history: list[dict[str, Any]],
    search_space: dict[str, ParameterSpec],
    candidates_per_round: int,
    timeout_seconds: int = 90,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, str]]]:
    messages = build_openai_messages(
        base_params=base_params,
        split_months=split_months,
        symbols=symbols,
        history=history,
        search_space=search_space,
        candidates_per_round=candidates_per_round,
    )
    response = requests.post(
        "https://api.openai.com/v1/responses",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "store": False,
            "input": messages,
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "majors_parameter_candidates",
                    "schema": build_structured_output_schema(search_space, candidates_per_round),
                    "strict": True,
                }
            },
        },
        timeout=timeout_seconds,
    )
    response.raise_for_status()
    payload = response.json()
    parsed = json.loads(extract_response_text(payload))
    candidates = parsed.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("OpenAI response did not contain candidates.")
    return candidates, payload, messages


def mutate_params(
    reference: dict[str, float],
    search_space: dict[str, ParameterSpec],
    rng: random.Random,
    strong: bool,
) -> dict[str, float]:
    mutated = dict(reference)
    change_count = min(len(search_space), 5 if strong else 4)
    keys = rng.sample(list(search_space.keys()), k=change_count)
    for key in keys:
        spec = search_space[key]
        step_units = rng.randint(1, 3 if strong else 2)
        direction = rng.choice([-1, 1])
        delta = spec.step * step_units * direction
        mutated[key] = reference[key] + delta
    return coerce_candidate_params(mutated, reference, search_space)


def handcrafted_seed_candidates(base_params: dict[str, float]) -> list[dict[str, Any]]:
    return [
        {
            "name": "baseline_anchor",
            "rationale": "Keep the current baseline as an anchor for later comparison.",
            "params": dict(base_params),
        },
        {
            "name": "conservative_filter",
            "rationale": "Tighten trend and liquidity filters to reduce false breakouts and drawdown.",
            "params": {
                **base_params,
                "htf_adx_threshold": base_params["htf_adx_threshold"] + 3,
                "majors_impulse_threshold": base_params["majors_impulse_threshold"] + 0.002,
                "majors_vwap_extension_cap": base_params["majors_vwap_extension_cap"] - 0.002,
                "majors_volume_ratio_threshold": base_params["majors_volume_ratio_threshold"] + 0.1,
                "partial_rr": base_params["partial_rr"] + 0.1,
            },
        },
        {
            "name": "activity_restore",
            "rationale": "Loosen the trigger slightly to restore trade count if the baseline is too selective.",
            "params": {
                **base_params,
                "htf_adx_threshold": base_params["htf_adx_threshold"] - 2,
                "majors_impulse_threshold": base_params["majors_impulse_threshold"] - 0.002,
                "majors_pullback_atr_tolerance": base_params["majors_pullback_atr_tolerance"] + 0.1,
                "majors_volume_ratio_threshold": base_params["majors_volume_ratio_threshold"] - 0.05,
                "majors_vwap_extension_cap": base_params["majors_vwap_extension_cap"] + 0.002,
            },
        },
        {
            "name": "quicker_protection",
            "rationale": "Take partials sooner and tighten exits so losing rotations do less damage.",
            "params": {
                **base_params,
                "atr_stop_mult": base_params["atr_stop_mult"] - 0.1,
                "trail_atr_mult": base_params["trail_atr_mult"] - 0.2,
                "partial_rr": base_params["partial_rr"] - 0.2,
                "majors_exit_rsi": base_params["majors_exit_rsi"] + 2,
            },
        },
    ]


def select_history_references(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    references: list[dict[str, Any]] = []
    for round_summary in reversed(history):
        references.extend(round_summary.get("top_candidates", []))
    references.sort(key=lambda item: item.get("selection_score", -999.0), reverse=True)
    return references[:3]


def offline_candidate_proposals(
    base_params: dict[str, float],
    history: list[dict[str, Any]],
    search_space: dict[str, ParameterSpec],
    candidates_per_round: int,
    round_index: int,
    seed: int,
    seen_signatures: set[str] | None = None,
) -> list[dict[str, Any]]:
    rng = random.Random((seed * 10_003) + round_index)
    seen_signatures = set(seen_signatures or set())
    candidates: list[dict[str, Any]] = []

    def append_candidate(name: str, rationale: str, params: dict[str, Any]) -> None:
        coerced = coerce_candidate_params(params, base_params, search_space)
        signature = candidate_signature(coerced)
        if signature in seen_signatures:
            return
        seen_signatures.add(signature)
        candidates.append(
            {
                "name": name,
                "rationale": rationale,
                "params": coerced,
            }
        )

    if not history:
        for proposal in handcrafted_seed_candidates(base_params):
            append_candidate(proposal["name"], proposal["rationale"], proposal["params"])
    else:
        references = select_history_references(history)
        if references:
            best = references[0]["params"]
            append_candidate(
                "exploit_best",
                "Stay near the best validation candidate while nudging the most sensitive thresholds.",
                mutate_params(best, search_space, rng, strong=False),
            )
            append_candidate(
                "reduce_drawdown",
                "Tighten filters around the current leader to see if drawdown can be lowered across lagging symbols.",
                {
                    **best,
                    "htf_adx_threshold": best["htf_adx_threshold"] + 2,
                    "majors_impulse_threshold": best["majors_impulse_threshold"] + 0.001,
                    "majors_vwap_extension_cap": best["majors_vwap_extension_cap"] - 0.001,
                    "majors_volume_ratio_threshold": best["majors_volume_ratio_threshold"] + 0.05,
                },
            )
            append_candidate(
                "restore_trade_count",
                "Loosen the top candidate slightly to see if broad symbol participation improves.",
                {
                    **best,
                    "htf_adx_threshold": best["htf_adx_threshold"] - 1,
                    "majors_impulse_threshold": best["majors_impulse_threshold"] - 0.001,
                    "majors_pullback_atr_tolerance": best["majors_pullback_atr_tolerance"] + 0.05,
                    "majors_volume_ratio_threshold": best["majors_volume_ratio_threshold"] - 0.05,
                },
            )
            for index, reference in enumerate(references[1:], start=2):
                append_candidate(
                    f"exploit_ref_{index}",
                    "Probe a nearby variation around another strong candidate from previous rounds.",
                    mutate_params(reference["params"], search_space, rng, strong=False),
                )

    while len(candidates) < candidates_per_round:
        reference = base_params
        rationale = "Explore a wider neighborhood in the search space for overlooked parameter pockets."
        references = select_history_references(history)
        if references:
            reference = references[rng.randrange(len(references))]["params"]
            rationale = "Explore around prior high-ranking candidates without fully collapsing onto one solution."
        append_candidate(
            f"explore_{len(candidates) + 1}",
            rationale,
            mutate_params(reference, search_space, rng, strong=True),
        )

    return candidates[:candidates_per_round]


def attach_round_metadata(
    results: dict[str, Any],
    proposals: list[dict[str, Any]],
    round_index: int,
    combo_offset: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    proposal_names = {index: proposal["name"] for index, proposal in enumerate(proposals, start=1)}
    proposal_rationales = {index: proposal["rationale"] for index, proposal in enumerate(proposals, start=1)}

    symbol_results = results["symbol_results"].copy()
    aggregate_results = results["aggregate_results"].copy()
    selection_table = results["selection_table"].copy()

    for frame in (symbol_results, aggregate_results, selection_table):
        frame["round"] = round_index
        frame["round_combo_id"] = frame["combo_id"]
        frame["combo_id"] = frame["combo_id"] + combo_offset
        frame["proposal_name"] = frame["round_combo_id"].map(proposal_names)
        frame["proposal_rationale"] = frame["round_combo_id"].map(proposal_rationales)

    return symbol_results, aggregate_results, selection_table


def render_majors_gpt_report(
    payload: dict[str, Any],
    selection_table: pd.DataFrame,
    round_history: list[dict[str, Any]],
    model_name: str,
    used_openai: bool,
) -> str:
    best_combo = payload.get("best_combo")
    lines = [
        "# 主流币 5m GPT 调参报告",
        "",
        f"- 模型: {model_name}",
        f"- GPT 在线提案: {'是' if used_openai else '否（本地回退模式）'}",
        f"- 覆盖币种: {', '.join(payload.get('symbols', []))}",
        f"- 训练月份: {', '.join(payload.get('split_months', {}).get('train', [])) or '无'}",
        f"- 验证月份: {', '.join(payload.get('split_months', {}).get('valid', [])) or '无'}",
        f"- 已完成轮数: {len(round_history)}",
        "",
    ]

    if not best_combo:
        lines.append("没有得到可用的最优组合。")
        return "\n".join(lines)

    lines.extend(["## 最优参数", ""])
    for key, value in best_combo["params"].items():
        lines.append(f"- `{key}`: {value}")

    lines.extend(["", "## 分组表现", ""])
    for split_name, metrics in best_combo["split_metrics"].items():
        lines.append(
            f"- `{split_name}`: score={metrics['robustness_score']}, "
            f"mean_return={metrics['mean_return_pct']}%, profitable_ratio={metrics['profitable_ratio']}, "
            f"avg_dd={metrics['avg_max_drawdown_pct']}%, avg_pf={metrics['avg_profit_factor']}, "
            f"mean_trades={metrics['mean_total_trades']}"
        )

    lines.extend(["", "## 验证集币种明细", ""])
    valid_rows = best_combo["per_symbol"].get("valid", best_combo["per_symbol"].get("train", []))
    for row in valid_rows:
        lines.append(
            f"- `{row['symbol']}`: return={row['return_pct']}%, max_dd={row['max_drawdown_pct']}%, "
            f"trades={row['total_trades']}, pf={row['profit_factor']}, win_rate={row['win_rate_pct']}%"
        )

    lines.extend(["", "## 各轮最佳组合", ""])
    for round_summary in round_history:
        top_candidates = round_summary.get("top_candidates", [])
        if not top_candidates:
            continue
        best = top_candidates[0]
        valid_metrics = best["split_metrics"].get("valid", best["split_metrics"].get("train", {}))
        lines.append(
            f"- round {round_summary['round']}: selection={best['selection_score']}, "
            f"valid_score={valid_metrics.get('robustness_score', 'n/a')}, "
            f"valid_return={valid_metrics.get('mean_return_pct', 'n/a')}%, "
            f"params={best['params']}"
        )

    lines.extend(["", "## 全局前五组合", ""])
    for _, row in selection_table.head(5).iterrows():
        params = ", ".join(
            f"{column.replace('param_', '')}={row[column]}"
            for column in selection_table.columns
            if column.startswith("param_")
        )
        train_score = row.get("train_robustness_score", "n/a")
        valid_score = row.get("valid_robustness_score", "n/a")
        lines.append(
            f"- round {int(row['round'])} / combo {int(row['combo_id'])}: selection={round(float(row['selection_score']), 4)}, "
            f"train={train_score}, valid={valid_score}; {params}"
        )

    return "\n".join(lines)
