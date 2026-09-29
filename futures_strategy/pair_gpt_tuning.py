from __future__ import annotations

import json
import random
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from typing import Any

import pandas as pd
import requests

from futures_strategy.pairs import PairConfig

DEFAULT_PAIR_OPENAI_MODEL = "gpt-5.4"


@dataclass(frozen=True)
class ParameterSpec:
    kind: str
    minimum: float
    maximum: float
    step: float


DEFAULT_PAIR_SEARCH_SPACE = {
    "beta_window": ParameterSpec(kind="int", minimum=288, maximum=720, step=144),
    "zscore_window": ParameterSpec(kind="int", minimum=288, maximum=720, step=144),
    "entry_z": ParameterSpec(kind="float", minimum=1.8, maximum=3.4, step=0.2),
    "exit_z": ParameterSpec(kind="float", minimum=0.1, maximum=0.8, step=0.1),
    "stop_z": ParameterSpec(kind="float", minimum=3.5, maximum=5.5, step=0.5),
    "min_correlation": ParameterSpec(kind="float", minimum=0.8, maximum=0.95, step=0.01),
    "correlation_exit_buffer": ParameterSpec(kind="float", minimum=0.05, maximum=0.2, step=0.01),
    "max_holding_bars": ParameterSpec(kind="int", minimum=48, maximum=192, step=24),
}


def strategy_param_snapshot(config: PairConfig, search_space: dict[str, ParameterSpec]) -> dict[str, float]:
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
    if params["exit_z"] >= params["entry_z"]:
        params["exit_z"] = round(
            max(search_space["exit_z"].minimum, params["entry_z"] - 0.4),
            decimal_places(search_space["exit_z"].step),
        )
    if params["stop_z"] <= params["entry_z"]:
        params["stop_z"] = round(
            min(search_space["stop_z"].maximum, params["entry_z"] + 1.5),
            decimal_places(search_space["stop_z"].step),
        )
    if params["correlation_exit_buffer"] >= params["min_correlation"]:
        params["correlation_exit_buffer"] = round(
            max(search_space["correlation_exit_buffer"].minimum, params["min_correlation"] * 0.25),
            decimal_places(search_space["correlation_exit_buffer"].step),
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


def summarize_round_history(
    round_index: int,
    selection_table: pd.DataFrame,
    aggregate_results: pd.DataFrame,
    cell_results: pd.DataFrame,
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
        aggregate_subset = aggregate_results[aggregate_results["combo_id"] == combo_id]
        cell_subset = cell_results[cell_results["combo_id"] == combo_id]
        split_metrics = {
            split_name: aggregate_subset[aggregate_subset["split"] == split_name].iloc[0].to_dict()
            for split_name in aggregate_subset["split"].drop_duplicates()
        }
        valid_rows = (
            cell_subset[cell_subset["split"] == "valid"]
            .groupby("pair")
            .agg(mean_return_pct=("return_pct", "mean"), windows_tested=("window_id", "nunique"))
            .reset_index()
            .sort_values("mean_return_pct", ascending=False)
            .to_dict(orient="records")
        )
        summary["top_candidates"].append(
            {
                "combo_id": combo_id,
                "selection_score": round(float(row["selection_score"]), 4),
                "params": {
                    column.replace("param_", ""): row[column]
                    for column in selection_table.columns
                    if column.startswith("param_")
                },
                "split_metrics": split_metrics,
                "per_pair": valid_rows,
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
    pairs: list[str],
    rolling_windows: list[dict[str, Any]],
    history: list[dict[str, Any]],
    search_space: dict[str, ParameterSpec],
    candidates_per_round: int,
) -> list[dict[str, str]]:
    system_prompt = (
        "You are a quantitative trading research assistant. "
        "Propose the next parameter candidates for a market-neutral pair-trading backtest. "
        "Prioritize rolling-window robustness across multiple pairs, not one lucky window."
    )
    user_payload = {
        "objective": {
            "priority_order": [
                "maximize validation robustness_score across rolling windows",
                "keep profitable_pair_ratio broad across the pair universe",
                "limit drawdown and avoid configurations that only work on one window",
                "prefer stable, low-frequency pair-trading candidates over noisy overfit variants",
            ],
            "strategy_context": "Mean-reversion spread trading between correlated perpetual futures pairs.",
            "candidates_per_round": candidates_per_round,
        },
        "pairs": pairs,
        "rolling_windows": rolling_windows,
        "base_params": base_params,
        "search_space": space_for_prompt(search_space),
        "history": history[-3:],
        "requirements": [
            "Return exactly the requested number of candidates.",
            "Stay within the search space and respect the step size conceptually.",
            "Include at least one conservative low-frequency variant and one broader-coverage variant.",
            "Avoid repeating the current best candidate unless you materially improve a weakness.",
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
    pairs: list[str],
    rolling_windows: list[dict[str, Any]],
    history: list[dict[str, Any]],
    search_space: dict[str, ParameterSpec],
    candidates_per_round: int,
    timeout_seconds: int = 90,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, str]]]:
    messages = build_openai_messages(
        base_params=base_params,
        pairs=pairs,
        rolling_windows=rolling_windows,
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
                    "name": "pair_parameter_candidates",
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


def handcrafted_seed_candidates(base_params: dict[str, float]) -> list[dict[str, Any]]:
    return [
        {
            "name": "baseline_anchor",
            "rationale": "Keep the current pair-trading baseline as the anchor.",
            "params": dict(base_params),
        },
        {
            "name": "conservative_extreme_only",
            "rationale": "Trade only more extreme spread expansions under stricter correlation control.",
            "params": {
                **base_params,
                "entry_z": base_params["entry_z"] + 0.2,
                "stop_z": base_params["stop_z"] + 0.5,
                "min_correlation": base_params["min_correlation"] + 0.01,
                "max_holding_bars": base_params["max_holding_bars"] - 24,
            },
        },
        {
            "name": "broader_coverage",
            "rationale": "Loosen triggers slightly to improve cross-pair participation and reduce zero-trade windows.",
            "params": {
                **base_params,
                "entry_z": base_params["entry_z"] - 0.2,
                "exit_z": base_params["exit_z"] + 0.1,
                "min_correlation": base_params["min_correlation"] - 0.02,
                "max_holding_bars": base_params["max_holding_bars"] + 24,
            },
        },
    ]


def select_history_references(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    references: list[dict[str, Any]] = []
    for round_summary in reversed(history):
        references.extend(round_summary.get("top_candidates", []))
    references.sort(key=lambda item: item.get("selection_score", -999.0), reverse=True)
    return references[:3]


def mutate_params(
    reference: dict[str, float],
    search_space: dict[str, ParameterSpec],
    rng: random.Random,
    strong: bool,
) -> dict[str, float]:
    mutated = dict(reference)
    change_count = min(len(search_space), 4 if strong else 3)
    keys = rng.sample(list(search_space.keys()), k=change_count)
    for key in keys:
        spec = search_space[key]
        step_units = rng.randint(1, 2 if strong else 1)
        direction = rng.choice([-1, 1])
        mutated[key] = reference[key] + (spec.step * step_units * direction)
    return mutated


def offline_candidate_proposals(
    base_params: dict[str, float],
    history: list[dict[str, Any]],
    search_space: dict[str, ParameterSpec],
    candidates_per_round: int,
    round_index: int,
    seed: int,
    seen_signatures: set[str] | None = None,
) -> list[dict[str, Any]]:
    rng = random.Random((seed * 20_011) + round_index)
    seen_signatures = set(seen_signatures or set())
    candidates: list[dict[str, Any]] = []

    def append_candidate(name: str, rationale: str, params: dict[str, Any]) -> None:
        coerced = coerce_candidate_params(params, base_params, search_space)
        signature = candidate_signature(coerced)
        if signature in seen_signatures:
            return
        seen_signatures.add(signature)
        candidates.append({"name": name, "rationale": rationale, "params": coerced})

    if not history:
        for proposal in handcrafted_seed_candidates(base_params):
            append_candidate(proposal["name"], proposal["rationale"], proposal["params"])
    else:
        references = select_history_references(history)
        if references:
            best = references[0]["params"]
            append_candidate(
                "exploit_best",
                "Stay near the current best rolling-window candidate while nudging its weakest axes.",
                mutate_params(best, search_space, rng, strong=False),
            )
            append_candidate(
                "lower_drawdown",
                "Push the best candidate toward stricter entries and shorter holding time.",
                {
                    **best,
                    "entry_z": best["entry_z"] + 0.2,
                    "stop_z": best["stop_z"] + 0.5,
                    "min_correlation": best["min_correlation"] + 0.01,
                    "max_holding_bars": best["max_holding_bars"] - 24,
                },
            )
            append_candidate(
                "improve_coverage",
                "Relax the best candidate slightly to improve pair and window coverage.",
                {
                    **best,
                    "entry_z": best["entry_z"] - 0.2,
                    "exit_z": best["exit_z"] + 0.1,
                    "min_correlation": best["min_correlation"] - 0.01,
                    "max_holding_bars": best["max_holding_bars"] + 24,
                },
            )

    while len(candidates) < candidates_per_round:
        reference = base_params
        if history:
            references = select_history_references(history)
            if references:
                reference = references[rng.randrange(len(references))]["params"]
        append_candidate(
            f"explore_{len(candidates) + 1}",
            "Explore a nearby region in the pair-trading search space.",
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

    cell_results = results["cell_results"].copy()
    aggregate_results = results["aggregate_results"].copy()
    selection_table = results["selection_table"].copy()

    for frame in (cell_results, aggregate_results, selection_table):
        frame["round"] = round_index
        frame["round_combo_id"] = frame["combo_id"]
        frame["combo_id"] = frame["combo_id"] + combo_offset
        frame["proposal_name"] = frame["round_combo_id"].map(proposal_names)
        frame["proposal_rationale"] = frame["round_combo_id"].map(proposal_rationales)

    return cell_results, aggregate_results, selection_table


def render_pair_gpt_report(
    payload: dict[str, Any],
    selection_table: pd.DataFrame,
    round_history: list[dict[str, Any]],
    model_name: str,
    used_openai: bool,
) -> str:
    best_combo = payload.get("best_combo")
    lines = [
        "# 多配对 GPT 调参报告",
        "",
        f"- 模型: {model_name}",
        f"- GPT 在线提案: {'是' if used_openai else '否（本地回退模式）'}",
        f"- 覆盖配对: {', '.join(payload.get('pairs', []))}",
        f"- 滚动窗口数: {len(payload.get('rolling_windows', []))}",
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
            f"mean_return={metrics['mean_return_pct']}%, profitable_pair_ratio={metrics['profitable_pair_ratio']}, "
            f"avg_dd={metrics['avg_max_drawdown_pct']}%, avg_pf={metrics['avg_profit_factor']}, "
            f"mean_trades={metrics['mean_total_trades']}"
        )

    lines.extend(["", "## 验证集配对明细", ""])
    valid_rows = best_combo["per_pair"].get("valid", best_combo["per_pair"].get("train", []))
    for row in valid_rows:
        lines.append(
            f"- `{row['pair']}`: mean_return={round(float(row['mean_return_pct']), 4)}%, "
            f"min_return={round(float(row['min_return_pct']), 4)}%, "
            f"avg_dd={round(float(row['avg_max_drawdown_pct']), 4)}%, "
            f"trades={int(row['total_trades'])}, profitable_windows={int(row['profitable_windows'])}/{int(row['windows_tested'])}"
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
