from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from futures_strategy.optimization import expand_months
from futures_strategy.pair_gpt_tuning import (
    DEFAULT_PAIR_OPENAI_MODEL,
    DEFAULT_PAIR_SEARCH_SPACE,
    attach_round_metadata,
    build_openai_messages,
    candidate_signature,
    coerce_candidate_params,
    offline_candidate_proposals,
    render_pair_gpt_report,
    request_openai_candidates,
    strategy_param_snapshot,
    summarize_round_history,
)
from futures_strategy.pair_optimization import (
    DEFAULT_PAIR_UNIVERSE,
    best_pair_payload,
    build_rolling_windows,
    optimize_pair_universe,
    parse_pair_list,
)
from futures_strategy.pairs import load_pair_config


def combined_output_dir(output_dir: str | None, months: list[str]) -> Path:
    if output_dir:
        return Path(output_dir)
    return Path("outputs") / f"pairs_gpt_optimizer_{months[0]}__{months[-1]}"


def unique_proposals(
    proposals: list[dict[str, Any]],
    base_params: dict[str, float],
    seen_signatures: set[str],
) -> list[dict[str, Any]]:
    local_seen = set(seen_signatures)
    unique: list[dict[str, Any]] = []
    for proposal in proposals:
        coerced = coerce_candidate_params(proposal.get("params"), base_params, DEFAULT_PAIR_SEARCH_SPACE)
        signature = candidate_signature(coerced)
        if signature in local_seen:
            continue
        local_seen.add(signature)
        unique.append(
            {
                "name": proposal.get("name", f"candidate_{len(unique) + 1}"),
                "rationale": proposal.get("rationale", ""),
                "params": coerced,
            }
        )
    return unique


def write_best_config(base_config_path: str, best_params: dict[str, float], destination: Path) -> None:
    config = load_pair_config(base_config_path)
    for key, value in best_params.items():
        setattr(config.strategy, key, value)
    config.output.output_dir = str(destination.parent)
    destination.write_text(
        yaml.safe_dump(asdict(config), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune the pair-trading strategy with GPT proposals and rolling validation")
    parser.add_argument("--base-config", default="configs/binance_pairs_base.yaml", help="Base YAML config for pair strategy settings")
    parser.add_argument(
        "--pairs",
        default=",".join(f"{left}:{right}" for left, right in DEFAULT_PAIR_UNIVERSE),
        help="Comma separated pairs, such as BTCUSDT:ETHUSDT,SOLUSDT:ETHUSDT,BNBUSDT:ETHUSDT",
    )
    parser.add_argument("--months", default="2025-09:2026-02", help="Months to scan, e.g. 2025-09:2026-02")
    parser.add_argument("--train-window-months", type=int, default=2, help="How many months in each rolling train window")
    parser.add_argument("--valid-window-months", type=int, default=1, help="How many months in each rolling validation window")
    parser.add_argument("--market", default="um", choices=["um", "cm"], help="Binance archive market type")
    parser.add_argument("--cache-dir", default="data/binance_archive", help="Folder for monthly CSV cache")
    parser.add_argument("--output-dir", default=None, help="Folder for optimization outputs")
    parser.add_argument("--rounds", type=int, default=3, help="How many GPT/backtest rounds to run")
    parser.add_argument("--candidates-per-round", type=int, default=4, help="How many parameter candidates per round")
    parser.add_argument("--model", default=DEFAULT_PAIR_OPENAI_MODEL, help="OpenAI model name for online proposal rounds")
    parser.add_argument("--dry-run", action="store_true", help="Skip OpenAI calls and use the offline fallback proposer")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for offline fallback proposals")
    args = parser.parse_args()

    base_config = load_pair_config(args.base_config)
    pairs = parse_pair_list(args.pairs, DEFAULT_PAIR_UNIVERSE)
    months = expand_months(args.months)
    rolling_windows = build_rolling_windows(months, args.train_window_months, args.valid_window_months)
    pair_labels = [f"{left}:{right}" for left, right in pairs]
    output_dir = combined_output_dir(args.output_dir, months)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_params = strategy_param_snapshot(base_config, DEFAULT_PAIR_SEARCH_SPACE)
    seen_signatures: set[str] = set()
    round_history: list[dict[str, Any]] = []
    all_cell_results: list[pd.DataFrame] = []
    all_aggregate_results: list[pd.DataFrame] = []
    all_selection_tables: list[pd.DataFrame] = []
    combo_offset = 0
    api_key = os.getenv("OPENAI_API_KEY")
    gpt_requested = bool(api_key) and not args.dry_run
    openai_round_used = False

    print("=== 多配对 GPT 调参设置 ===")
    print(f"base_config: {Path(args.base_config).resolve()}")
    print(f"pairs: {pair_labels}")
    print(f"months: {months}")
    print(f"rolling_windows: {len(rolling_windows)}")
    print(f"rounds: {args.rounds}")
    print(f"candidates_per_round: {args.candidates_per_round}")
    print(f"model: {args.model}")
    print(f"gpt_online: {gpt_requested}")
    print(f"output_dir: {output_dir.resolve()}")
    print("")

    for round_index in range(1, args.rounds + 1):
        round_dir = output_dir / f"round_{round_index:02d}"
        round_dir.mkdir(parents=True, exist_ok=True)

        prompt_messages = build_openai_messages(
            base_params=base_params,
            pairs=pair_labels,
            rolling_windows=rolling_windows,
            history=round_history,
            search_space=DEFAULT_PAIR_SEARCH_SPACE,
            candidates_per_round=args.candidates_per_round,
        )
        proposals: list[dict[str, Any]]
        openai_payload: dict[str, Any] | None = None
        proposal_source = "offline"

        if gpt_requested:
            try:
                raw_candidates, openai_payload, prompt_messages = request_openai_candidates(
                    api_key=api_key or "",
                    model=args.model,
                    base_params=base_params,
                    pairs=pair_labels,
                    rolling_windows=rolling_windows,
                    history=round_history,
                    search_space=DEFAULT_PAIR_SEARCH_SPACE,
                    candidates_per_round=args.candidates_per_round,
                )
                proposals = unique_proposals(raw_candidates, base_params, seen_signatures)
                proposal_source = "openai"
                if proposals:
                    openai_round_used = True
            except Exception as exc:
                print(f"round {round_index}: OpenAI 提案失败，回退到本地模式: {exc}")
                proposals = offline_candidate_proposals(
                    base_params=base_params,
                    history=round_history,
                    search_space=DEFAULT_PAIR_SEARCH_SPACE,
                    candidates_per_round=args.candidates_per_round,
                    round_index=round_index,
                    seed=args.seed,
                    seen_signatures=seen_signatures,
                )
        else:
            proposals = offline_candidate_proposals(
                base_params=base_params,
                history=round_history,
                search_space=DEFAULT_PAIR_SEARCH_SPACE,
                candidates_per_round=args.candidates_per_round,
                round_index=round_index,
                seed=args.seed,
                seen_signatures=seen_signatures,
            )

        if len(proposals) < args.candidates_per_round:
            filler = offline_candidate_proposals(
                base_params=base_params,
                history=round_history,
                search_space=DEFAULT_PAIR_SEARCH_SPACE,
                candidates_per_round=args.candidates_per_round * 2,
                round_index=round_index + 101,
                seed=args.seed,
                seen_signatures=seen_signatures,
            )
            proposals = unique_proposals([*proposals, *filler], base_params, seen_signatures)

        proposals = proposals[: args.candidates_per_round]
        seen_signatures.update(candidate_signature(proposal["params"]) for proposal in proposals)
        param_grid = [proposal["params"] for proposal in proposals]

        print(f"=== round {round_index}/{args.rounds} ===")
        print(f"proposal_source: {proposal_source}")
        for index, proposal in enumerate(proposals, start=1):
            print(f"{index}. {proposal['name']}: {proposal['params']}")
        print("")

        (round_dir / "prompt_messages.json").write_text(
            json.dumps(prompt_messages, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (round_dir / "proposals.json").write_text(
            json.dumps(proposals, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        if openai_payload is not None:
            (round_dir / "openai_response.json").write_text(
                json.dumps(openai_payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        results = optimize_pair_universe(
            base_config=base_config,
            pairs=pairs,
            rolling_windows=rolling_windows,
            param_grid=param_grid,
            cache_dir=Path(args.cache_dir),
            market=args.market,
            logger=print,
        )
        cell_results, aggregate_results, selection_table = attach_round_metadata(
            results=results,
            proposals=proposals,
            round_index=round_index,
            combo_offset=combo_offset,
        )
        combo_offset = int(selection_table["combo_id"].max()) if not selection_table.empty else combo_offset

        cell_results.to_csv(round_dir / "cell_results.csv", index=False)
        aggregate_results.to_csv(round_dir / "aggregate_results.csv", index=False)
        selection_table.to_csv(round_dir / "selection_table.csv", index=False)

        round_prompt_summary = summarize_round_history(
            round_index=round_index,
            selection_table=selection_table,
            aggregate_results=aggregate_results,
            cell_results=cell_results,
        )
        round_history.append(round_prompt_summary)
        (round_dir / "round_summary.json").write_text(
            json.dumps(round_prompt_summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        all_cell_results.append(cell_results)
        all_aggregate_results.append(aggregate_results)
        all_selection_tables.append(selection_table)

        if not selection_table.empty:
            best = selection_table.iloc[0]
            print(
                f"round {round_index} best: combo={int(best['combo_id'])}, "
                f"selection_score={round(float(best['selection_score']), 4)}, "
                f"proposal={best['proposal_name']}"
            )
        print("")

    if not all_cell_results or not all_aggregate_results or not all_selection_tables:
        raise RuntimeError("No tuning results were produced.")

    combined_cell_results = pd.concat(all_cell_results, ignore_index=True)
    combined_aggregate_results = pd.concat(all_aggregate_results, ignore_index=True)
    combined_selection_table = pd.concat(all_selection_tables, ignore_index=True)
    combined_selection_table = combined_selection_table.sort_values("selection_score", ascending=False).reset_index(drop=True)

    payload = best_pair_payload(
        selection_table=combined_selection_table,
        aggregate_results=combined_aggregate_results,
        cell_results=combined_cell_results,
        rolling_windows=rolling_windows,
        pairs=results["usable_pairs"],
    )
    payload["meta"] = {
        "model": args.model,
        "used_openai": openai_round_used,
        "openai_api_key_present": bool(api_key),
        "rounds": args.rounds,
        "candidates_per_round": args.candidates_per_round,
        "months": months,
    }

    combined_cell_results.to_csv(output_dir / "cell_results.csv", index=False)
    combined_aggregate_results.to_csv(output_dir / "aggregate_results.csv", index=False)
    combined_selection_table.to_csv(output_dir / "selection_table.csv", index=False)
    (output_dir / "best_params.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "round_history.json").write_text(
        json.dumps(round_history, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "rolling_windows.json").write_text(
        json.dumps(rolling_windows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "report.md").write_text(
        render_pair_gpt_report(
            payload=payload,
            selection_table=combined_selection_table,
            round_history=round_history,
            model_name=args.model,
            used_openai=openai_round_used,
        ),
        encoding="utf-8",
    )

    best_combo = payload.get("best_combo")
    if best_combo is not None:
        write_best_config(args.base_config, best_combo["params"], output_dir / "best_config.yaml")

    print("=== 总结 ===")
    if best_combo is None:
        print("没有得到可用的最优组合。")
    else:
        print(f"best_combo: {best_combo['combo_id']}")
        print(f"selection_score: {best_combo['selection_score']}")
        for key, value in best_combo["params"].items():
            print(f"{key}: {value}")
    print(f"results_dir: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
