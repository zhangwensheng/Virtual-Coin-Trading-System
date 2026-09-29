"""Diagnose Williams %R features on the saved BOLL short trade set."""
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TRADES = ROOT / "outputs/boll_short_feature_diagnostics_20260911/trade_features.csv"
DATA = ROOT / "data/top_gainers_boll_short_1m/realtime_enriched"
DAILY = ROOT / "data/top_gainers_boll_short_1m/daily"
OUT = ROOT / "outputs/wr_trade_feature_diagnostics_20260911"


def williams_r(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    highest = high.rolling(period, min_periods=period).max()
    lowest = low.rolling(period, min_periods=period).min()
    return -100.0 * (highest - close) / (highest - lowest).replace(0.0, np.nan)


def previous_daily_wr(symbol: str, dates: pd.Series, closes: pd.Series, period: int) -> pd.Series:
    files = sorted(DAILY.glob(f"{symbol.lower()}_1d_*.csv"))
    if not files:
        return pd.Series(np.nan, index=dates.index)
    daily = pd.concat([pd.read_csv(file) for file in files], ignore_index=True)
    daily["timestamp"] = pd.to_datetime(daily["timestamp"], utc=True)
    daily = daily.drop_duplicates("timestamp").sort_values("timestamp").set_index("timestamp")
    for column in ["high", "low"]:
        daily[column] = pd.to_numeric(daily[column], errors="coerce")
    lookup = pd.DataFrame(
        {
            "trade_date": daily.index.normalize(),
            "prev_high": daily["high"].rolling(period, min_periods=period).max().shift(1),
            "prev_low": daily["low"].rolling(period, min_periods=period).min().shift(1),
        }
    ).dropna()
    merged = pd.DataFrame({"trade_date": dates, "close": closes}).merge(lookup, on="trade_date", how="left")
    return -100.0 * (merged["prev_high"] - merged["close"]) / (merged["prev_high"] - merged["prev_low"]).replace(0.0, np.nan)


def summarize(mask: pd.Series, frame: pd.DataFrame) -> dict:
    group = frame[mask]
    rest = frame[~mask]
    def stats(g):
        if g.empty:
            return {"n": 0, "win_pct": 0.0, "net": 0.0, "avg": 0.0, "pf": None}
        pnl = g["net_pnl"]
        gross_loss = float(-pnl[pnl <= 0].sum())
        return {
            "n": int(len(g)),
            "win_pct": float((pnl > 0).mean() * 100),
            "net": float(pnl.sum()),
            "avg": float(pnl.mean()),
            "pf": float(pnl[pnl > 0].sum() / gross_loss) if gross_loss else None,
        }
    return {"group": stats(group), "rest": stats(rest)}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    trades = pd.read_csv(TRADES)
    for column in ["entry_time", "breakdown_time", "exit_time"]:
        trades[column] = pd.to_datetime(trades[column], utc=True)
    trades["trade_date"] = trades["breakdown_time"].dt.normalize()
    trades["month"] = trades["entry_time"].dt.strftime("%Y-%m")

    enriched = []
    for symbol, group in trades.groupby("symbol", sort=False):
        files = sorted(DATA.glob(f"{symbol.lower()}_1m_*.csv"))
        if not files:
            continue
        frame = pd.concat([pd.read_csv(file) for file in files], ignore_index=True)
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
        frame = frame.drop_duplicates("timestamp").sort_values("timestamp").set_index("timestamp")
        for column in ["open", "high", "low", "close", "volume"]:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        for period in [9, 14, 20, 28]:
            frame[f"wr_{period}"] = williams_r(frame["high"], frame["low"], frame["close"], period)
        for idx, row in group.iterrows():
            breakdown = row["breakdown_time"]
            pump = row["entry_time"] - pd.Timedelta(minutes=int(row["pump_to_entry_minutes"]))
            if breakdown not in frame.index or pump not in frame.index:
                continue
            rec = trades.loc[idx].to_dict()
            for period in [9, 14, 20, 28]:
                rec[f"pump_wr_{period}"] = float(frame.at[pump, f"wr_{period}"])
                rec[f"breakdown_wr_{period}"] = float(frame.at[breakdown, f"wr_{period}"])
                rec[f"wr_drop_{period}"] = rec[f"pump_wr_{period}"] - rec[f"breakdown_wr_{period}"]
            for period in [14, 20, 28]:
                rec[f"daily_wr_{period}"] = float(
                    previous_daily_wr(
                        symbol,
                        pd.Series([row["trade_date"]]),
                        pd.Series([frame.at[breakdown, "close"]]),
                        period,
                    ).iloc[0]
                )
            enriched.append(rec)
    result = pd.DataFrame(enriched)
    result.to_csv(OUT / "wr_trade_features.csv", index=False)

    checks = []
    thresholds = []
    for period in [9, 14, 20, 28]:
        thresholds.extend(
            [
                (f"pump_wr_{period} >= -5", result[f"pump_wr_{period}"] >= -5),
                (f"pump_wr_{period} >= -10", result[f"pump_wr_{period}"] >= -10),
                (f"breakdown_wr_{period} <= -20", result[f"breakdown_wr_{period}"] <= -20),
                (f"breakdown_wr_{period} <= -30", result[f"breakdown_wr_{period}"] <= -30),
                (f"wr_drop_{period} >= 20", result[f"wr_drop_{period}"] >= 20),
                (f"wr_drop_{period} >= 30", result[f"wr_drop_{period}"] >= 30),
            ]
        )
    for period in [14, 20, 28]:
        thresholds.extend(
            [
                (f"daily_wr_{period} >= -10", result[f"daily_wr_{period}"] >= -10),
                (f"daily_wr_{period} >= -15", result[f"daily_wr_{period}"] >= -15),
                (f"daily_wr_{period} >= -20", result[f"daily_wr_{period}"] >= -20),
            ]
        )
    for name, mask in thresholds:
        row = {"rule": name, **summarize(mask.fillna(False), result)}
        row["group_avg_minus_rest"] = row["group"]["avg"] - row["rest"]["avg"]
        checks.append(row)
    checks_frame = pd.DataFrame(
        [
            {
                "rule": row["rule"],
                "n": row["group"]["n"],
                "win_pct": row["group"]["win_pct"],
                "net": row["group"]["net"],
                "avg": row["group"]["avg"],
                "pf": row["group"]["pf"],
                "rest_n": row["rest"]["n"],
                "rest_avg": row["rest"]["avg"],
                "avg_minus_rest": row["group_avg_minus_rest"],
            }
            for row in checks
        ]
    ).sort_values(["avg_minus_rest", "avg"], ascending=False)
    checks_frame.to_csv(OUT / "wr_threshold_checks.csv", index=False)
    medians = []
    wr_columns = [column for column in result.columns if column.startswith(("pump_wr_", "breakdown_wr_", "wr_drop_", "daily_wr_"))]
    for column in wr_columns:
        medians.append(
            {
                "feature": column,
                "winner_median": float(result.loc[result["net_pnl"] > 0, column].median()),
                "loser_median": float(result.loc[result["net_pnl"] <= 0, column].median()),
            }
        )
    pd.DataFrame(medians).to_csv(OUT / "wr_winner_loser_medians.csv", index=False)
    audit = {
        "rows": int(len(result)),
        "source": str(TRADES),
        "notes": [
            "These are entry-time features for the old accepted trade set, not a full WR strategy backtest.",
            "Williams %R is in [-100, 0]; values near 0 mean price is near the lookback high.",
        ],
    }
    (OUT / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"audit": audit, "top_rules": checks_frame.head(12).to_dict("records")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
