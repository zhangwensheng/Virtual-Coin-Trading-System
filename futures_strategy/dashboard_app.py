from __future__ import annotations

import json
import queue
import threading
import time
import tkinter as tk
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from tkinter import messagebox, scrolledtext, ttk
from typing import Any

import pandas as pd

from futures_strategy.auto_trader import (
    AutoTraderCycleResult,
    AutoTraderRuntimeConfig,
    apply_manual_flat_cooldown,
    execute_auto_trader_cycle,
)
from futures_strategy.binance_account_client import (
    BINANCE_FUTURES_BASE_URL,
    BINANCE_FUTURES_TESTNET_BASE_URL,
    BinanceCredentials,
    BinanceFuturesAccountClient,
)
from futures_strategy.binance_secure_store import (
    DashboardSettings,
    app_settings_dir,
    clear_dashboard_settings,
    default_auto_trader_config_dir,
    default_factor_cache_dir,
    load_dashboard_settings,
    save_dashboard_settings,
)


def format_timestamp_ms(value: Any) -> str:
    if value in (None, ""):
        return ""
    try:
        timestamp = int(value) / 1000
        return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")
    except (TypeError, ValueError, OSError):
        return format_iso_timestamp(value)


def format_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:,.4f}"
    return str(value)


def format_percent_value(value: Any, *, already_percent: bool = False) -> str:
    if value in (None, ""):
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not already_percent:
        number *= 100.0
    return f"{number:,.2f}%"


def format_bool_value(value: Any) -> str:
    if isinstance(value, bool):
        return "是" if value else "否"
    if str(value).lower() in {"true", "1", "yes"}:
        return "是"
    if str(value).lower() in {"false", "0", "no"}:
        return "否"
    return format_value(value)


def format_iso_timestamp(value: Any) -> str:
    if value in (None, ""):
        return ""
    try:
        timestamp = pd.Timestamp(value)
        if timestamp.tzinfo is not None:
            timestamp = timestamp.tz_convert("Asia/Shanghai")
        return timestamp.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:  # noqa: BLE001
        return str(value)


def symbols_to_text(symbols: list[str]) -> str:
    return ",".join(symbols)


def parse_symbols(raw: str) -> list[str]:
    return [item.strip().upper() for item in raw.split(",") if item.strip()]


def parse_paths(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


@dataclass
class DashboardResult:
    payload: dict[str, Any]
    include_history: bool
    fetched_at: float


class BinanceDashboardApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Binance Futures Dashboard")
        self.geometry("1520x980")
        self.minsize(1320, 820)

        self.settings = load_dashboard_settings()
        self.client: BinanceFuturesAccountClient | None = None
        self.result_queue: queue.Queue[tuple[str, Any]] = queue.Queue()
        self.refresh_inflight = False
        self.last_fast_refresh = 0.0
        self.last_slow_refresh = 0.0
        self.latest_payload: dict[str, Any] | None = None

        self.auto_trader_thread: threading.Thread | None = None
        self.auto_trader_stop_event: threading.Event | None = None
        self.auto_trader_running = False

        self.style = ttk.Style(self)
        with self._ignore_tcl_errors():
            self.style.theme_use("clam")

        self.summary_vars: dict[str, tk.StringVar] = {}
        self.summary_value_labels: dict[str, ttk.Label] = {}
        self.paper_summary_vars: dict[str, tk.StringVar] = {}
        self.status_var = tk.StringVar(value="请先在“API 设置”里填写 Binance Futures API Key。")
        self.last_update_var = tk.StringVar(value="未刷新")
        self.mode_var = tk.StringVar(value=self._mode_label(self.settings.use_testnet))
        self.auto_refresh_var = tk.BooleanVar(value=self.settings.auto_refresh)
        self.live_armed_var = tk.BooleanVar(value=False)
        self.auto_status_var = tk.StringVar(value="未启动")
        self.auto_last_cycle_var = tk.StringVar(value="未执行")
        self.paper_status_var = tk.StringVar(value="等待读取纸交易日志。")
        self.paper_output_dir_var = tk.StringVar(value=str(Path("outputs") / "top5_opportunity_paper_10x_1u"))

        self.api_key_var = tk.StringVar(value=self.settings.api_key)
        self.api_secret_var = tk.StringVar(value=self.settings.api_secret)
        self.tracked_symbols_var = tk.StringVar(value=symbols_to_text(self.settings.tracked_symbols))
        self.fast_refresh_var = tk.StringVar(value=str(self.settings.fast_refresh_sec))
        self.slow_refresh_var = tk.StringVar(value=str(self.settings.slow_refresh_sec))
        self.secret_visible_var = tk.BooleanVar(value=False)
        self.use_testnet_var = tk.BooleanVar(value=self.settings.use_testnet)

        self.auto_config_dir_var = tk.StringVar(value=self.settings.auto_trader_config_dir)
        self.auto_configs_var = tk.StringVar(value=",".join(self.settings.auto_trader_configs))
        self.auto_cache_dir_var = tk.StringVar(value=self.settings.auto_trader_cache_dir)
        self.auto_poll_var = tk.StringVar(value=str(self.settings.auto_trader_poll_sec))
        self.auto_days_back_var = tk.StringVar(value=str(self.settings.auto_trader_days_back))
        self.auto_top_n_var = tk.StringVar(value=str(self.settings.auto_trader_top_n))
        self.auto_max_positions_var = tk.StringVar(value=str(self.settings.auto_trader_max_positions))
        self.auto_leverage_var = tk.StringVar(value=str(self.settings.auto_trader_leverage))
        self.auto_margin_type_var = tk.StringVar(value=self.settings.auto_trader_margin_type)
        self.auto_risk_var = tk.StringVar(value=str(self.settings.auto_trader_risk_per_trade))
        self.auto_working_type_var = tk.StringVar(value=self.settings.auto_trader_working_type)

        self.assets_tree: ttk.Treeview | None = None
        self.positions_tree: ttk.Treeview | None = None
        self.orders_tree: ttk.Treeview | None = None
        self.trades_tree: ttk.Treeview | None = None
        self.income_tree: ttk.Treeview | None = None
        self.auto_candidates_tree: ttk.Treeview | None = None
        self.auto_managed_tree: ttk.Treeview | None = None
        self.paper_positions_tree: ttk.Treeview | None = None
        self.paper_trades_tree: ttk.Treeview | None = None
        self.paper_top_ranked_tree: ttk.Treeview | None = None
        self.paper_gate_tree: ttk.Treeview | None = None
        self.paper_candidates_tree: ttk.Treeview | None = None
        self.paper_diagnostics_tree: ttk.Treeview | None = None
        self.paper_equity_tree: ttk.Treeview | None = None
        self.auto_log_text: scrolledtext.ScrolledText | None = None
        self.auto_start_button: ttk.Button | None = None
        self.auto_stop_button: ttk.Button | None = None
        self.close_selected_position_button: ttk.Button | None = None

        self._build_layout()
        self._ensure_client()
        self.after(250, self._drain_queue)
        self.after(1000, self._refresh_scheduler)
        self.after(1500, self._paper_refresh_scheduler)
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        if self.client is not None:
            self.trigger_refresh(include_history=True)

    class _ignore_tcl_errors:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return exc_type is tk.TclError

    def _mode_label(self, use_testnet: bool) -> str:
        return "账户模式: TESTNET" if use_testnet else "账户模式: LIVE"

    def _current_base_url(self) -> str:
        return BINANCE_FUTURES_TESTNET_BASE_URL if self.use_testnet_var.get() else BINANCE_FUTURES_BASE_URL

    def _build_layout(self) -> None:
        root = ttk.Frame(self, padding=12)
        root.pack(fill=tk.BOTH, expand=True)

        top_bar = ttk.Frame(root)
        top_bar.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(top_bar, text="Binance Futures Dashboard", font=("Microsoft YaHei UI", 16, "bold")).pack(side=tk.LEFT)
        ttk.Label(top_bar, textvariable=self.mode_var, foreground="#8a4f00").pack(side=tk.LEFT, padx=(12, 0))
        ttk.Label(top_bar, textvariable=self.last_update_var).pack(side=tk.RIGHT)
        ttk.Button(top_bar, text="手动刷新", command=lambda: self.trigger_refresh(include_history=True, force=True)).pack(
            side=tk.RIGHT, padx=8
        )
        ttk.Checkbutton(top_bar, text="自动刷新", variable=self.auto_refresh_var, command=self._toggle_auto_refresh).pack(
            side=tk.RIGHT, padx=8
        )

        status_bar = ttk.Frame(root)
        status_bar.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(status_bar, textvariable=self.status_var, foreground="#0f4c81").pack(side=tk.LEFT)

        notebook = ttk.Notebook(root)
        notebook.pack(fill=tk.BOTH, expand=True)

        overview_tab = ttk.Frame(notebook, padding=8)
        orders_tab = ttk.Frame(notebook, padding=8)
        income_tab = ttk.Frame(notebook, padding=8)
        auto_tab = ttk.Frame(notebook, padding=8)
        paper_tab = ttk.Frame(notebook, padding=8)
        settings_tab = ttk.Frame(notebook, padding=8)

        notebook.add(overview_tab, text="账户总览")
        notebook.add(orders_tab, text="挂单")
        notebook.add(income_tab, text="收益流水")
        notebook.add(auto_tab, text="自动交易")
        notebook.add(paper_tab, text="纸交易监控")
        notebook.add(settings_tab, text="API 设置")

        self._build_overview_tab(overview_tab)
        self._build_orders_tab(orders_tab)
        self._build_income_tab(income_tab)
        self._build_auto_tab(auto_tab)
        self._build_paper_tab(paper_tab)
        self._build_settings_tab(settings_tab)

    def _build_overview_tab(self, parent: ttk.Frame) -> None:
        metrics = ttk.LabelFrame(parent, text="账户指标", padding=10)
        metrics.pack(fill=tk.X, pady=(0, 10))
        metric_specs = [
            ("total_wallet_balance", "钱包总资金"),
            ("available_balance", "可用余额"),
            ("total_margin_balance", "保证金余额"),
            ("total_unrealized_profit", "未实现盈亏"),
            ("realized_pnl_24h", "24h 已实现"),
            ("realized_pnl_7d", "7d 已实现"),
            ("net_income_7d", "7d 净收入"),
            ("position_count", "持仓数量"),
            ("open_order_count", "挂单数量"),
        ]
        for index, (key, label) in enumerate(metric_specs):
            card = ttk.Frame(metrics, padding=(8, 6))
            row = 0
            column = index
            card.grid(row=row, column=column, sticky="nsew", padx=4, pady=4)
            metrics.columnconfigure(column, weight=1)
            value_var = tk.StringVar(value="--")
            self.summary_vars[key] = value_var
            ttk.Label(card, text=label).pack(anchor=tk.W)
            value_label = ttk.Label(card, textvariable=value_var, font=("Consolas", 12, "bold"))
            value_label.pack(anchor=tk.W, pady=(2, 0))
            self.summary_value_labels[key] = value_label

        detail_notebook = ttk.Notebook(parent)
        detail_notebook.pack(fill=tk.BOTH, expand=True)
        positions_tab = ttk.Frame(detail_notebook, padding=8)
        trades_tab = ttk.Frame(detail_notebook, padding=8)
        detail_notebook.add(positions_tab, text="持仓列表")
        detail_notebook.add(trades_tab, text="成交历史")
        self._build_positions_tab(positions_tab)
        self._build_trades_tab(trades_tab)

    def _build_positions_tab(self, parent: ttk.Frame) -> None:
        toolbar = ttk.Frame(parent)
        toolbar.pack(fill=tk.X, pady=(0, 8))
        self.close_selected_position_button = ttk.Button(
            toolbar,
            text="平掉所选持仓",
            command=self._close_selected_position,
        )
        self.close_selected_position_button.pack(side=tk.LEFT)
        ttk.Label(toolbar, text="先在下面选中一行持仓，再点击平仓。", foreground="#7a4b00").pack(side=tk.LEFT, padx=(12, 0))

        table_frame = ttk.Frame(parent)
        table_frame.pack(fill=tk.BOTH, expand=True)
        self.positions_tree = self._build_treeview(
            table_frame,
            columns=[
                ("symbol", 110),
                ("side", 80),
                ("qty", 100),
                ("entry_price", 120),
                ("mark_price", 120),
                ("notional", 120),
                ("unrealized_pnl", 120),
                ("roe_pct", 90),
                ("leverage", 80),
                ("liquidation_price", 120),
                ("margin_type", 90),
            ],
            headings={
                "symbol": "交易对",
                "side": "方向",
                "qty": "数量",
                "entry_price": "开仓均价",
                "mark_price": "标记价格",
                "notional": "名义价值",
                "unrealized_pnl": "未实现盈亏",
                "roe_pct": "ROE %",
                "leverage": "杠杆",
                "liquidation_price": "强平价格",
                "margin_type": "保证金模式",
            },
        )

    def _build_orders_tab(self, parent: ttk.Frame) -> None:
        self.orders_tree = self._build_treeview(
            parent,
            columns=[
                ("symbol", 110),
                ("side", 80),
                ("type", 110),
                ("status", 110),
                ("orig_qty", 100),
                ("price", 110),
                ("stop_price", 110),
                ("avg_price", 110),
                ("update_time", 160),
            ],
            headings={
                "symbol": "交易对",
                "side": "方向",
                "type": "类型",
                "status": "状态",
                "orig_qty": "委托数量",
                "price": "委托价",
                "stop_price": "触发价",
                "avg_price": "成交均价",
                "update_time": "更新时间",
            },
        )

    def _build_trades_tab(self, parent: ttk.Frame) -> None:
        self.trades_tree = self._build_treeview(
            parent,
            columns=[
                ("time", 160),
                ("symbol", 110),
                ("side", 80),
                ("position_side", 90),
                ("qty", 100),
                ("price", 110),
                ("realized_pnl", 120),
                ("commission", 110),
                ("commission_asset", 100),
                ("reason", 160),
            ],
            headings={
                "time": "时间",
                "symbol": "交易对",
                "side": "方向",
                "position_side": "持仓方向",
                "qty": "数量",
                "price": "成交价",
                "realized_pnl": "已实现盈亏",
                "commission": "手续费",
                "commission_asset": "手续费资产",
                "reason": "原因/事件",
            },
        )

    def _build_income_tab(self, parent: ttk.Frame) -> None:
        self.income_tree = self._build_treeview(
            parent,
            columns=[
                ("time", 160),
                ("income_type", 130),
                ("symbol", 110),
                ("income", 120),
                ("asset", 80),
                ("info", 260),
            ],
            headings={
                "time": "时间",
                "income_type": "类型",
                "symbol": "交易对",
                "income": "金额",
                "asset": "资产",
                "info": "备注",
            },
        )

    def _build_auto_tab(self, parent: ttk.Frame) -> None:
        header = ttk.LabelFrame(parent, text="运行状态", padding=10)
        header.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(header, textvariable=self.mode_var).grid(row=0, column=0, sticky=tk.W, padx=(0, 16))
        ttk.Label(header, textvariable=self.auto_status_var).grid(row=0, column=1, sticky=tk.W, padx=(0, 16))
        ttk.Label(header, textvariable=self.auto_last_cycle_var).grid(row=0, column=2, sticky=tk.W)

        controls = ttk.LabelFrame(parent, text="自动交易参数", padding=10)
        controls.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(controls, text="策略目录").grid(row=0, column=0, sticky=tk.W, pady=4)
        ttk.Entry(controls, textvariable=self.auto_config_dir_var, width=80).grid(row=0, column=1, sticky="ew", padx=8, pady=4)

        ttk.Label(controls, text="指定配置").grid(row=1, column=0, sticky=tk.W, pady=4)
        ttk.Entry(controls, textvariable=self.auto_configs_var, width=80).grid(row=1, column=1, sticky="ew", padx=8, pady=4)

        ttk.Label(controls, text="缓存目录").grid(row=2, column=0, sticky=tk.W, pady=4)
        ttk.Entry(controls, textvariable=self.auto_cache_dir_var, width=80).grid(row=2, column=1, sticky="ew", padx=8, pady=4)

        row3 = ttk.Frame(controls)
        row3.grid(row=3, column=0, columnspan=2, sticky="ew", pady=4)
        ttk.Label(row3, text="轮询秒数").pack(side=tk.LEFT)
        ttk.Entry(row3, textvariable=self.auto_poll_var, width=8).pack(side=tk.LEFT, padx=(6, 12))
        ttk.Label(row3, text="回看天数").pack(side=tk.LEFT)
        ttk.Entry(row3, textvariable=self.auto_days_back_var, width=8).pack(side=tk.LEFT, padx=(6, 12))
        ttk.Label(row3, text="候选 TopN").pack(side=tk.LEFT)
        ttk.Entry(row3, textvariable=self.auto_top_n_var, width=8).pack(side=tk.LEFT, padx=(6, 12))
        ttk.Label(row3, text="最大持仓").pack(side=tk.LEFT)
        ttk.Entry(row3, textvariable=self.auto_max_positions_var, width=8).pack(side=tk.LEFT, padx=(6, 12))
        ttk.Label(row3, text="杠杆").pack(side=tk.LEFT)
        ttk.Entry(row3, textvariable=self.auto_leverage_var, width=8).pack(side=tk.LEFT, padx=(6, 12))

        row4 = ttk.Frame(controls)
        row4.grid(row=4, column=0, columnspan=2, sticky="ew", pady=4)
        ttk.Label(row4, text="单笔风险").pack(side=tk.LEFT)
        ttk.Entry(row4, textvariable=self.auto_risk_var, width=10).pack(side=tk.LEFT, padx=(6, 12))
        ttk.Label(row4, text="保证金模式").pack(side=tk.LEFT)
        ttk.Combobox(
            row4,
            textvariable=self.auto_margin_type_var,
            values=["ISOLATED", "CROSSED"],
            width=12,
            state="readonly",
        ).pack(side=tk.LEFT, padx=(6, 12))
        ttk.Label(row4, text="触发价格").pack(side=tk.LEFT)
        ttk.Combobox(
            row4,
            textvariable=self.auto_working_type_var,
            values=["MARK_PRICE", "CONTRACT_PRICE"],
            width=14,
            state="readonly",
        ).pack(side=tk.LEFT, padx=(6, 12))
        ttk.Checkbutton(row4, text="实盘已解锁", variable=self.live_armed_var).pack(side=tk.LEFT, padx=(12, 0))

        buttons = ttk.Frame(controls)
        buttons.grid(row=5, column=0, columnspan=2, sticky=tk.W, pady=(10, 0))
        ttk.Button(buttons, text="保存参数", command=self._save_settings).pack(side=tk.LEFT)
        self.auto_start_button = ttk.Button(buttons, text="启动自动交易", command=self._start_auto_trader)
        self.auto_start_button.pack(side=tk.LEFT, padx=8)
        self.auto_stop_button = ttk.Button(buttons, text="停止自动交易", command=self._stop_auto_trader)
        self.auto_stop_button.pack(side=tk.LEFT, padx=8)
        ttk.Button(buttons, text="立即执行一轮", command=self._run_auto_trader_once).pack(side=tk.LEFT, padx=8)

        note = ttk.Label(
            controls,
            text="默认只在 Testnet 自动执行；切到 LIVE 后，必须勾选“实盘已解锁”才能启动自动交易。",
            foreground="#7a4b00",
        )
        note.grid(row=6, column=0, columnspan=2, sticky=tk.W, pady=(10, 0))
        controls.columnconfigure(1, weight=1)

        managed_frame = ttk.LabelFrame(parent, text="自动持仓", padding=8)
        managed_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        self.auto_managed_tree = self._build_treeview(
            managed_frame,
            columns=[
                ("symbol", 130),
                ("entry_time", 160),
                ("entry_price", 110),
                ("qty", 100),
                ("mark_price", 110),
                ("unrealized_pnl", 110),
                ("stop_price", 110),
                ("tp_price", 110),
                ("partial_taken", 90),
                ("status", 100),
            ],
            headings={
                "symbol": "交易对",
                "entry_time": "开仓时间",
                "entry_price": "开仓价",
                "qty": "数量",
                "mark_price": "标记价",
                "unrealized_pnl": "浮盈亏",
                "stop_price": "止损价",
                "tp_price": "止盈价",
                "partial_taken": "已分批",
                "status": "状态",
            },
        )

        candidate_frame = ttk.LabelFrame(parent, text="当前候选", padding=8)
        candidate_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        self.auto_candidates_tree = self._build_treeview(
            candidate_frame,
            columns=[
                ("symbol", 130),
                ("action", 220),
                ("portfolio_rank", 90),
                ("rank_score", 100),
                ("factor_score", 90),
                ("risk_multiplier", 90),
                ("funding_rate", 110),
                ("oi_value_change_pct", 110),
                ("stop_hint", 110),
            ],
            headings={
                "symbol": "交易对",
                "action": "动作",
                "portfolio_rank": "排名",
                "rank_score": "总分",
                "factor_score": "因子分",
                "risk_multiplier": "风险倍率",
                "funding_rate": "资金费率",
                "oi_value_change_pct": "OI 变化%",
                "stop_hint": "止损提示",
            },
        )

        log_frame = ttk.LabelFrame(parent, text="运行日志", padding=8)
        log_frame.pack(fill=tk.BOTH, expand=True)
        self.auto_log_text = scrolledtext.ScrolledText(log_frame, height=10, font=("Consolas", 10))
        self.auto_log_text.pack(fill=tk.BOTH, expand=True)
        self.auto_log_text.configure(state=tk.DISABLED)
        self._set_auto_running(False)

    def _build_paper_tab(self, parent: ttk.Frame) -> None:
        header = ttk.LabelFrame(parent, text="当前纸交易模式", padding=10)
        header.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(header, text="日志目录").grid(row=0, column=0, sticky=tk.W, pady=4)
        ttk.Entry(header, textvariable=self.paper_output_dir_var, width=90).grid(
            row=0, column=1, sticky="ew", padx=8, pady=4
        )
        ttk.Button(header, text="刷新纸交易", command=self._manual_refresh_paper).grid(
            row=0, column=2, sticky=tk.W, pady=4
        )
        quick = ttk.Frame(header)
        quick.grid(row=1, column=1, columnspan=2, sticky=tk.W, padx=8, pady=(2, 0))
        ttk.Button(
            quick,
            text="查看 Top5 纸交易",
            command=lambda: self._set_paper_output_dir(str(Path("outputs") / "top5_opportunity_paper_10x_1u")),
        ).pack(side=tk.LEFT)
        ttk.Button(
            quick,
            text="查看日K空头纸交易",
            command=lambda: self._set_paper_output_dir(str(Path("outputs") / "daily_short_gate_1m_paper_10x_1u")),
        ).pack(side=tk.LEFT, padx=8)
        ttk.Button(
            quick,
            text="查看日K空头实盘",
            command=lambda: self._set_paper_output_dir(str(Path("outputs") / "daily_short_gate_1m_live_10x_1u")),
        ).pack(side=tk.LEFT)
        ttk.Button(
            quick,
            text="查看短线转跌雷达",
            command=lambda: self._set_paper_output_dir(str(Path("outputs") / "shortline_downtrend_score_paper_10x_1u")),
        ).pack(side=tk.LEFT, padx=8)
        ttk.Button(
            quick,
            text="查看短线转跌实盘",
            command=lambda: self._set_paper_output_dir(str(Path("outputs") / "shortline_downtrend_score_live_10x_1u")),
        ).pack(side=tk.LEFT)
        ttk.Label(header, textvariable=self.paper_status_var, foreground="#0f4c81").grid(
            row=2, column=0, columnspan=3, sticky=tk.W, pady=(6, 0)
        )
        ttk.Label(
            header,
            text="这个页面只读取本地纸交易日志，不会发送 Binance 真实订单；适合盯实时持仓、成交、候选和权益变化。",
            foreground="#7a4b00",
        ).grid(row=3, column=0, columnspan=3, sticky=tk.W, pady=(6, 0))
        header.columnconfigure(1, weight=1)

        metrics = ttk.LabelFrame(parent, text="纸账户概览", padding=10)
        metrics.pack(fill=tk.X, pady=(0, 10))
        metric_specs = [
            ("mode", "模式"),
            ("strategy", "策略"),
            ("paper_equity", "账户权益"),
            ("realized_pnl", "已实现盈亏"),
            ("unrealized_pnl", "浮动盈亏"),
            ("open_positions", "持仓数"),
            ("candidate_count", "候选数"),
            ("prepared_symbols", "已准备币数"),
            ("cycle_time", "最近轮询"),
        ]
        for index, (key, label) in enumerate(metric_specs):
            card = ttk.Frame(metrics, padding=(8, 6))
            row = index // 3
            column = index % 3
            card.grid(row=row, column=column, sticky="nsew", padx=6, pady=6)
            metrics.columnconfigure(column, weight=1)
            value_var = tk.StringVar(value="--")
            self.paper_summary_vars[key] = value_var
            ttk.Label(card, text=label).pack(anchor=tk.W)
            ttk.Label(card, textvariable=value_var, font=("Consolas", 12, "bold")).pack(anchor=tk.W, pady=(2, 0))

        paper_notebook = ttk.Notebook(parent)
        paper_notebook.pack(fill=tk.BOTH, expand=True)

        positions_tab = ttk.Frame(paper_notebook, padding=8)
        trades_tab = ttk.Frame(paper_notebook, padding=8)
        gate_tab = ttk.Frame(paper_notebook, padding=8)
        ranked_tab = ttk.Frame(paper_notebook, padding=8)
        candidates_tab = ttk.Frame(paper_notebook, padding=8)
        diagnostics_tab = ttk.Frame(paper_notebook, padding=8)
        equity_tab = ttk.Frame(paper_notebook, padding=8)

        paper_notebook.add(positions_tab, text="纸持仓")
        paper_notebook.add(trades_tab, text="纸交易历史")
        paper_notebook.add(gate_tab, text="日K过滤")
        paper_notebook.add(ranked_tab, text="涨幅榜")
        paper_notebook.add(candidates_tab, text="候选信号")
        paper_notebook.add(diagnostics_tab, text="未开仓诊断")
        paper_notebook.add(equity_tab, text="权益流水")

        self.paper_positions_tree = self._build_treeview(
            positions_tab,
            columns=[
                ("symbol", 120),
                ("side", 80),
                ("entry_time", 170),
                ("entry_price", 110),
                ("mark_price", 110),
                ("remaining_qty", 110),
                ("notional_usdt", 110),
                ("margin_usdt", 100),
                ("leverage", 80),
                ("unrealized_pnl", 120),
                ("realized_pnl", 120),
                ("stop_price", 110),
                ("tp_price", 110),
                ("bars_held", 90),
                ("partial_taken", 90),
            ],
            headings={
                "symbol": "交易对",
                "side": "方向",
                "entry_time": "开仓时间",
                "entry_price": "开仓价",
                "mark_price": "现价",
                "remaining_qty": "剩余数量",
                "notional_usdt": "名义价值",
                "margin_usdt": "保证金",
                "leverage": "杠杆",
                "unrealized_pnl": "浮盈亏",
                "realized_pnl": "已锁定盈亏",
                "stop_price": "止损价",
                "tp_price": "止盈价",
                "bars_held": "持仓K数",
                "partial_taken": "已分批",
            },
        )

        self.paper_trades_tree = self._build_treeview(
            trades_tab,
            columns=[
                ("event", 150),
                ("timestamp", 170),
                ("symbol", 120),
                ("side", 80),
                ("entry_price", 110),
                ("exit_price", 110),
                ("price", 110),
                ("qty", 110),
                ("pnl", 120),
                ("realized_pnl", 120),
                ("reason", 120),
                ("leverage", 80),
                ("margin_usdt", 100),
            ],
            headings={
                "event": "事件",
                "timestamp": "时间",
                "symbol": "交易对",
                "side": "方向",
                "entry_price": "开仓价",
                "exit_price": "平仓价",
                "price": "成交价",
                "qty": "数量",
                "pnl": "本次盈亏",
                "realized_pnl": "已实现盈亏",
                "reason": "原因",
                "leverage": "杠杆",
                "margin_usdt": "保证金",
            },
        )

        self.paper_gate_tree = self._build_treeview(
            gate_tab,
            columns=[
                ("symbol", 120),
                ("daily_short_allowed", 100),
                ("whitelisted", 100),
                ("gate_rank", 90),
                ("score_trades", 90),
                ("score_profit_factor", 110),
                ("score_pnl_pct", 110),
                ("score_win_rate", 110),
                ("signal_day_return", 110),
                ("signal_volume_ratio", 120),
                ("signal_sell_volume_ratio", 130),
                ("signal_down_hour_ratio", 130),
                ("signal_atr_pct", 110),
            ],
            headings={
                "symbol": "交易对",
                "daily_short_allowed": "今日允许",
                "whitelisted": "滚动白名单",
                "gate_rank": "过滤排名",
                "score_trades": "评分交易数",
                "score_profit_factor": "滚动PF",
                "score_pnl_pct": "滚动收益",
                "score_win_rate": "滚动胜率",
                "signal_day_return": "信号日涨跌",
                "signal_volume_ratio": "信号量能",
                "signal_sell_volume_ratio": "卖压占比",
                "signal_down_hour_ratio": "下跌小时比",
                "signal_atr_pct": "ATR%",
            },
        )

        self.paper_top_ranked_tree = self._build_treeview(
            ranked_tab,
            columns=[
                ("daily_rank", 90),
                ("symbol", 120),
                ("daily_return_pct", 120),
                ("daily_close", 110),
                ("daily_upper_break", 120),
                ("selected_day", 110),
            ],
            headings={
                "daily_rank": "日内排名",
                "symbol": "交易对",
                "daily_return_pct": "日内涨幅",
                "daily_close": "当前价",
                "daily_upper_break": "日K破上轨",
                "selected_day": "进入策略池",
            },
        )

        self.paper_candidates_tree = self._build_treeview(
            candidates_tab,
            columns=[
                ("symbol", 120),
                ("timestamp", 170),
                ("rank_score", 110),
                ("close", 110),
                ("daily_rank", 90),
                ("daily_return_pct", 120),
                ("volume_ratio", 110),
                ("vwap_gap", 110),
            ],
            headings={
                "symbol": "交易对",
                "timestamp": "信号时间",
                "rank_score": "信号分",
                "close": "价格",
                "daily_rank": "日内排名",
                "daily_return_pct": "日内涨幅",
                "volume_ratio": "量能比",
                "vwap_gap": "VWAP距离",
            },
        )

        self.paper_diagnostics_tree = self._build_treeview(
            diagnostics_tab,
            columns=[
                ("symbol", 120),
                ("daily_rank", 90),
                ("daily_return_pct", 120),
                ("close", 110),
                ("rsi", 90),
                ("volume_ratio", 110),
                ("trade_ratio", 110),
                ("delta_ratio", 110),
                ("htf_volume_ratio", 130),
                ("vwap_gap_pct", 120),
                ("atr_pct", 110),
                ("failed_checks", 420),
            ],
            headings={
                "symbol": "交易对",
                "daily_rank": "日内排名",
                "daily_return_pct": "日内涨幅",
                "close": "价格",
                "rsi": "RSI",
                "volume_ratio": "量能比",
                "trade_ratio": "交易数比",
                "delta_ratio": "主动买卖",
                "htf_volume_ratio": "5m量能比",
                "vwap_gap_pct": "VWAP距离",
                "atr_pct": "ATR%",
                "failed_checks": "未通过条件",
            },
        )

        self.paper_equity_tree = self._build_treeview(
            equity_tab,
            columns=[
                ("timestamp", 170),
                ("paper_equity", 130),
                ("realized_pnl", 130),
                ("unrealized_pnl", 130),
                ("open_positions", 110),
                ("candidate_count", 110),
                ("prepared_symbols", 120),
            ],
            headings={
                "timestamp": "时间",
                "paper_equity": "纸账户权益",
                "realized_pnl": "已实现盈亏",
                "unrealized_pnl": "浮动盈亏",
                "open_positions": "持仓数",
                "candidate_count": "候选数",
                "prepared_symbols": "已准备币数",
            },
        )

    def _build_settings_tab(self, parent: ttk.Frame) -> None:
        form = ttk.LabelFrame(parent, text="Binance API", padding=12)
        form.pack(fill=tk.X)

        ttk.Label(form, text="API Key").grid(row=0, column=0, sticky=tk.W, pady=6)
        ttk.Entry(form, textvariable=self.api_key_var, width=72).grid(row=0, column=1, sticky="ew", padx=8, pady=6)

        ttk.Label(form, text="API Secret").grid(row=1, column=0, sticky=tk.W, pady=6)
        self.api_secret_entry = ttk.Entry(form, textvariable=self.api_secret_var, width=72, show="*")
        self.api_secret_entry.grid(row=1, column=1, sticky="ew", padx=8, pady=6)
        ttk.Checkbutton(
            form,
            text="显示",
            variable=self.secret_visible_var,
            command=self._toggle_secret_visibility,
        ).grid(row=1, column=2, sticky=tk.W)

        ttk.Label(form, text="账户模式").grid(row=2, column=0, sticky=tk.W, pady=6)
        ttk.Checkbutton(
            form,
            text="使用 Binance Futures Testnet",
            variable=self.use_testnet_var,
            command=self._handle_mode_toggle,
        ).grid(row=2, column=1, sticky=tk.W, padx=8, pady=6)

        ttk.Label(form, text="监控币种").grid(row=3, column=0, sticky=tk.W, pady=6)
        ttk.Entry(form, textvariable=self.tracked_symbols_var, width=72).grid(row=3, column=1, sticky="ew", padx=8, pady=6)

        ttk.Label(form, text="快刷秒数").grid(row=4, column=0, sticky=tk.W, pady=6)
        ttk.Entry(form, textvariable=self.fast_refresh_var, width=12).grid(row=4, column=1, sticky=tk.W, padx=8, pady=6)

        ttk.Label(form, text="慢刷秒数").grid(row=5, column=0, sticky=tk.W, pady=6)
        ttk.Entry(form, textvariable=self.slow_refresh_var, width=12).grid(row=5, column=1, sticky=tk.W, padx=8, pady=6)

        button_row = ttk.Frame(form)
        button_row.grid(row=6, column=0, columnspan=3, sticky=tk.W, pady=(12, 0))
        ttk.Button(button_row, text="保存 Key", command=self._save_settings).pack(side=tk.LEFT)
        ttk.Button(button_row, text="测试连接", command=self._test_connection).pack(side=tk.LEFT, padx=8)
        ttk.Button(button_row, text="清空本地配置", command=self._clear_settings).pack(side=tk.LEFT)

        note = ttk.Label(
            form,
            text="自动交易默认先接 Testnet。实盘模式下请只使用你明确授权的 Binance Futures 交易 Key，且不要开启提币权限。",
            foreground="#7a4b00",
        )
        note.grid(row=7, column=0, columnspan=3, sticky=tk.W, pady=(12, 0))
        form.columnconfigure(1, weight=1)

    def _build_treeview(
        self,
        parent: ttk.Frame,
        *,
        columns: list[tuple[str, int]],
        headings: dict[str, str],
    ) -> ttk.Treeview:
        wrapper = ttk.Frame(parent)
        wrapper.pack(fill=tk.BOTH, expand=True)
        column_ids = [item[0] for item in columns]
        tree = ttk.Treeview(wrapper, columns=column_ids, show="headings", height=18)
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        y_scroll = ttk.Scrollbar(wrapper, orient=tk.VERTICAL, command=tree.yview)
        y_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        x_scroll = ttk.Scrollbar(parent, orient=tk.HORIZONTAL, command=tree.xview)
        x_scroll.pack(fill=tk.X)
        tree.configure(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)

        for column_id, width in columns:
            tree.heading(column_id, text=headings[column_id])
            tree.column(column_id, width=width, minwidth=width, anchor=tk.CENTER, stretch=True)
        tree.tag_configure("profit", foreground="#087f23")
        tree.tag_configure("loss", foreground="#b42318")
        tree.tag_configure("flat", foreground="#444b53")
        return tree

    def _toggle_secret_visibility(self) -> None:
        self.api_secret_entry.configure(show="" if self.secret_visible_var.get() else "*")

    def _toggle_auto_refresh(self) -> None:
        self.settings.auto_refresh = self.auto_refresh_var.get()
        save_dashboard_settings(self.settings)

    def _handle_mode_toggle(self) -> None:
        self.mode_var.set(self._mode_label(self.use_testnet_var.get()))
        self.client = None
        self._ensure_client()

    def _ensure_client(self) -> None:
        api_key = self.api_key_var.get().strip()
        api_secret = self.api_secret_var.get().strip()
        self.mode_var.set(self._mode_label(self.use_testnet_var.get()))
        if not api_key or not api_secret:
            self.client = None
            return
        credentials = BinanceCredentials(api_key=api_key, api_secret=api_secret)
        self.client = BinanceFuturesAccountClient(credentials, base_url=self._current_base_url())

    def _read_settings_from_form(self) -> DashboardSettings:
        tracked = parse_symbols(self.tracked_symbols_var.get())
        if not tracked:
            tracked = ["ADAUSDT", "HBARUSDT", "SOLUSDT"]
        return DashboardSettings(
            api_key=self.api_key_var.get().strip(),
            api_secret=self.api_secret_var.get().strip(),
            tracked_symbols=tracked,
            fast_refresh_sec=max(3, int(self.fast_refresh_var.get().strip() or "5")),
            slow_refresh_sec=max(10, int(self.slow_refresh_var.get().strip() or "30")),
            auto_refresh=self.auto_refresh_var.get(),
            use_testnet=self.use_testnet_var.get(),
            auto_trader_config_dir=self.auto_config_dir_var.get().strip() or default_auto_trader_config_dir(),
            auto_trader_configs=parse_paths(self.auto_configs_var.get()),
            auto_trader_cache_dir=self.auto_cache_dir_var.get().strip() or default_factor_cache_dir(),
            auto_trader_poll_sec=max(15, int(self.auto_poll_var.get().strip() or "60")),
            auto_trader_days_back=max(3, min(27, int(self.auto_days_back_var.get().strip() or "27"))),
            auto_trader_top_n=max(1, int(self.auto_top_n_var.get().strip() or "1")),
            auto_trader_max_positions=max(1, int(self.auto_max_positions_var.get().strip() or "1")),
            auto_trader_leverage=max(1, int(self.auto_leverage_var.get().strip() or "20")),
            auto_trader_margin_type=self.auto_margin_type_var.get().strip().upper() or "ISOLATED",
            auto_trader_risk_per_trade=max(0.001, float(self.auto_risk_var.get().strip() or "0.03")),
            auto_trader_working_type=self.auto_working_type_var.get().strip().upper() or "MARK_PRICE",
        )

    def _auto_runtime_from_settings(self, settings: DashboardSettings) -> AutoTraderRuntimeConfig:
        state_path = app_settings_dir() / "auto_trader_state.json"
        return AutoTraderRuntimeConfig(
            config_dir=settings.auto_trader_config_dir,
            configs=",".join(settings.auto_trader_configs),
            cache_dir=settings.auto_trader_cache_dir,
            state_path=str(state_path),
            poll_sec=settings.auto_trader_poll_sec,
            top_n=settings.auto_trader_top_n,
            days_back=settings.auto_trader_days_back,
            leverage=settings.auto_trader_leverage,
            margin_type=settings.auto_trader_margin_type,
            risk_per_trade=settings.auto_trader_risk_per_trade,
            max_positions=settings.auto_trader_max_positions,
            working_type=settings.auto_trader_working_type,
            use_testnet=settings.use_testnet,
        )

    def _save_settings(self) -> None:
        try:
            self.settings = self._read_settings_from_form()
        except ValueError:
            messagebox.showerror("保存失败", "请检查数字参数格式。")
            return
        save_dashboard_settings(self.settings)
        self._ensure_client()
        self.status_var.set("本地配置已保存。")
        if self.client is not None:
            self.trigger_refresh(include_history=True, force=True)

    def _clear_settings(self) -> None:
        clear_dashboard_settings()
        self.settings = DashboardSettings()
        self.api_key_var.set("")
        self.api_secret_var.set("")
        self.tracked_symbols_var.set(symbols_to_text(self.settings.tracked_symbols))
        self.fast_refresh_var.set(str(self.settings.fast_refresh_sec))
        self.slow_refresh_var.set(str(self.settings.slow_refresh_sec))
        self.auto_refresh_var.set(self.settings.auto_refresh)
        self.use_testnet_var.set(self.settings.use_testnet)
        self.auto_config_dir_var.set(self.settings.auto_trader_config_dir)
        self.auto_configs_var.set(",".join(self.settings.auto_trader_configs))
        self.auto_cache_dir_var.set(self.settings.auto_trader_cache_dir)
        self.auto_poll_var.set(str(self.settings.auto_trader_poll_sec))
        self.auto_days_back_var.set(str(self.settings.auto_trader_days_back))
        self.auto_top_n_var.set(str(self.settings.auto_trader_top_n))
        self.auto_max_positions_var.set(str(self.settings.auto_trader_max_positions))
        self.auto_leverage_var.set(str(self.settings.auto_trader_leverage))
        self.auto_margin_type_var.set(self.settings.auto_trader_margin_type)
        self.auto_risk_var.set(str(self.settings.auto_trader_risk_per_trade))
        self.auto_working_type_var.set(self.settings.auto_trader_working_type)
        self.mode_var.set(self._mode_label(self.settings.use_testnet))
        self.client = None
        self.status_var.set("本地配置已清空。")

    def _test_connection(self) -> None:
        try:
            self.settings = self._read_settings_from_form()
        except ValueError:
            messagebox.showerror("连接失败", "请检查数字参数格式。")
            return

        def worker() -> None:
            try:
                client = BinanceFuturesAccountClient(
                    BinanceCredentials(self.settings.api_key, self.settings.api_secret),
                    base_url=BINANCE_FUTURES_TESTNET_BASE_URL if self.settings.use_testnet else BINANCE_FUTURES_BASE_URL,
                )
                payload = client.ping_account()
                self.result_queue.put(("test_ok", payload))
            except Exception as exc:  # noqa: BLE001
                self.result_queue.put(("test_error", str(exc)))

        threading.Thread(target=worker, daemon=True).start()
        self.status_var.set("正在测试连接...")

    def trigger_refresh(self, *, include_history: bool, force: bool = False) -> None:
        if self.refresh_inflight and not force:
            return
        self._ensure_client()
        if self.client is None:
            self.status_var.set("请先填写并保存 Binance Futures API Key。")
            return

        self.refresh_inflight = True
        self.status_var.set("正在刷新账户数据...")

        try:
            settings = self._read_settings_from_form()
        except ValueError:
            self.refresh_inflight = False
            messagebox.showerror("刷新失败", "请检查数字参数格式。")
            return
        tracked_symbols = self._expanded_history_symbols(settings.tracked_symbols) if include_history else settings.tracked_symbols

        def worker() -> None:
            try:
                payload = self.client.fetch_dashboard_snapshot(
                    tracked_symbols,
                    include_history=include_history,
                )
                self.result_queue.put(
                    (
                        "refresh",
                        DashboardResult(payload=payload, include_history=include_history, fetched_at=time.time()),
                    )
                )
            except Exception as exc:  # noqa: BLE001
                self.result_queue.put(("error", str(exc)))

        threading.Thread(target=worker, daemon=True).start()

    def _paper_output_dir(self) -> Path:
        raw = self.paper_output_dir_var.get().strip() or str(Path("outputs") / "top5_opportunity_paper_10x_1u")
        path = Path(raw)
        if not path.is_absolute():
            path = Path.cwd() / path
        return path

    def _read_json_file(self, path: Path) -> dict[str, Any]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return {}
        return payload if isinstance(payload, dict) else {}

    def _read_jsonl_tail(self, path: Path, limit: int) -> list[dict[str, Any]]:
        try:
            with path.open("r", encoding="utf-8") as handle:
                lines = deque(handle, maxlen=limit)
        except OSError:
            return []
        rows: list[dict[str, Any]] = []
        for line in lines:
            raw = line.strip()
            if not raw:
                continue
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                rows.append(payload)
        return rows

    def _live_log_dirs(self) -> list[Path]:
        return [
            Path.cwd() / "outputs" / "shortline_downtrend_score_live_10x_1u",
            Path.cwd() / "outputs" / "daily_short_gate_1m_live_10x_1u",
        ]

    def _expanded_history_symbols(self, tracked_symbols: list[str]) -> list[str]:
        symbols = list(dict.fromkeys(str(item).strip().upper() for item in tracked_symbols if str(item).strip()))
        for directory in self._live_log_dirs():
            status = self._read_json_file(directory / "latest_status.json")
            positions = status.get("positions") or {}
            if isinstance(positions, dict):
                for symbol in positions:
                    normalized = str(symbol).strip().upper()
                    if normalized and normalized not in symbols:
                        symbols.append(normalized)
            for key in ["opened", "position_events", "candidates", "top_ranked"]:
                rows = status.get(key) or []
                if isinstance(rows, list):
                    for row in rows:
                        if not isinstance(row, dict):
                            continue
                        normalized = str(row.get("symbol", "")).strip().upper()
                        if normalized and normalized not in symbols:
                            symbols.append(normalized)
            for row in self._read_jsonl_tail(directory / "trades.jsonl", 120):
                normalized = str(row.get("symbol", "")).strip().upper()
                if normalized and normalized not in symbols:
                    symbols.append(normalized)
        return symbols[:60]

    def _local_live_trade_rows(self, limit: int = 200) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for directory in self._live_log_dirs():
            for row in self._read_jsonl_tail(directory / "trades.jsonl", limit):
                if not isinstance(row, dict):
                    continue
                rows.append(
                    {
                        "time": row.get("timestamp", ""),
                        "symbol": row.get("symbol", ""),
                        "side": row.get("side", ""),
                        "position_side": "LOCAL",
                        "qty": row.get("qty", row.get("remaining_qty", "")),
                        "price": row.get("price", row.get("entry_price", row.get("exit_price", ""))),
                        "realized_pnl": row.get("pnl", row.get("realized_pnl", "")),
                        "commission": "",
                        "commission_asset": "LOCAL",
                        "reason": row.get("reason", row.get("event", "")),
                    }
                )
        rows.sort(key=lambda item: str(item.get("time", "")), reverse=True)
        return rows[:limit]

    def _read_equity_tail(self, path: Path, limit: int) -> list[dict[str, Any]]:
        try:
            frame = pd.read_csv(path).tail(limit)
        except (FileNotFoundError, OSError, pd.errors.EmptyDataError, pd.errors.ParserError):
            return []
        return frame.to_dict("records")

    def _manual_refresh_paper(self) -> None:
        self._refresh_paper_tab()

    def _set_paper_output_dir(self, path: str) -> None:
        self.paper_output_dir_var.set(path)
        self._refresh_paper_tab()

    def _paper_refresh_scheduler(self) -> None:
        self._refresh_paper_tab()
        self.after(2000, self._paper_refresh_scheduler)

    def _refresh_paper_tab(self) -> None:
        paper_dir = self._paper_output_dir()
        status_path = paper_dir / "latest_status.json"
        status = self._read_json_file(status_path)
        if not status:
            self.paper_status_var.set(f"未读取到纸交易状态: {status_path}")
            return

        positions_payload = self._read_json_file(paper_dir / "positions.json")
        runtime_payload = self._read_json_file(paper_dir / "runtime.json")
        runtime = runtime_payload.get("runtime", {}) if isinstance(runtime_payload.get("runtime"), dict) else {}

        try:
            modified_at = datetime.fromtimestamp(status_path.stat().st_mtime)
            age_sec = max(0, int(time.time() - status_path.stat().st_mtime))
            stale_hint = "实时读取中" if age_sec <= 180 else f"日志已 {age_sec}s 未更新，检查纸交易进程"
            modified_text = modified_at.strftime("%Y-%m-%d %H:%M:%S")
        except OSError:
            stale_hint = "状态文件时间未知"
            modified_text = "--"

        leverage = status.get("leverage", runtime.get("leverage", "--"))
        margin = status.get("margin_usdt_per_trade", runtime.get("margin_usdt", "--"))
        notional = status.get("notional_usdt_per_trade", "")
        mode_text = f"{status.get('mode', '--')} | {leverage}x | {margin}U/笔"
        if notional not in ("", None):
            mode_text = f"{mode_text} | 名义 {format_value(notional)}U"

        positions = status.get("positions") or positions_payload.get("positions") or {}
        marks = status.get("marks") or positions_payload.get("marks") or {}
        if not isinstance(positions, dict):
            positions = {}
        if not isinstance(marks, dict):
            marks = {}

        summary_values = {
            "mode": mode_text,
            "strategy": status.get("strategy", runtime.get("config_path", "--")),
            "paper_equity": status.get("paper_equity", "--"),
            "realized_pnl": status.get("realized_pnl", "--"),
            "unrealized_pnl": status.get("unrealized_pnl", "--"),
            "open_positions": len(positions),
            "candidate_count": status.get("candidate_count", "--"),
            "prepared_symbols": f"{status.get('prepared_symbols', '--')}/{status.get('scanned_symbols', '--')}",
            "cycle_time": format_iso_timestamp(status.get("cycle_time")),
        }
        for key, value in summary_values.items():
            variable = self.paper_summary_vars.get(key)
            if variable is not None:
                variable.set(format_value(value))

        self.paper_status_var.set(f"{stale_hint} | 文件更新: {modified_text} | 目录: {paper_dir}")

        position_rows: list[dict[str, Any]] = []
        for symbol, position in positions.items():
            if not isinstance(position, dict):
                continue
            entry_price = float(position.get("entry_price", 0.0) or 0.0)
            remaining_qty = float(position.get("remaining_qty", position.get("qty", 0.0)) or 0.0)
            mark_price = marks.get(symbol)
            unrealized: float | str = ""
            try:
                mark_number = float(mark_price)
                side = str(position.get("side", "long")).lower()
                direction = -1.0 if side == "short" else 1.0
                unrealized = direction * (mark_number - entry_price) * remaining_qty
            except (TypeError, ValueError):
                mark_number = ""
            position_rows.append(
                {
                    **position,
                    "symbol": position.get("symbol", symbol),
                    "mark_price": mark_number,
                    "remaining_qty": remaining_qty,
                    "unrealized_pnl": unrealized,
                    "partial_taken": format_bool_value(position.get("partial_taken", False)),
                }
            )

        trade_rows = list(reversed(self._read_jsonl_tail(paper_dir / "trades.jsonl", 200)))
        ranked_rows = [
            {
                **row,
                "daily_return_pct": format_percent_value(row.get("daily_return_pct")),
                "daily_upper_break": format_bool_value(row.get("daily_upper_break")),
                "selected_day": format_bool_value(row.get("selected_day")),
            }
            for row in status.get("top_ranked", [])
            if isinstance(row, dict)
        ]
        candidate_rows = [
            {
                **row,
                "daily_return_pct": format_percent_value(row.get("daily_return_pct")),
                "vwap_gap": format_percent_value(row.get("vwap_gap")),
            }
            for row in status.get("candidates", [])
            if isinstance(row, dict)
        ]
        gate_rows = [
            {
                **row,
                "daily_short_allowed": format_bool_value(row.get("daily_short_allowed")),
                "whitelisted": format_bool_value(row.get("whitelisted")),
                "score_pnl_pct": format_percent_value(row.get("score_pnl_pct")),
                "score_win_rate": format_percent_value(row.get("score_win_rate")),
                "signal_day_return": format_percent_value(row.get("signal_day_return")),
                "signal_sell_volume_ratio": format_percent_value(row.get("signal_sell_volume_ratio")),
                "signal_down_hour_ratio": format_percent_value(row.get("signal_down_hour_ratio")),
                "signal_atr_pct": format_percent_value(row.get("signal_atr_pct")),
            }
            for row in status.get("gate_rows", [])
            if isinstance(row, dict)
        ]
        diagnostic_rows = [
            {
                **row,
                "daily_return_pct": format_percent_value(row.get("daily_return_pct"), already_percent=True),
                "vwap_gap_pct": format_percent_value(row.get("vwap_gap_pct"), already_percent=True),
                "atr_pct": format_percent_value(row.get("atr_pct"), already_percent=True),
                "failed_checks": ", ".join(row.get("failed_checks", [])),
            }
            for row in status.get("diagnostics", [])
            if isinstance(row, dict)
        ]
        equity_rows = list(reversed(self._read_equity_tail(paper_dir / "equity.csv", 200)))

        self._populate_tree(
            self.paper_positions_tree,
            position_rows,
            [
                "symbol",
                "side",
                "entry_time",
                "entry_price",
                "mark_price",
                "remaining_qty",
                "notional_usdt",
                "margin_usdt",
                "leverage",
                "unrealized_pnl",
                "realized_pnl",
                "stop_price",
                "tp_price",
                "bars_held",
                "partial_taken",
            ],
            iso_time_columns={"entry_time"},
        )
        self._populate_tree(
            self.paper_trades_tree,
            trade_rows,
            ["event", "timestamp", "symbol", "side", "entry_price", "exit_price", "price", "qty", "pnl", "realized_pnl", "reason", "leverage", "margin_usdt"],
            iso_time_columns={"timestamp"},
        )
        self._populate_tree(
            self.paper_top_ranked_tree,
            ranked_rows,
            ["daily_rank", "symbol", "daily_return_pct", "daily_close", "daily_upper_break", "selected_day"],
        )
        self._populate_tree(
            self.paper_gate_tree,
            gate_rows,
            [
                "symbol",
                "daily_short_allowed",
                "whitelisted",
                "gate_rank",
                "score_trades",
                "score_profit_factor",
                "score_pnl_pct",
                "score_win_rate",
                "signal_day_return",
                "signal_volume_ratio",
                "signal_sell_volume_ratio",
                "signal_down_hour_ratio",
                "signal_atr_pct",
            ],
        )
        self._populate_tree(
            self.paper_candidates_tree,
            candidate_rows,
            ["symbol", "timestamp", "rank_score", "close", "daily_rank", "daily_return_pct", "volume_ratio", "vwap_gap"],
            iso_time_columns={"timestamp"},
        )
        self._populate_tree(
            self.paper_diagnostics_tree,
            diagnostic_rows,
            [
                "symbol",
                "daily_rank",
                "daily_return_pct",
                "close",
                "rsi",
                "volume_ratio",
                "trade_ratio",
                "delta_ratio",
                "htf_volume_ratio",
                "vwap_gap_pct",
                "atr_pct",
                "failed_checks",
            ],
        )
        self._populate_tree(
            self.paper_equity_tree,
            equity_rows,
            ["timestamp", "paper_equity", "realized_pnl", "unrealized_pnl", "open_positions", "candidate_count", "prepared_symbols"],
            iso_time_columns={"timestamp"},
        )

    def _refresh_scheduler(self) -> None:
        try:
            settings = self._read_settings_from_form()
        except ValueError:
            self.after(1000, self._refresh_scheduler)
            return

        if self.auto_refresh_var.get() and not self.refresh_inflight and self.client is not None:
            now = time.time()
            slow_due = (now - self.last_slow_refresh) >= settings.slow_refresh_sec
            fast_due = (now - self.last_fast_refresh) >= settings.fast_refresh_sec
            if slow_due:
                self.trigger_refresh(include_history=True)
            elif fast_due:
                self.trigger_refresh(include_history=False)
        self.after(1000, self._refresh_scheduler)

    def _append_auto_log(self, message: str) -> None:
        if self.auto_log_text is None:
            return
        self.auto_log_text.configure(state=tk.NORMAL)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.auto_log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.auto_log_text.see(tk.END)
        self.auto_log_text.configure(state=tk.DISABLED)

    def _set_auto_running(self, running: bool) -> None:
        self.auto_trader_running = running
        if self.auto_start_button is not None:
            self.auto_start_button.configure(state=tk.DISABLED if running else tk.NORMAL)
        if self.auto_stop_button is not None:
            self.auto_stop_button.configure(state=tk.NORMAL if running else tk.DISABLED)
        if not running and self.auto_status_var.get() == "运行中":
            self.auto_status_var.set("已停止")

    def _start_auto_trader(self) -> None:
        if self.auto_trader_running:
            return
        try:
            self.settings = self._read_settings_from_form()
            runtime = self._auto_runtime_from_settings(self.settings)
            runtime.resolved_config_paths()
        except ValueError as exc:
            messagebox.showerror("启动失败", str(exc))
            return

        if not self.settings.use_testnet and not self.live_armed_var.get():
            messagebox.showwarning("未解锁实盘", "当前是 LIVE 模式，先勾选“实盘已解锁”后才能启动自动交易。")
            return

        save_dashboard_settings(self.settings)
        self._ensure_client()
        if self.client is None:
            messagebox.showerror("启动失败", "请先保存可用的 Binance API Key。")
            return

        self.auto_trader_stop_event = threading.Event()
        self._set_auto_running(True)
        self.auto_status_var.set("运行中")
        self._append_auto_log(f"自动交易已启动，模式: {runtime.mode_label}")

        api_key = self.settings.api_key
        api_secret = self.settings.api_secret
        base_url = self._current_base_url()
        stop_event = self.auto_trader_stop_event

        def worker() -> None:
            auto_client = BinanceFuturesAccountClient(
                BinanceCredentials(api_key, api_secret),
                base_url=base_url,
            )
            while stop_event is not None and not stop_event.is_set():
                try:
                    result = execute_auto_trader_cycle(auto_client, runtime)
                    self.result_queue.put(("auto_cycle", result))
                except Exception as exc:  # noqa: BLE001
                    self.result_queue.put(("auto_error", str(exc)))
                if stop_event.wait(runtime.poll_sec):
                    break
            self.result_queue.put(("auto_stopped", None))

        self.auto_trader_thread = threading.Thread(target=worker, daemon=True)
        self.auto_trader_thread.start()

    def _stop_auto_trader(self) -> None:
        if self.auto_trader_stop_event is not None:
            self.auto_trader_stop_event.set()
        self._append_auto_log("正在停止自动交易线程...")

    def _run_auto_trader_once(self) -> None:
        if self.auto_trader_running:
            messagebox.showinfo("提示", "自动交易已经在运行中。")
            return
        try:
            self.settings = self._read_settings_from_form()
            runtime = self._auto_runtime_from_settings(self.settings)
            runtime.resolved_config_paths()
        except ValueError as exc:
            messagebox.showerror("执行失败", str(exc))
            return

        if not self.settings.use_testnet and not self.live_armed_var.get():
            messagebox.showwarning("未解锁实盘", "当前是 LIVE 模式，先勾选“实盘已解锁”后才能执行。")
            return

        save_dashboard_settings(self.settings)
        self._append_auto_log(f"开始执行单轮自动交易，模式: {runtime.mode_label}")

        def worker() -> None:
            try:
                auto_client = BinanceFuturesAccountClient(
                    BinanceCredentials(self.settings.api_key, self.settings.api_secret),
                    base_url=self._current_base_url(),
                )
                result = execute_auto_trader_cycle(auto_client, runtime)
                self.result_queue.put(("auto_cycle", result))
            except Exception as exc:  # noqa: BLE001
                self.result_queue.put(("auto_error", str(exc)))

        threading.Thread(target=worker, daemon=True).start()

    def _selected_position_payload(self) -> dict[str, Any] | None:
        if self.positions_tree is None or not self.latest_payload:
            return None
        selection = self.positions_tree.selection()
        if not selection:
            return None
        values = self.positions_tree.item(selection[0], "values")
        if len(values) < 3:
            return None
        symbol = str(values[0])
        side = str(values[1])
        try:
            qty_hint = abs(float(str(values[2]).replace(",", "")))
        except ValueError:
            qty_hint = 0.0

        candidates = [
            row
            for row in self.latest_payload.get("positions", [])
            if str(row.get("symbol", "")) == symbol and str(row.get("side", "")) == side
        ]
        if not candidates:
            candidates = [row for row in self.latest_payload.get("positions", []) if str(row.get("symbol", "")) == symbol]
        if not candidates:
            return None
        if qty_hint > 0:
            candidates.sort(key=lambda row: abs(abs(float(row.get("qty", 0.0))) - qty_hint))
        return candidates[0]

    def _close_selected_position(self) -> None:
        self._ensure_client()
        if self.client is None:
            messagebox.showerror("平仓失败", "请先填写并保存 Binance Futures API Key。")
            return

        position = self._selected_position_payload()
        if position is None:
            messagebox.showwarning("未选择持仓", "请先在持仓列表中选中一行。")
            return

        symbol = str(position.get("symbol", ""))
        side = str(position.get("side", ""))
        raw_qty = float(position.get("qty", 0.0))
        position_side = str(position.get("position_side", "BOTH") or "BOTH")
        quantity = abs(raw_qty)
        if quantity <= 0:
            messagebox.showerror("平仓失败", "当前选中的持仓数量无效。")
            return

        mode_label = "TESTNET" if self.use_testnet_var.get() else "LIVE"
        confirm = messagebox.askyesno(
            "确认平仓",
            f"确认要在 {mode_label} 上平掉 {symbol} 的 {side} 持仓吗？\n\n数量: {quantity}",
        )
        if not confirm:
            return

        try:
            settings = self._read_settings_from_form()
        except ValueError:
            settings = self.settings
        runtime = self._auto_runtime_from_settings(settings)
        if self.close_selected_position_button is not None:
            self.close_selected_position_button.configure(state=tk.DISABLED)
        self.status_var.set(f"正在平仓 {symbol} ...")

        def worker() -> None:
            try:
                close_qty = self.client.quantize_quantity(symbol, quantity, market=True)
                if close_qty <= 0:
                    raise ValueError(f"Invalid close quantity for {symbol}: {quantity}")
                cancelled = self.client.cancel_all_open_orders(symbol)
                order = self.client.close_position_market(
                    symbol,
                    close_qty,
                    position_amt=raw_qty,
                    position_side=position_side,
                )
                apply_manual_flat_cooldown(runtime.state_path, symbol, cooldown_hours=4)
                self.result_queue.put(
                    (
                        "manual_close_ok",
                        {
                            "symbol": symbol,
                            "side": side,
                            "quantity": close_qty,
                            "cancelled": len(cancelled),
                            "order": order,
                        },
                    )
                )
            except Exception as exc:  # noqa: BLE001
                self.result_queue.put(("manual_close_error", {"symbol": symbol, "error": str(exc)}))

        threading.Thread(target=worker, daemon=True).start()

    def _drain_queue(self) -> None:
        while True:
            try:
                event, payload = self.result_queue.get_nowait()
            except queue.Empty:
                break

            if event == "refresh":
                assert isinstance(payload, DashboardResult)
                self._apply_payload(payload.payload, include_history=payload.include_history)
                self.refresh_inflight = False
                self.last_fast_refresh = payload.fetched_at
                if payload.include_history:
                    self.last_slow_refresh = payload.fetched_at
                self.last_update_var.set(f"最近更新: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                self.status_var.set("数据已更新。")
            elif event == "test_ok":
                self.status_var.set("连接成功，可以读取 Binance Futures 账户数据。")
                messagebox.showinfo("测试成功", "Binance Futures API 连接成功。")
            elif event == "test_error":
                self.refresh_inflight = False
                self.status_var.set(f"连接失败: {payload}")
                messagebox.showerror("连接失败", str(payload))
            elif event == "error":
                self.refresh_inflight = False
                self.status_var.set(f"刷新失败: {payload}")
            elif event == "auto_cycle":
                assert isinstance(payload, AutoTraderCycleResult)
                self._apply_auto_cycle(payload)
            elif event == "auto_error":
                self.auto_status_var.set(f"自动交易报错: {payload}")
                self._append_auto_log(f"自动交易报错: {payload}")
            elif event == "auto_stopped":
                self._set_auto_running(False)
                self.auto_status_var.set("已停止")
                self._append_auto_log("自动交易已停止。")
            elif event == "manual_close_ok":
                if self.close_selected_position_button is not None:
                    self.close_selected_position_button.configure(state=tk.NORMAL)
                details = dict(payload)
                symbol = str(details.get("symbol", ""))
                quantity = details.get("quantity", "")
                cancelled = details.get("cancelled", 0)
                self.status_var.set(f"{symbol} 平仓单已发送。")
                self._append_auto_log(f"[{symbol}] 手动平仓已发送 qty={quantity}，并撤销 {cancelled} 个挂单。")
                messagebox.showinfo("平仓成功", f"{symbol} 平仓单已发送，已撤销 {cancelled} 个挂单。")
                self.trigger_refresh(include_history=True, force=True)
            elif event == "manual_close_error":
                if self.close_selected_position_button is not None:
                    self.close_selected_position_button.configure(state=tk.NORMAL)
                details = dict(payload)
                symbol = str(details.get("symbol", ""))
                error = str(details.get("error", "Unknown error"))
                self.status_var.set(f"{symbol} 平仓失败: {error}")
                messagebox.showerror("平仓失败", error)

        self.after(250, self._drain_queue)

    def _apply_auto_cycle(self, result: AutoTraderCycleResult) -> None:
        self.auto_last_cycle_var.set(f"最近执行: {format_iso_timestamp(result.cycle_time)}")
        self.auto_status_var.set(f"{result.mode_label} | {result.status}")
        self._populate_tree(
            self.auto_candidates_tree,
            result.candidate_rows,
            ["symbol", "action", "portfolio_rank", "rank_score", "factor_score", "risk_multiplier", "funding_rate", "oi_value_change_pct", "stop_hint"],
        )
        self._populate_tree(
            self.auto_managed_tree,
            result.managed_rows,
            ["symbol", "entry_time", "entry_price", "qty", "mark_price", "unrealized_pnl", "stop_price", "tp_price", "partial_taken", "status"],
            iso_time_columns={"entry_time"},
        )
        for message in result.messages:
            self._append_auto_log(message)
        self.trigger_refresh(include_history=False)

    def _apply_payload(self, payload: dict[str, Any], *, include_history: bool) -> None:
        self.latest_payload = payload if include_history or self.latest_payload is None else {**self.latest_payload, **payload}
        summary = payload.get("summary", {})
        for key, variable in self.summary_vars.items():
            variable.set(format_value(summary.get(key, "--")))
        self._apply_summary_profit_colors(summary)

        self._populate_tree(
            self.positions_tree,
            payload.get("positions", []),
            ["symbol", "side", "qty", "entry_price", "mark_price", "notional", "unrealized_pnl", "roe_pct", "leverage", "liquidation_price", "margin_type"],
        )
        self._populate_tree(
            self.orders_tree,
            payload.get("open_orders", []),
            ["symbol", "side", "type", "status", "orig_qty", "price", "stop_price", "avg_price", "update_time"],
            time_columns={"update_time"},
        )
        if include_history:
            trade_rows = list(payload.get("trades", [])) + self._local_live_trade_rows(200)
            self._populate_tree(
                self.trades_tree,
                trade_rows,
                ["time", "symbol", "side", "position_side", "qty", "price", "realized_pnl", "commission", "commission_asset", "reason"],
                time_columns={"time"},
            )
            self._populate_tree(
                self.income_tree,
                payload.get("income", []),
                ["time", "income_type", "symbol", "income", "asset", "info"],
                time_columns={"time"},
            )

    def _profit_tag_for_row(self, row: dict[str, Any]) -> str:
        for key in ["unrealized_pnl", "realized_pnl", "pnl", "income", "roe_pct", "unrealized_profit"]:
            if key not in row:
                continue
            value = row.get(key)
            if value in ("", None):
                continue
            try:
                number = float(str(value).replace("%", "").replace(",", ""))
            except (TypeError, ValueError):
                continue
            if number > 0:
                return "profit"
            if number < 0:
                return "loss"
            return "flat"
        return ""

    def _profit_color(self, value: Any) -> str:
        try:
            number = float(str(value).replace(",", ""))
        except (TypeError, ValueError):
            return "#1f2933"
        if number > 0:
            return "#087f23"
        if number < 0:
            return "#b42318"
        return "#1f2933"

    def _apply_summary_profit_colors(self, summary: dict[str, Any]) -> None:
        for key in ["total_unrealized_profit", "realized_pnl_24h", "realized_pnl_7d", "net_income_7d"]:
            label = self.summary_value_labels.get(key)
            if label is not None:
                label.configure(foreground=self._profit_color(summary.get(key, 0.0)))

    def _populate_tree(
        self,
        tree: ttk.Treeview | None,
        rows: list[dict[str, Any]],
        columns: list[str],
        *,
        time_columns: set[str] | None = None,
        iso_time_columns: set[str] | None = None,
    ) -> None:
        if tree is None:
            return
        for item in tree.get_children():
            tree.delete(item)
        time_columns = time_columns or set()
        iso_time_columns = iso_time_columns or set()
        for row in rows:
            values: list[str] = []
            for column in columns:
                raw = row.get(column, "")
                if column in time_columns:
                    values.append(format_timestamp_ms(raw))
                elif column in iso_time_columns:
                    values.append(format_iso_timestamp(raw))
                else:
                    values.append(format_value(raw))
            tag = self._profit_tag_for_row(row)
            tree.insert("", tk.END, values=values, tags=(tag,) if tag else ())

    def _on_close(self) -> None:
        if self.auto_trader_stop_event is not None:
            self.auto_trader_stop_event.set()
        self.destroy()


def run_dashboard() -> None:
    app = BinanceDashboardApp()
    app.mainloop()
