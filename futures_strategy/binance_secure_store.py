from __future__ import annotations

import base64
import ctypes
import json
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


APP_DIR_NAME = "BinanceFuturesDashboard"
SETTINGS_FILE_NAME = "settings.json"
LEGACY_AUTO_TRADER_CONFIG_SUFFIX = str(
    Path("outputs") / "funding_oi_short_optimizer_profit_2026-03-18_universe10" / "symbol_configs"
)
STRICT_AUTO_TRADER_CONFIG_SUFFIX = str(Path("outputs") / "funding_oi_short_strict_liquid_top30" / "symbol_configs")
SURVIVOR_AUTO_TRADER_CONFIG_SUFFIX = str(
    Path("outputs") / "universal_short_screen_2024-03__2026-02" / "auto_whitelist_11_configs"
)
SURVIVOR_COMBOS_CONFIG_SUFFIX = str(
    Path("outputs") / "universal_short_screen_2024-03__2026-02" / "survivor_configs"
)


def default_workspace_root() -> Path:
    if getattr(sys, "frozen", False):
        exe_dir = Path(sys.executable).resolve().parent
        candidate = exe_dir.parent
        if (candidate / "outputs").exists():
            return candidate
        return exe_dir
    return Path(__file__).resolve().parents[1]


def default_auto_trader_config_dir() -> str:
    return str(default_workspace_root() / SURVIVOR_AUTO_TRADER_CONFIG_SUFFIX)


def _normalized_path_suffix(raw_path: str | Path) -> str:
    return str(raw_path).replace("/", "\\").lower()


def normalize_auto_trader_config_dir(raw_path: str | None) -> str:
    if not raw_path:
        return default_auto_trader_config_dir()
    resolved = str(raw_path)
    default_path = default_auto_trader_config_dir()
    normalized = _normalized_path_suffix(resolved)
    if normalized.endswith(_normalized_path_suffix(LEGACY_AUTO_TRADER_CONFIG_SUFFIX)):
        return default_path
    if normalized.endswith(_normalized_path_suffix(STRICT_AUTO_TRADER_CONFIG_SUFFIX)):
        return default_path
    if normalized.endswith(_normalized_path_suffix(SURVIVOR_COMBOS_CONFIG_SUFFIX)):
        return default_path
    return resolved


def default_factor_cache_dir() -> str:
    return str(default_workspace_root() / "data" / "binance_factor_bundle")


@dataclass
class DashboardSettings:
    api_key: str = ""
    api_secret: str = ""
    tracked_symbols: list[str] = field(default_factory=lambda: ["AVAXUSDT", "SUIUSDT", "SOLUSDT"])
    fast_refresh_sec: int = 5
    slow_refresh_sec: int = 30
    auto_refresh: bool = True
    use_testnet: bool = True
    auto_trader_config_dir: str = field(default_factory=default_auto_trader_config_dir)
    auto_trader_configs: list[str] = field(default_factory=list)
    auto_trader_cache_dir: str = field(default_factory=default_factor_cache_dir)
    auto_trader_poll_sec: int = 60
    auto_trader_days_back: int = 27
    auto_trader_top_n: int = 1
    auto_trader_max_positions: int = 1
    auto_trader_leverage: int = 20
    auto_trader_margin_type: str = "ISOLATED"
    auto_trader_risk_per_trade: float = 0.03
    auto_trader_working_type: str = "MARK_PRICE"

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "DashboardSettings":
        payload = payload or {}
        tracked = payload.get("tracked_symbols") or []
        if isinstance(tracked, str):
            tracked = [item.strip().upper() for item in tracked.split(",") if item.strip()]
        auto_configs = payload.get("auto_trader_configs") or []
        if isinstance(auto_configs, str):
            auto_configs = [item.strip() for item in auto_configs.split(",") if item.strip()]
        return cls(
            api_key=str(payload.get("api_key", "")),
            api_secret=str(payload.get("api_secret", "")),
            tracked_symbols=[str(item).upper() for item in tracked] or ["AVAXUSDT", "SUIUSDT", "SOLUSDT"],
            fast_refresh_sec=max(3, int(payload.get("fast_refresh_sec", 5))),
            slow_refresh_sec=max(10, int(payload.get("slow_refresh_sec", 30))),
            auto_refresh=bool(payload.get("auto_refresh", True)),
            use_testnet=bool(payload.get("use_testnet", True)),
            auto_trader_config_dir=normalize_auto_trader_config_dir(str(payload.get("auto_trader_config_dir") or "")),
            auto_trader_configs=[str(item) for item in auto_configs],
            auto_trader_cache_dir=str(payload.get("auto_trader_cache_dir") or default_factor_cache_dir()),
            auto_trader_poll_sec=max(15, int(payload.get("auto_trader_poll_sec", 60))),
            auto_trader_days_back=max(3, min(27, int(payload.get("auto_trader_days_back", 27)))),
            auto_trader_top_n=max(1, int(payload.get("auto_trader_top_n", 1))),
            auto_trader_max_positions=max(1, int(payload.get("auto_trader_max_positions", 1))),
            auto_trader_leverage=max(1, int(payload.get("auto_trader_leverage", 20))),
            auto_trader_margin_type=str(payload.get("auto_trader_margin_type", "ISOLATED")).upper(),
            auto_trader_risk_per_trade=max(0.001, float(payload.get("auto_trader_risk_per_trade", 0.03))),
            auto_trader_working_type=str(payload.get("auto_trader_working_type", "MARK_PRICE")).upper(),
        )


def app_settings_dir() -> Path:
    root = Path(os.getenv("APPDATA", Path.home()))
    path = root / APP_DIR_NAME
    path.mkdir(parents=True, exist_ok=True)
    return path


def settings_file_path() -> Path:
    return app_settings_dir() / SETTINGS_FILE_NAME


def _fallback_encrypt(value: str) -> str:
    return base64.b64encode(value.encode("utf-8")).decode("ascii")


def _fallback_decrypt(value: str) -> str:
    return base64.b64decode(value.encode("ascii")).decode("utf-8")


if os.name == "nt":
    CRYPTPROTECT_UI_FORBIDDEN = 0x01

    class DATA_BLOB(ctypes.Structure):
        _fields_ = [
            ("cbData", ctypes.c_uint),
            ("pbData", ctypes.POINTER(ctypes.c_char)),
        ]


    crypt32 = ctypes.windll.crypt32
    kernel32 = ctypes.windll.kernel32


    def _blob_from_bytes(data: bytes) -> DATA_BLOB:
        buffer = ctypes.create_string_buffer(data)
        return DATA_BLOB(len(data), ctypes.cast(buffer, ctypes.POINTER(ctypes.c_char)))


    def _encrypt_value(value: str) -> str:
        raw = value.encode("utf-8")
        in_blob = _blob_from_bytes(raw)
        out_blob = DATA_BLOB()
        if not crypt32.CryptProtectData(
            ctypes.byref(in_blob),
            "binance_dashboard",
            None,
            None,
            None,
            CRYPTPROTECT_UI_FORBIDDEN,
            ctypes.byref(out_blob),
        ):
            raise ctypes.WinError()
        try:
            encrypted = ctypes.string_at(out_blob.pbData, out_blob.cbData)
            return base64.b64encode(encrypted).decode("ascii")
        finally:
            if out_blob.pbData:
                kernel32.LocalFree(out_blob.pbData)


    def _decrypt_value(value: str) -> str:
        encrypted = base64.b64decode(value.encode("ascii"))
        in_blob = _blob_from_bytes(encrypted)
        out_blob = DATA_BLOB()
        if not crypt32.CryptUnprotectData(
            ctypes.byref(in_blob),
            None,
            None,
            None,
            None,
            CRYPTPROTECT_UI_FORBIDDEN,
            ctypes.byref(out_blob),
        ):
            raise ctypes.WinError()
        try:
            decrypted = ctypes.string_at(out_blob.pbData, out_blob.cbData)
            return decrypted.decode("utf-8")
        finally:
            if out_blob.pbData:
                kernel32.LocalFree(out_blob.pbData)
else:
    _encrypt_value = _fallback_encrypt
    _decrypt_value = _fallback_decrypt


def load_dashboard_settings() -> DashboardSettings:
    path = settings_file_path()
    if not path.exists():
        return DashboardSettings()
    payload = json.loads(path.read_text(encoding="utf-8"))
    decrypted = dict(payload)
    for field_name in ("api_key", "api_secret"):
        raw = payload.get(field_name)
        if raw:
            decrypted[field_name] = _decrypt_value(str(raw))
    return DashboardSettings.from_dict(decrypted)


def save_dashboard_settings(settings: DashboardSettings) -> Path:
    path = settings_file_path()
    payload = asdict(settings)
    encrypted = dict(payload)
    encrypted["api_key"] = _encrypt_value(settings.api_key) if settings.api_key else ""
    encrypted["api_secret"] = _encrypt_value(settings.api_secret) if settings.api_secret else ""
    path.write_text(json.dumps(encrypted, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def clear_dashboard_settings() -> Path:
    settings = DashboardSettings()
    return save_dashboard_settings(settings)
