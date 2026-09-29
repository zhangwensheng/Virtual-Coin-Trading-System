# 开源发布清单

这个项目已按“源码可公开、私密数据默认不公开”的方式准备 Git 规则。

## 已默认排除

- `参考/`：历史参考脚本里包含硬编码交易所 `api_key/api_secret` 样式内容。
- `outputs/`：原始回测、模拟盘、实盘日志和下载审计，可能包含本地路径、运行状态或公共下载失败 URL。
- `data/`：原始行情数据体量较大，且不适合直接塞进源码仓库。
- `build/`、`dist/`、`*.spec`：本机构建产物。
- `.env*`、密钥、凭据、缓存文件。

## 可以公开的回测结果

公开版回测结果放在：

- `docs/backtest-results/README.md`
- `docs/backtest-results/summary.csv`

这些文件只汇总 `summary.json` 中的核心指标，不包含原始交易明细、绝对本地路径、下载失败 URL 或账户配置。

## 发布前再检查

```bash
git status --short
git ls-files
rg -n -i "(api[_-]?key|api[_-]?secret|secret[_-]?key|password|token|x-mbx-apikey|ref=|invite)" .
rg -n -i "https?://.*binance|wss?://.*binance" .
```

说明：源码里如果保留交易所公共 API 域名，是程序运行所需的公共接口；不要放你的返佣链接、邀请链接、账户链接、密钥或私有配置。

## 如果密钥曾经是真实可用的

即使它们没有被 Git 提交，也建议到交易所后台立刻删除或轮换历史密钥。历史脚本里出现过明文密钥样式内容，开源前务必把这类目录排除或彻底清理。
