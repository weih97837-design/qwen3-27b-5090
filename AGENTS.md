# AGENTS.md

Guidance for AI agents working in this repository. / 給在此 repo 工作的 AI agent 的說明。

- Deploy with `docs/deploy-cli.md` (runbook). Windows tray app and /ui settings: `docs/deploy-launcher.md`.
  部署請依 `docs/deploy-cli.md`。Windows 托盤程式與 /ui 設定頁見 `docs/deploy-launcher.md`。
- This branch's `serve` image is not published: it is built locally as `local/qwen3-27b-5090-serve:tiered-kv-cache` (`pull_policy: build`), with `scripts/serve.sh up --build`. Do not `docker compose pull`.
  本分支的 `serve` image 未發佈：本地名稱為 `local/qwen3-27b-5090-serve:tiered-kv-cache`（`pull_policy: build`），用 `scripts/serve.sh up --build` 建置，不要 `docker compose pull`。
- Never commit `.env`, the weights in `hf/`, or `dist/`. Check `git status` before committing; commit only when asked.
  不可提交 `.env`、`hf/` 內的權重或 `dist/`。提交前檢查 `git status`；只在使用者要求時才 commit。
- Config overrides go in `.env` (gitignored). Do not change the defaults in `compose.yaml` to override a setting.
  設定覆寫寫在 `.env`（已被 gitignore）。不要為了覆寫而改 `compose.yaml` 的預設值。
- Tiered KV cache (RAM and SSD spill) settings: `docs/tiered-cache.md`. Its SSD volume `kvcache` holds the spilled checkpoints.
  分層 KV 快取（RAM 與 SSD）的設定見 `docs/tiered-cache.md`。SSD 層是具名磁碟區 `kvcache`。
- Frontend tests: `cd frontend && python -m pytest tests -q`.
  前端測試：`cd frontend && python -m pytest tests -q`。
- Tools tests: `python -m pytest tools/tests -q`.
  工具測試：`python -m pytest tools/tests -q`。
- Do not stop or change a running engine (`docker compose down`, `POST /admin/engine`) without the user's request: it frees VRAM and fails in-flight requests.
  未經使用者要求，不要停止或變更正在執行的引擎（`docker compose down`、`POST /admin/engine`）：會釋放 VRAM 並使進行中的請求失敗。
