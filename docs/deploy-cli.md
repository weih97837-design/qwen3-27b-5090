# 部署 runbook（命令列，給 AI agent 使用）

本文件是確定性的操作手冊：每一步都有**指令**、**成功判準**與**失敗時的處理**。所有指令都取自本 repo，或是標準的 `docker`、`git`、`curl`、`wsl` 指令。
標記 **「未驗證」** 的內容，代表撰寫時無法從 repo 內容確認（例如需要實機 GPU、外部服務或 Windows 才能確認的輸出）。
本文件是依 repo 內容撰寫的，**未在 RTX 5090 實機上逐步執行過**。

> **本分支的注意事項**：`tiered-kv-cache` 分支不發佈 image。`compose.yaml` 的 `serve` 服務 image 為本地名稱 `local/qwen3-27b-5090-serve:tiered-kv-cache`，並設 `pull_policy: build`，由本 checkout 建置。
> 因此**不要**使用 `docker compose pull serve`（沒有可拉的 registry image）。上游 mkyt 的 image（`ghcr.io/mkyt/qwen3-27b-5090`）不含本分支的分層快取，也不要拉。`scripts/serve.sh update` 等同 `up --build`。
> 一律用本地建置：`scripts/serve.sh up --build` 或 `docker compose up -d --build serve`。

執行環境：Windows 上的 WSL2 Linux 發行版（或原生 Linux），指令都在 repo 根目錄執行。

---

## 1. 前置條件檢查

每一項都要通過才往下做。任何一項失敗，停在這裡並回報。

| # | 指令 | 成功判準 | 失敗時 |
| --- | --- | --- | --- |
| 1 | `nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader` | 名稱含 `RTX 5090`，`memory.total` 約 32 GB。驅動支援 CUDA 13（README 測試於驅動 617.42） | 不是 RTX 5090：**停止**。README 明載其他 GPU 不支援（核心使用 SM120 指令） |
| 2 | `docker info --format '{{.ServerVersion}}'` | 印出版本號，沒有錯誤 | Docker 未啟動：在 WSL2 內啟動 Docker（**未驗證**：本 repo 未說明啟動方式） |
| 3 | `docker compose version` | 印出 `Docker Compose version v2.x`（**未驗證** 確切版本） | 安裝 Compose v2 |
| 4 | `docker info --format '{{json .Runtimes}}'` | 輸出含 `"nvidia"` | 未安裝 NVIDIA Container Toolkit，`gpus: all` 會失敗（**未驗證**：輸出格式未在 repo 內確認） |
| 5 | `free -g` | `Mem: total` 至少 16（README：至少 16 GB 主機 RAM） | 增加 WSL2 記憶體，或改用較小的 `SPILL_RAM_GIB`（見 `docs/tiered-cache.md`） |
| 6 | `df -h .` | 可用空間至少 25 GB 供權重（README：權重約 23 GB）；Docker 建置也需要空間 | 清出空間，或把 `./hf` 做成符號連結到別的磁碟（見第 4 節） |
| 7 | `python3 --version` | 3.10 或以上（`scripts/download-weights.sh` 與 `tools/recommend_config.py` 需要；否則腳本會改用 `uv`） | 安裝 Python 3.10+ 或 `uv` |
| 8 | `docker ps --filter "publish=46320" --format '{{.Names}}'` | 沒有輸出，或只有 `qwen3-27b-5090` | 其他容器佔用 46320 埠。只能同時跑一個模型（GPU 只放得下一個），先停掉它（`scripts/serve.sh up` 也會做同樣的檢查） |

---

## 2. 取得程式碼

```bash
git clone --branch tiered-kv-cache https://github.com/weih97837-design/qwen3-27b-5090 qwen3-27b-5090
cd qwen3-27b-5090
git branch --show-current
git log --oneline -1
```

- **成功判準**：`git branch --show-current` 印出 `tiered-kv-cache`；`git log` 顯示最新 commit。
- **目錄已存在時**：不要重新 clone，改用 `cd qwen3-27b-5090 && git fetch origin && git checkout tiered-kv-cache && git pull --ff-only`。
- **失敗時**：`git` 無法連線，檢查網路或 Proxy；分支不存在則停止並回報。

---

## 3. 建議配置

本節是**可選**的；不做的話，`compose.yaml` 的預設值就會生效。

```bash
python3 tools/recommend_config.py            # 繁體中文報告 + .env 區塊（只讀，不寫檔）
python3 tools/recommend_config.py --json     # 同樣內容，機器可讀，輸出到 stdout
```

- **成功判準**：結束碼 0，輸出含建議值（`KV_POOL_TOKENS`、`MAX_TOKENS`、`MAX_SEQS`、`KV_DTYPE`、`SPILL_*` 等）。偵測失敗時會印出註記，並保留 `compose.yaml` 的預設值（工具本身結束碼仍為 0）。
- **審閱**：確認 `MAX_SEQS` 為 4、`KV_DTYPE` 為 `nvfp4`，以及 `KV_POOL_TOKENS` 與 `SPILL_RAM_GIB`、`SPILL_SSD_GIB` 符合你的記憶體與磁碟。
- **WSL 沒有 `%UserProfile%\.wslconfig` 時**：WSL2 預設只會看到約一半的主機 RAM（例如 96 GB 主機約 48 GB，**未驗證**：本 repo 無法在實機確認），因此建議的 `SPILL_RAM_GIB` 會較小。工具在 RAM 不足時會印出 `.wslconfig` 的 `memory=` 提示；依提示設定後執行 `wsl --shutdown`，再重新執行工具取得較大的建議值。
- **寫入 `.env`（只在需要時）**：

  ```bash
  python3 tools/recommend_config.py --write .env
  ```

  - **成功判準**：`.env` 產生或更新，結束碼 0。若 `.env` 已存在，工具會先建立 `.env.bak`。
  - 只會更新工具管理的鍵：`KV_POOL_TOKENS`、`MAX_TOKENS`、`MAX_SEQS`、`KV_DTYPE`、`SPILL_RAM_GIB`、`SPILL_SSD_GIB`、`SPILL_DIR`、`SPILL_RAM_RETENTION_S`、`SPILL_TTL_DAYS`、`SPILL_TICK_S`。
  - **失敗時**：`--write` 失敗時結束碼為 1；檢查路徑與權限。
- **`.env` 不存在時**：`compose.yaml` 的 `${VAR:-預設}` 預設值全部生效（`MAX_SEQS=4`、`KV_POOL_TOKENS=524288`、`SPILL_RAM_GIB=32`、`SPILL_SSD_GIB=100` 等）。這是正常狀態，不是錯誤。
- **SSD 層所在磁碟的量測（第一次啟動後才能做）**：`kvcache` 是 Docker 具名磁碟區，位於 WSL 虛擬磁碟。取得其主機路徑後，用 `--spill-dir` 量測：

  ```bash
  docker volume inspect qwen3-27b-5090_kvcache --format '{{.Mountpoint}}'
  python3 tools/recommend_config.py --spill-dir <上一行輸出的路徑>
  ```

  **未驗證**：讀取 `/var/lib/docker/...` 可能需要 root 權限。
- `.env` 已列在 `.gitignore`，**不可提交**。

---

## 4. 下載權重

```bash
scripts/download-weights.sh
```

- 下載預設的 `gdnfp4` 權重（約 19.5 GB）與 DFlash2 草稿模型（約 3.5 GB）到 `./hf`。Hugging Face 快取資料夾（位於 `./hf/hub/`）：`models--juzbox--Qwen3.8-27B-NVFP4-GDN`（gdnfp4 權重）與 `models--incoai--Qwen3.8-27B-DFlash2`（草稿模型）；每個資料夾內以提交雜湊值分出 `snapshots/<commit>/`。`--quasar` 另外加入 `models--QUASAR-QAT--Qwen3.8-27B-QUASAR-NVFP4` 與 `models--nvidia--Qwen3.8-27B-NVFP4`。
- 其他選項（需要時才加）：`--quasar`；`--repo ORG/NAME[@REVISION]`（另一個 checkpoint，之後設定 `WEIGHTS=ORG/NAME`）。`--dev` 與 `--bf16` 只為測試與參考模型，部署不需要。
- 要把權重放在別的磁碟：先建立符號連結再下載：`ln -s /其他磁碟/路徑 hf`。

- **成功判準**：腳本結束碼 0，`ls hf/hub` 出現 `models--juzbox--Qwen3.8-27B-NVFP4-GDN` 與 `models--incoai--Qwen3.8-27B-DFlash2`。
- **失敗時**：中斷或網路錯誤後，**重新執行同一個指令**即可續傳（每個 LFS 檔案都會做 SHA-256 檢查）。缺少 Python 3.10+ 時腳本會改用 `uv`，若兩者都沒有則安裝其一。`HF_TOKEN` 只影響速率限制（本 repo 的權重不需要授權）。

---

## 5. 建置與啟動

先檢查 Compose 設定是否有效：

```bash
docker compose config --quiet
```

- **成功判準**：無輸出，結束碼 0。
- **失敗時**：依錯誤訊息修正 `.env`（語法錯誤）或確認目前目錄是 repo 根目錄。

建置 image 並啟動（會等到 `/v1/models` 回應，最多 15 分鐘，與 `scripts/serve.sh` 相同）：

```bash
scripts/serve.sh up --build
```

等價的分步做法：

```bash
docker compose up -d --build serve
docker compose logs -f serve      # 等到 "engine ready"（`frontend/server.py:284`）後按 Ctrl+C；伺服器會繼續執行
```

只開伺服器、不載入引擎（不佔 VRAM，之後從 `/ui` 啟動引擎）：

```bash
scripts/serve.sh up --build --idle   # 等同 ENGINE_AUTOSTART=0
```

- **成功判準**：`serve.sh` 印出 `server up: http://127.0.0.1:46320/ui`，結束碼 0；`docker compose ps serve` 顯示 running。
- **建置時間**：README 記載首次約 2 分鐘（**未驗證**：CUDA 編譯時間依機器而異）。
- **失敗時**：`serve.sh` 會印出 `docker compose logs --tail 30 serve` 的內容並以 1 結束。見第 11 節。

---

## 6. 驗證

### 6.1 等待引擎就緒（有上限的重試迴圈）

```bash
deadline=$(( $(date +%s) + 900 ))   # 15 minutes, wall-clock
ready=0
while [ "$(date +%s)" -lt "$deadline" ]; do
  body=$(curl -sf -m 5 http://127.0.0.1:46320/health) && \
    python3 -c 'import json,sys; sys.exit(0 if json.load(sys.stdin).get("status") == "ok" else 1)' <<<"$body" && { ready=1; break; }
  sleep 2
done
if [ "$ready" = 1 ]; then echo READY; else echo "TIMEOUT: docker compose logs --tail 100 serve"; fi
```

- **成功判準**：印出 `READY`。`curl -s http://127.0.0.1:46320/health` 回傳 `{"status": "ok"}`。
- **注意**：`/health` 在引擎未就緒時回傳 503，內容為 `{"status": "<狀態>"}`，狀態為 `stopped`、`loading`、`restarting` 或 `stopping`。只有 `status` 為 `ok` 才代表可以送請求。若 `ENGINE_AUTOSTART=0`（`--idle`），或閒置卸載後，服務會回傳 200 且 `status` 為 `unloaded`（載入中為 `loading`）；此時引擎**尚未載入**，上面的迴圈會逾時，需先在 `/ui` 啟動或送出第一個請求。
- **逾時（15 分鐘）時**：`docker compose logs --tail 100 serve`；必要時看第 11 節。

Compose 內建的健康檢查（`compose.yaml:101-105`，`start_period: 600s`）也可以查：

```bash
docker inspect -f '{{.State.Health.Status}}' qwen3-27b-5090    # starting / healthy / unhealthy
```

### 6.2 模型清單

```bash
curl -s http://127.0.0.1:46320/v1/models | python3 -m json.tool
```

- **成功判準**：`"object": "list"`，`data[0].id` 為 `qwen3.8-27b`（`SERVED_MODEL_NAME` 的預設值，`compose.yaml:60`），`data[0].max_model_len` 為實際生效的 `MAX_TOKENS`。

### 6.3 一次簡短的聊天補全

```bash
curl -s http://127.0.0.1:46320/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "qwen3.8-27b", "messages": [{"role": "user", "content": "Say hi in one word."}], "max_tokens": 32, "chat_template_kwargs": {"enable_thinking": false}}' \
  | python3 -m json.tool
```

- **成功判準**（`scripts/smoke_test.py` 也用 `max_tokens` 發送請求）：
  - `"object": "chat.completion"`
  - `choices[0].message.content` 非空
  - `choices[0].finish_reason` 為 `stop`（若為 `length` 代表 32 個 token 用完，仍視為成功，請求有被處理）
  - `usage.prompt_tokens`、`usage.completion_tokens` 皆有值
- **失敗時**：HTTP 4xx 代表請求格式錯誤（看錯誤 JSON）；HTTP 503 代表引擎未就緒或已停止，回到 6.1。
- 更完整的測試：`python3 scripts/smoke_test.py`、`python3 scripts/api_test.py`（README 所述，標準函式庫即可執行，**未在本次撰寫時執行**）。

### 6.4 確認分層快取已啟用

```bash
curl -s http://127.0.0.1:46320/metrics | grep -E '^q27:tier_'
```

- **成功判準**：出現 `q27:tier_ram_bytes`、`q27:tier_in_ram`、`q27:tier_on_disk`、`q27:tier_spilled`、`q27:tier_restored`、`q27:tier_written`、`q27:tier_dropped`、`q27:tier_subsumed`、`q27:tier_failed`、`q27:tier_ssd_bytes`、`q27:tier_ssd_budget_bytes` 等行。
- **沒有輸出時**：`frontend/worker.py:821` 只在分層啟用時才輸出這些指標。依序檢查：
  1. 設定是否啟用：`docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' qwen3-27b-5090 | grep -E '^(SPILL_RAM_GIB|SPILL_DIR|SPILL_SSD_GIB)='`。`SPILL_RAM_GIB=0` 代表分層關閉（`docs/tiered-cache.md`）。
  2. 容器是否為本分支建置：見第 11 節「image 版本不對」。
- **分層是否真的在運作**：送幾個請求後（最好有一個長到會把 checkpoint 逐出上下文池的提示），檢查：
  ```bash
  curl -s http://127.0.0.1:46320/metrics | grep -E '^q27:tier_(written|failed) '
  ```
  - `q27:tier_failed` 應維持 `0`。若增加，表示寫入 SSD 層失敗，請確認 `SPILL_DIR`（容器內 `/cache/kv`）可寫入。舊版建置建立的 `kvcache` 磁碟區可能屬於 root，修正方法：`docker compose down`，`docker volume rm qwen3-27b-5090_kvcache`（磁碟區名稱是 compose 專案名稱 `qwen3-27b-5090`（見 `compose.yaml` 第 1 行 `name:`）加上 `_kvcache`），再重新啟動。**這會刪除 SSD 層的全部快取**（見第 10 節）。
  - `q27:tier_written` 大於 0，只會在 checkpoint 在 RAM 中停留超過 `SPILL_RAM_RETENTION_S`（預設 3600 秒，`compose.yaml`）且引擎閒置之後才出現。剛發生逐出時它仍是 0，這不代表分層故障。
- **注意**：distroless 基底沒有 shell，`docker compose exec serve sh` 無法使用（`Dockerfile` 的最終階段）。

---

## 7. 狀態網頁

- 網址：**http://127.0.0.1:46320/ui**
- 檢查（回傳 200）：

  ```bash
  curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:46320/ui
  ```

- **成功判準**：輸出 `200`；瀏覽器可以看到引擎狀態、VRAM、分層快取與請求記錄。
- 頁面上的 **Start**、**Stop**、**Restart** 按鈕控制引擎；「Start settings」對話框的按鈕為 **Reset**、**Save for the next load**、**Start**、**Close**（`frontend/ui/index.html`）。說明見 `docs/configuration.md#start-settings`。

---

## 8. 連接 OpenAI 相容用戶端

- **Base URL**：`http://127.0.0.1:46320/v1`，API key 可填任意值。
- **模型名稱**：用下列指令查詢 `/v1/models` 回傳的名稱（預設 `qwen3.8-27b`）。
- **並行**：用戶端同時送出的請求數最多為 `MAX_SEQS`（本 runbook 為 4）時可以並行執行；超過的請求排隊等候。

```bash
curl -s http://127.0.0.1:46320/v1/models | python3 -m json.tool
```

- **成功判準**：`data` 中有模型名稱，且用該名稱送出的請求得到回應。

---

## 9. 更新

```bash
git pull --ff-only
scripts/serve.sh up --build
```

- 不要使用 `docker compose pull serve`（見文件開頭的注意事項）。`scripts/serve.sh update` 只是 `up --build`，可以用。
- `docker compose up -d serve` 或 `docker compose restart serve` **不會**套用新程式碼，必須 `--build`。
- **會發生什麼**：容器重建會中斷進行中的請求，並重新載入引擎（前綴快取會清空）。`./hf` 的權重不會重新下載。`kvcache` 具名磁碟區會保留，所以 SSD 層的 checkpoint 在重建後仍在。
- **成功判準**：重跑第 6 節的驗證全部通過。

---

## 10. 停止與清除

**停止引擎（保留伺服器，釋放 VRAM）**：在 `/ui` 按 **Stop**，或：

```bash
curl -s -X POST http://127.0.0.1:46320/admin/engine -H 'Content-Type: application/json' -d '{"action": "stop"}'
```

- **成功判準**：`/health` 不再回傳 200（引擎已停止），`nvidia-smi` 的 VRAM 使用量下降。

**停止伺服器（保留權重與 SSD 層）**：

```bash
scripts/serve.sh down        # 等同 docker compose down
```

- **成功判準**：`docker compose ps serve` 沒有執行中的容器。`./hf` 與 `kvcache` 磁碟區保留。

**清除 SSD 層（會刪除 SSD 上的快取，不可復原）**：

> **警告**：`kvcache` 具名磁碟區就是 SSD 層（`SPILL_DIR=/cache/kv`）。刪除後，所有被分層保存的 checkpoint 會消失。**權重不受影響**：`./hf` 是主機上的綁定掛載，不在磁碟區內。

```bash
docker compose down
docker volume ls --filter name=kvcache              # 應該看到 qwen3-27b-5090_kvcache
docker volume rm qwen3-27b-5090_kvcache
```

等價的寫法是 `docker compose down -v`，它會刪除本專案所有具名磁碟區（目前只有 `kvcache`）。

**清除舊的 image（會影響其他專案）**：

```bash
docker image prune                                  # 會先詢問；會刪除所有未使用的 dangling image，包括其他專案的
```

README 也明確提醒這一點。

---

## 11. 疑難排解

| 症狀 | 診斷指令 | 處理 |
| --- | --- | --- |
| `docker compose` 找不到 `compose.yaml` 或設定錯誤 | `docker compose config --quiet` | 確認在 repo 根目錄；檢查 `.env` 語法 |
| 建置失敗 | `docker compose build serve`（看完整輸出） | 檢查磁碟空間（第 1 節第 6 項）與網路；**未驗證**：本 repo 未記載常見建置錯誤 |
| 容器內看不到 GPU | `docker info --format '{{json .Runtimes}}'`；`nvidia-smi`（主機） | 安裝或修復 NVIDIA Container Toolkit（**未驗證** 確切步驟） |
| 46320 埠被佔用 | `docker ps --filter "publish=46320"` | `serve.sh` 會列出佔用者；先停止它。只能同時執行一個模型 |
| `/health` 長時間 503 | `curl -s http://127.0.0.1:46320/health`；`docker compose logs --tail 100 serve` | 看狀態字串與日誌中的錯誤。`restart: on-failure:3`（`compose.yaml:108`）會自動重啟最多 3 次 |
| 引擎拒絕載入（VRAM 不足） | `curl -s http://127.0.0.1:46320/admin/engine`（看 `refused` 欄位）；`nvidia-smi` | 關閉其他 GPU 程式；降低 `MAX_TOKENS`，或調低 `VRAM_RESERVE_GIB`（保留的 VRAM 越少，引擎可用越多，但接近 WDDM 把 GPU 記憶體換到系統 RAM 的臨界點）；`python3 tools/recommend_config.py` 取得建議 |
| 權重缺失或載入失敗 | `ls -la hf/hub` | 重跑 `scripts/download-weights.sh`（會續傳） |
| `/metrics` 沒有 `q27:tier_*` | 第 6.4 節的 `docker inspect` 指令 | `SPILL_RAM_GIB` 為 0 時分層關閉；確認 image 為本分支建置 |
| SSD 層佔滿磁碟 | `docker volume inspect qwen3-27b-5090_kvcache --format '{{.Mountpoint}}'`；`curl -s .../metrics \| grep tier_ssd` | 引擎會自動刪除最舊的檔案（`SPILL_SSD_GIB` 上限、`SPILL_SSD_MIN_FREE_GIB` 保留）。要調整就改 `.env` 後 `scripts/serve.sh up` |
| image 不是本分支本地建置的（例如拉到上游 image） | `docker image inspect local/qwen3-27b-5090-serve:tiered-kv-cache --format '{{index .Config.Labels "org.opencontainers.image.revision"}}'` | 本地建置的 image 此欄位為空（README）。有值代表它不是本地建置：執行 `scripts/serve.sh up --build` |
| 同時只有一個請求在跑 | `curl -s http://127.0.0.1:46320/admin/concurrency`（看 `target_seqs`） | 確認用戶端確實同時送出多個請求，且 `MAX_SEQS=4`（`target_seqs` 不會超過它） |
| 生效的設定與預期不同 | `docker compose config`（解析後的 compose 設定）；`docker inspect ...`（容器實際環境） | `.env` 的值優先於 `compose.yaml` 預設值；shell 變數優先於 `.env`。修改後要 `up` 才會套用（`docker compose restart` 不會） |
