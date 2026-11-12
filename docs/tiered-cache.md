# 分層 KV 快取（VRAM → RAM → SSD）

長上下文的代理（agent）對話會佔用大量 KV 快取。GPU 上的 KV 池（`KV_POOL_TOKENS`，預設 512K tokens）裝不下所有對話時，
引擎會把被逐出的 checkpoint 存到主機 RAM 和 SSD，而不是直接丟棄。之後同一段對話再來時，從較深的 checkpoint 還原，
不必重算整段前綴。

## 概要

流程如下：

1. **VRAM**：活躍的 KV 池。池滿時，最久未用的 checkpoint 被逐出。
2. **RAM**：引擎把被逐出的 checkpoint 存進主機 RAM（`SPILL_RAM_GIB`，預設 32 GiB），而不是丟棄。
3. **SSD**：`frontend/tiercache.py` 在閒置時，把存在 RAM 超過 `SPILL_RAM_RETENTION_S`（預設 3600 秒）的項目寫到
   `SPILL_DIR`（預設 `/cache/kv`，即 Docker 具名磁碟區 `kvcache`）。忙碌時則在超過兩倍保留時間後寫出。
   同一個程式也會執行 SSD 容量上限（`SPILL_SSD_GIB`，最舊的檔案先刪）。此上限一律執行，即使磁碟已被其他檔案佔滿。
   可用空間保留（`SPILL_SSD_MIN_FREE_GIB`）只在能靠刪除本層自己的檔案達成時才會追求。
   超過 `SPILL_TTL_DAYS` 天的檔案會被刪除，TTL 自寫入 SSD 起算（重用後會重新計時）。每 `SPILL_TICK_S` 秒執行一次。
4. **刪除**：超過 TTL 或超過容量上限的檔案被刪除。

請求到達時，`prefix_begin` 會找出對話前綴最深的命中點，並從 RAM 或 SSD 把該 checkpoint 還原回 VRAM。引擎端的 RAM
上限是最後一道防線，即使 `frontend/tiercache.py` 沒有執行，RAM 也不會無限制成長。

## 參數表

以下全部是 `compose.yaml` 中 `serve` 服務的環境變數，可在 `.env` 或 shell 中覆寫，於容器啟動時生效。

| 變數 | 預設 | 說明 |
| --- | --- | --- |
| `KV_POOL_TOKENS` | `524288`（512K） | GPU 上的 KV 池大小（tokens）。`0` 表示使用 VRAM 預算剩下的全部空間 |
| `MAX_SEQS` | `4` | 同時執行的請求數上限 |
| `SPILL_RAM_GIB` | `32` | 主機 RAM 中保存被逐出 checkpoint 的容量（GiB）。`0` 表示關閉分層 |
| `SPILL_DIR` | `/cache/kv` | checkpoint 寫入的目錄（容器內），對應具名磁碟區 `kvcache` |
| `SPILL_RAM_RETENTION_S` | `3600` | checkpoint 在 RAM 中保留的秒數，之後才寫入 SSD |
| `SPILL_SSD_GIB` | `100` | SSD 層容量上限（GiB），最舊的檔案先刪除 |
| `SPILL_SSD_MIN_FREE_GIB` | `10` | 磁碟可用空間目標。只在能靠刪除本層自己的檔案達成時才追求，不會刪除其他檔案 |
| `SPILL_TTL_DAYS` | `7` | 自寫入 SSD 起算（重用後會重新計時），超過此天數的檔案會被刪除 |
| `SPILL_TICK_S` | `30` | 分層維護程序的執行間隔（秒） |

`KV_DTYPE` 維持 `nvfp4`，每個請求的上限 `MAX_TOKENS` 維持 `262144`。

## 容量估算

- 每個 token 的 NVFP4 KV 約 18 KiB。
- 一個 256K tokens 的代理：KV 約 4.5 GiB + 狀態約 0.19 GiB，合計約 4.7 GiB。
- 32 GiB RAM 約可保存 6 個代理。
- 100 GiB SSD 約可保存 21 個代理。
- 512K 的 KV 池約佔 9 GiB VRAM。

## WSL2 注意事項

- **RAM 層計入 WSL VM 的記憶體。** 預設的 WSL2 記憶體上限不足以容納 32 GiB 的 RAM 層。請在 `%UserProfile%\.wslconfig`
  設定 `memory=`，例如在 96 GB 的機器上，扣掉系統保留 16 GB 後設為 `memory=80GB`（建議配置工具會依主機 RAM 算出這個值），之後執行 `wsl --shutdown` 重新啟動。
- **刪除檔案不會縮小 vhdx。** WSL2 的虛擬硬碟在檔案刪除後不會自動縮小，因此 SSD 上限不會把主機磁碟空間還給系統。
- **使用具名磁碟區。** `kvcache` 位於 WSL2 的 ext4 磁碟上。不要改用 `/mnt/c` 的 bind mount，drvfs 讀寫很慢。

## 限制

- 每段對話只保留最深的一個 checkpoint。
- 從 RAM 或 SSD 還原 4.5 GiB 的 KV 約需 0.5–3 秒，仍遠快於重算 256K tokens。
- 快取檔案與模型和 KV 型別綁定。若兩者改變，檔案會被丟棄。
- 尚未在實機上測試。

## 建議配置工具

`tools/recommend_config.py` 會依本機的 GPU、RAM 與磁碟，算出上述參數的建議值（只用標準函式庫，Python 3.10+）：

```
python3 tools/recommend_config.py                    # 報告與 .env 區塊
python3 tools/recommend_config.py --spill-dir <路徑>  # 測量 SSD 層所在的磁碟
python3 tools/recommend_config.py --write .env       # 只更新受管理的鍵（先建立 .env.bak）
python3 tools/recommend_config.py --json             # 機器可讀的輸出
```

- 在 WSL 主機執行。容器內可用 `docker compose --profile dev run --rm dev-cpu python tools/recommend_config.py`（serve 映像不含 `tools/`）。
- 具名磁碟區 `kvcache` 位於 WSL 虛擬磁碟。要測量它，請把 `--spill-dir` 指向該磁碟區的主機路徑。`.env` 中的 `SPILL_DIR` 一律寫入容器內路徑。
- 偵測不到的值使用 `compose.yaml` 預設值，並在報告中註明。
- 計算 VRAM 前請先停止正在執行的 serve，否則其用量會被當成「其他程式」。
- 若報告提示 WSL VM 記憶體不足，依提示在 `%UserProfile%\.wslconfig` 設定 `memory=`，再於 Windows 執行 `wsl --shutdown`。
