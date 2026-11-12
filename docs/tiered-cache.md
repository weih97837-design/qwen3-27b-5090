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

## 閒置時預先備份（VRAM 保留）

引擎完全閒置達 `VRAM_BACKUP_IDLE_S`（預設 60 秒）後，會把 VRAM 中尚未備份的 checkpoint **複製一份到 RAM，原本的
checkpoint 保留在 VRAM**（`engine.spill_backup_one`）。每次只複製一個（大型 checkpoint 約需 0.3–1 秒）；每複製完一個，
都會先檢查是否有新請求，有就先處理請求。沒有可備份的項目後停止，直到下一次活動（請求進出、推理回合）才重新開始。

效果：

- 代理回來時，仍在 VRAM 的 checkpoint 立即命中，與沒有備份時相同。
- 負載上來需要逐出時，被逐出的 checkpoint 已有 RAM 副本，不必在請求等待中複製，逐出沒有複製停頓。
- 備份的 RAM 項目與其他項目相同：計入 `SPILL_RAM_GIB`，保留 `SPILL_RAM_RETENTION_S` 後寫入 SSD，並受 TTL 約束。
- 只在分層啟用（`SPILL_RAM_GIB` 大於 0）時運作。引擎若沒有 `spill_backup_one` 則不執行。`VRAM_BACKUP_IDLE_S=0` 關閉。

### 停止時的寫出（SPILL_FLUSH_ON_STOP_S）

引擎程序停止時（閒置卸載 `ENGINE_IDLE_UNLOAD_S`、重新啟動、或伺服器關閉），若設定了 `SPILL_DIR`，會在
`SPILL_FLUSH_ON_STOP_S` 秒內完成：先備份剩餘的 VRAM checkpoint（最多用一半時間），再把 RAM 中的項目寫入 SSD。
沒有這一步，閒置卸載會連 VRAM 與 RAM 的快取一起丟失。

預算上限為 8 秒。`frontend/server.py` 在卸載或重新啟動時等待引擎程序結束 15 秒、關閉時等待 10 秒，之後會強制終止。
超過的設定值會被截為 8 秒（記錄於日誌）。若要更長的寫出時間，須先調高 `server.py` 的等待時間。

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
| `VRAM_BACKUP_IDLE_S` | `60` | 引擎閒置多久後開始把 VRAM checkpoint 複製到 RAM（保留在 VRAM）。`0` 關閉，見上文 |
| `SPILL_FLUSH_ON_STOP_S` | `8` | 停止時備份與寫出的時間預算（秒），上限 8。`0` 關閉，見上文 |

`KV_DTYPE` 維持 `nvfp4`，每個請求的上限 `MAX_TOKENS` 維持 `262144`。

## 容量估算

- 每個 token 的 NVFP4 KV 約 18 KiB。
- 一個 256K tokens 的代理：KV 約 4.5 GiB + 狀態約 0.19 GiB，合計約 4.7 GiB。
- 32 GiB RAM 約可保存 6 個代理。
- 100 GiB SSD 約可保存 21 個代理。
- 512K 的 KV 池約佔 9 GiB VRAM。

## 加速選項（預設關閉）

目前每次 spill（逐出）與 promote（前綴命中時搬回）都是在運算 stream 上以一般（pageable）主機記憶體複製，複製完才繼續，
因此會暫停所有 GPU 工作與引擎執行緒，速度也只有 PCIe 的一半左右。兩個選項可改善，**預設都關閉**，行為與先前審查過的版本完全相同；
請先在自己的 GPU 上通過 `scripts/tiered_cache_test.py`（輸出 token 必須與參考完全相同）再開啟。

| 變數 | 預設 | 說明 |
| --- | --- | --- |
| `SPILL_PINNED_GIB` | `0` | spill entry 可使用的 pinned（page-locked）主機記憶體上限（GiB）。0 表示關閉 |
| `SPILL_ASYNC_IO` | `0` | `1`：KV 頁在專用的 I/O stream 上搬移，不再暫停運算 stream 與引擎執行緒 |
| `SPILL_PINNED_SLAB_MIB` | `1024` | 每次 pin 的區塊大小（MiB），一般不需調整 |

- **SPILL_PINNED_GIB**：pin 記憶體很慢（WSL2 上約每 GiB 0.45 秒，期間其他 CUDA 呼叫都要等），所以絕不在請求處理中進行：
  伺服器閒置時，前端每輪呼叫一次 `spill_grow_pinned`，每次 pin 一個區塊（slab），直到達到上限（與 state 快取的 `cache_grow` 相同）。
  每個 entry 在一個區塊內取得一段連續空間（first-fit）；放不下時（區塊用完、碎片、或 entry 大於一個區塊）改用一般記憶體，與先前相同。
  `/stats` 的 tiers 區塊有 `pinned_bytes`（已 pin 的總量）與 `pinned_used`（使用中）。
- **SPILL_ASYNC_IO**：複製改在獨立 stream 上進行，以事件（event）與運算 stream 在 GPU 端排序；CPU 只在需要讀寫
  entry 的主機記憶體時才等待（寫入 SSD、丟棄、備份完成）。需要 `SPILL_RAM_GIB` 大於 0。
  閒置備份仍會在該次呼叫結束前等它完成。

**Windows 共享 GPU 記憶體預算。** pinned 記憶體會計入 Windows 的「共享 GPU 記憶體」，上限約為主機 RAM 的一半。
引擎本身已經 pin 約 7 GiB（state 快取等），所以「約 7 GiB ＋ `SPILL_PINNED_GIB`」必須低於主機 RAM 的一半左右
（例如 64 GB 的機器約 32 GiB，`SPILL_PINNED_GIB` 不宜超過 20 左右）。pin 失敗不會讓引擎停止，只會記錄一行並改用一般記憶體。
已 pin 的區塊不論有沒有使用都佔用記憶體；放在其中的 entry 仍計入 `SPILL_RAM_GIB` 上限。兩者都受 WSL VM 的 `memory=` 限制。
entry 必須放進單一區塊，所以超過 `SPILL_PINNED_SLAB_MIB`（預設 1 GiB）的 entry 仍使用一般記憶體；
`spill_stats` 的 `pin_hits` / `pin_misses` 可看到有多少 entry 用到 pinned 記憶體。

比較開啟前後（每個引數是一組以空白分隔的 NAME=VALUE，每組各跑一次）：

```
docker compose --profile dev run --rm -v qwen3-27b-5090_kvcache:/cache/kv dev \
    python scripts/tiered_cache_bench.py --spill-dir /cache/kv/bench \
    --compare "SPILL_PINNED_GIB=0 SPILL_ASYNC_IO=0" "SPILL_PINNED_GIB=8 SPILL_ASYNC_IO=1"
```

## WSL2 注意事項

- **RAM 層計入 WSL VM 的記憶體。** 預設的 WSL2 記憶體上限不足以容納 32 GiB 的 RAM 層。請在 `%UserProfile%\.wslconfig`
  設定 `memory=`，例如在 96 GB 的機器上，扣掉系統保留 16 GB 後設為 `memory=80GB`（建議配置工具會依主機 RAM 算出這個值），之後執行 `wsl --shutdown` 重新啟動。
- **刪除檔案不會縮小 vhdx。** WSL2 的虛擬硬碟在檔案刪除後不會自動縮小，因此 SSD 上限不會把主機磁碟空間還給系統。
- **使用具名磁碟區。** `kvcache` 位於 WSL2 的 ext4 磁碟上。不要改用 `/mnt/c` 的 bind mount，drvfs 讀寫很慢。
- **權限修正前建立的磁碟區要刪除一次。** 較舊的建置建立的 `kvcache` 磁碟區可能屬於 root，容器無法寫入 SSD 層（`q27:tier_failed` 會增加）。請執行一次 `docker compose down`，再 `docker volume rm qwen3-27b-5090_kvcache`（會刪除 SSD 層的全部快取），然後重新啟動。

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

## 真機測試

在 RTX 5090 上以開發容器執行（與 `scripts/prefix_cache_test.py` 相同的方式，需要權重與 `bench/golden/`）：

```
# 精確性：各情境的輸出 token 必須與關閉快取的參考完全相同
docker compose --profile dev run --rm dev python scripts/tiered_cache_test.py
docker compose --profile dev run --rm dev python scripts/tiered_cache_test.py --scenarios a,c

# 把 SSD 層放在 kvcache 具名磁碟區（dev 服務預設沒有掛載，需要用 -v 加上）
docker compose --profile dev run --rm -v qwen3-27b-5090_kvcache:/cache/kv dev \
    python scripts/tiered_cache_test.py --keep-dir /cache/kv/test

# 耗時：32K / 128K / 256K 的 checkpoint
docker compose --profile dev run --rm -v qwen3-27b-5090_kvcache:/cache/kv dev \
    python scripts/tiered_cache_bench.py --spill-dir /cache/kv/bench

# 前後比較（每個引數是一組 NAME=VALUE）
docker compose --profile dev run --rm -v qwen3-27b-5090_kvcache:/cache/kv dev \
    python scripts/tiered_cache_bench.py --spill-dir /cache/kv/bench \
    --compare "SPILL_PINNED_GIB=0" "SPILL_PINNED_GIB=8"
```

`tiered_cache_test.py` 有五個情境，各自在獨立的行程中執行（環境變數在引擎建立時讀取）：

| 情境 | 強制走的路徑 | 檢查的計數器 |
| --- | --- | --- |
| a 從 RAM 還原 | 小的 `KV_POOL_TOKENS`：第二段對話的 prefill 逐出第一段的 checkpoint | `spilled` 增加；後續回合 `restored` 增加 |
| b 從 SSD 還原 | 同 a，再把 RAM 中的項目 `spill_export` 到 `SPILL_DIR` | 後續回合從檔案還原（`restored` 增加，且還原前該項目 `on_disk`） |
| c 重啟後匯入 | 第一個行程寫出檔案後結束；第二個行程 `spill_import` 後續談 | 匯入的檔案數；後續回合 `restored` 增加 |
| d 閒置備份重用 | `spill_backup_one` 直到 False；之後的逐出不再複製 | `backed_up` 增加；逐出時 `spilled` 不變；後續回合 `restored` 增加 |
| e 推測解碼 | 同 a，使用草稿模型（環形 checkpoint）；沒有草稿模型時跳過 | 同 a |

每個情境還會檢查續談是否從上一回合結束的位置開始（`cached` 等於 `pos`）。若情境失敗，先確認 `--pool-tokens`：池太大時不會逐出，
檢查會失敗而不是通過，此時把池調小（預設為一段對話加其 checkpoint 再加 1K tokens）。

**PASS 的意義**：該情境的每一段輸出 token（貪婪解碼，每回合 `--tokens` 個）與關閉快取的參考完全相同，而且所有檢查都通過，
表示確實走了該情境的路徑。任何一個 FAIL 則 exit code 為 1。

`tiered_cache_bench.py` 的計時不含引擎建立與 checkpoint 的 prefill。「逐出並複製」的時間是 prefill 時間減去同一段
prefill（沒有東西可逐出）的時間，是估計值。bench 的提示由 golden 文件重複拼接而成，只用於計時，不檢查輸出。

**新的加速旗標只能在 `tiered_cache_test.py` 全部 PASS 之後才預設開啟。** 例如 `SPILL_PINNED_GIB`、`SPILL_ASYNC_IO`
這類選用旗標，先用 `tiered_cache_bench.py --compare` 量測，再以 `tiered_cache_test.py` 驗證精確性；測試未通過前不要寫進
`compose.yaml` 或 `tools/recommend_config.py` 的預設值。
