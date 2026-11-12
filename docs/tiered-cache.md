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
- 預設權重為 `nvidia`（NVIDIA 的 checkpoint，GDN / attention 投影為 FP8）。不封頂的 KV 池比 `gdnfp4` 少約 55K tokens（gdnfp4 約 610K），預設的 512K 池仍然放得下。`WEIGHTS=gdnfp4` 時，上面的數字不變。

## 加速選項（預設關閉）

目前每次 spill（逐出）與 promote（前綴命中時搬回）都是在運算 stream 上以一般（pageable）主機記憶體複製，複製完才繼續，
因此會暫停所有 GPU 工作與引擎執行緒，速度也只有 PCIe 的一半左右。三個選項可改善，**預設都關閉**，行為與先前審查過的版本完全相同；
請先在自己的 GPU 上通過 `scripts/tiered_cache_test.py`（輸出 token 必須與參考完全相同）再開啟。

| 變數 | 預設 | 說明 |
| --- | --- | --- |
| `SPILL_PINNED_GIB` | `0` | spill entry 可使用的 pinned（page-locked）主機記憶體上限（GiB）。0 表示關閉 |
| `SPILL_ASYNC_IO` | `0` | `1`：KV 頁在專用的 I/O stream 上搬移，不再暫停運算 stream 與引擎執行緒 |
| `SPILL_ASYNC_PROMOTE` | `0` | `1`：命中已 spill 的 checkpoint 時，請求進入「載入快取」狀態，由引擎在背景載入，其他請求照常執行（delay hit） |
| `SPILL_PINNED_SLAB_MIB` | `1024` | 每次 pin 的區塊大小（MiB）：閒置時每輪只 pin 一個區塊，所以也是單次停頓的上限；一般不需調整 |

- **SPILL_PINNED_GIB**：pin 記憶體很慢（WSL2 上約每 GiB 0.45 秒，期間其他 CUDA 呼叫都要等），所以絕不在請求處理中進行：
  伺服器閒置時，前端每輪呼叫一次 `spill_grow_pinned`，每次 pin 一個區塊（slab），直到達到上限（與 state 快取的 `cache_grow` 相同）。
  **entry 可以分成多段（segment）**：每段位於單一區塊內、長度是 KV 頁大小的整數倍（所以沒有頁會跨段），一個 entry 可跨多個區塊，
  因此 128K–256K tokens 的大型 checkpoint（2–4.5 GiB）也能用 pinned 記憶體，不再受單一區塊大小限制。
  分配時先找能容納整個 entry 的連續空間（first-fit），找不到才依序取最大的空閒區段拼起來；
  **整個 entry 要嘛全部 pinned、要嘛全部一般記憶體**（pool 湊不齊就釋放已取的部分，改用一整塊一般記憶體，不會混用）。
  檔案格式不變（各段依序寫出就是同一條線性位元組流）。
  `/stats` 的 tiers 區塊有 `pinned_bytes`（已 pin 的總量）與 `pinned_used`（使用中）。
- **SPILL_ASYNC_IO**：複製改在獨立 stream 上進行，以事件（event）與運算 stream 在 GPU 端排序；CPU 只在需要讀寫
  entry 的主機記憶體時才等待（寫入 SSD、丟棄、備份完成）。需要 `SPILL_RAM_GIB` 大於 0。
  閒置備份仍會在該次呼叫結束前等它完成。**只有 entry 是 pinned 時，非同步才真的讓 CPU 不必等待**：
  對一般（pageable）記憶體的 `cudaMemcpyAsync` 在 CPU 端仍是同步的（驅動程式會先經由暫存區複製），
  所以單獨開 `SPILL_ASYNC_IO` 只改變 GPU 端的排序、CPU 仍會被卡住。**建議兩個選項一起開**（`SPILL_PINNED_GIB` 與 `SPILL_ASYNC_IO=1`）。

- **SPILL_ASYNC_PROMOTE（delay hit 排程，參考 Strata, OSDI'26：排程時不要為快取載入而停擺）**：原本的 promote 在 `prefix_begin`
  裡同步完成：RAM 的項目要等 H2D 複製，SSD 的項目還要在引擎執行緒上 `fread` 最多約 4.5 GiB，整個引擎停頓數秒。開啟後：
  1. 排程器在放行等待中的請求之前呼叫 `spill_match`（唯讀）找出比 KV 池自己的命中更深的 spill entry。
  2. 值得載入的話呼叫 `spill_restore_begin`：RAM 的項目立刻配好頁並把 H2D 排在 I/O stream 上（回傳 1）；
     只在 SSD 的項目由引擎執行緒配好緩衝區（可 pinned），再由一條背景執行緒**只做** `fread`（回傳 2；該執行緒不碰任何引擎狀態與 CUDA）。
  3. 請求進入 `restoring` 狀態（UI：「載入快取」，Slot 標題顯示「載入中 N」），**仍保留排隊順序，但被跳過**，後面的請求照常放行
     （這就是 delay hit：等的是快取，不是 GPU）。每個排程迴圈與解碼步驟之間都會 `spill_restore_poll`（不阻塞）：
     檔案讀完就接著配頁並排入 H2D；H2D 的 event 完成後，載入中的 checkpoint 才變成可見（`loading=false`），
     entry 才被釋放，請求再度可被放行，它的 `prefix_begin` 就命中這個 checkpoint。
  4. **載入中的 checkpoint 不可見也不可逐出**：不參與 `prefix_begin` 的查找、`spill_lookup`、重複檢查、閒置備份與
     `cache_stats` 的「可逐出頁數」（這些頁在載入完成前不能算作新請求的空間）；`reserve_area` 搬移 parking 區的頁之前，
     運算 stream 會先等它的 event。載入中的 entry 為 busy：不會被寫出、丟棄、TTL 刪除或再次 promote。
  4b. **載入完成後的 checkpoint 是 `held`（保留給等它的請求）**：`evict_checkpoint` 先逐出其他 checkpoint，其次才是最舊的 held，
     若只剩載入中的，則同步等待最舊那個載入完成（`cudaEventSynchronize`，少見的短暫停頓）再逐出，而不是回報 `KvFull`。
     `cache_stats` 與 `spill_room` 不把 held 的頁算作空間。解除 held 的時機：該請求的 `prefix_begin` 命中它、worker 呼叫
     `spill_release`（逾時、取消、結束、`begin` 之後），或只剩 held／載入中時的最舊者後備。**不設時間上限自動解除**：所有放棄等待的路徑
     都會呼叫 release，最舊 held 後備則限制了遺漏的損害。`spill_restore_poll` 回傳 `(id, ok)`，失敗只算在該 id 上。
     worker 在啟動載入前檢查 KV 池放得下（與 `kv_room` 同一套保留頁邏輯），放不下就不啟動；`KvFull` 時先 poll 一次載入並重試，才搶佔其他請求。
  5. 逾時（30 秒）、讀檔失敗、或載入完成時 KV 池已沒有空間，請求都不再等：照原本的路徑繼續（`prefix_begin` 的同步 promote 仍在，作為後備）。

  **載入還是重算（load-vs-recompute）。** 只有在 `估計載入時間 < 0.8 × 估計重算時間` 時才載入；否則放行請求讓 prefill 重算，
  並關閉這一次 `prefix_begin` 的同步 promote（否則引擎仍會照樣停頓去載入）。
  估計載入時間 ＝ entry 大小（KV＋state）÷ 載入速度 ＋ 0.05 秒；載入速度先用預設值（RAM pinned 12 GB/s、pageable 6 GB/s、SSD 2 GB/s），
  有完成的載入之後改用實測值（以預設值的 0.25–4 倍為界，避免一次偏慢的量測讓之後都不載入）。
  估計重算時間 ＝ 命中的 token 數 ÷ 實測的 prefill 速度（排程器每個 prefill chunk 量測；尚無量測時用 8000 tok/s，介於
  architecture.md 量到的 19K 11,295 tok/s 與 194K 5,875 tok/s 之間）。門檻都是 `frontend/worker.py` 開頭的常數
  （`RESTORE_MARGIN`、`RESTORE_TIMEOUT_S`、`RESTORE_BW_*`、`PREFILL_TPS_DEFAULT`）。
  `/stats` 的 tiers 區塊有 `restoring`（載入中）、`restored_async`、`restore_failed` 與實測的 `restore_gbps`。
  **沒有 `SPILL_ASYNC_IO=1` 時，複製本身仍是同步的**（SSD 的讀檔仍在背景，但 H2D 會停住運算 stream），建議三個選項一起開。

**Windows 共享 GPU 記憶體預算。** pinned 記憶體會計入 Windows 的「共享 GPU 記憶體」，上限約為主機 RAM 的一半。
引擎本身已經 pin 約 7 GiB（state 快取等），所以「約 7 GiB ＋ `SPILL_PINNED_GIB`」必須低於主機 RAM 的一半左右
（例如 64 GB 的機器約 32 GiB，`SPILL_PINNED_GIB` 不宜超過 20 左右）。pin 失敗不會讓引擎停止，只會記錄一行並改用一般記憶體。
**預算。** pinned pool 是 RAM 層預算的一部分，不是另外加上去的：放在 pinned 區段的 entry 照常計入 `SPILL_RAM_GIB` 上限（`spill_ram_`），
所以 `SPILL_PINNED_GIB` 會在啟動時夾在 `SPILL_RAM_GIB` 減去一個 checkpoint 的空間（至少 0.25 GiB）以下，並記錄一行。
整個 pinned pool 計入 `SPILL_RAM_GIB`（即使尚未使用，也是保留的主機記憶體，`spill_host_use`）；pinned 上限因此必須低於 RAM 上限；
因此實際主機用量約為「一般記憶體的 entry ＋ 整個 pinned pool」，pool 滿了才會有 entry 退回一般記憶體，通常不會超出 `SPILL_RAM_GIB` 很多。
兩者都受 WSL VM 的 `memory=` 限制。`spill_stats` 的 `pin_hits` / `pin_misses` 可看到有多少 entry 用到 pinned 記憶體
（`pin_misses`：pool 尚未長大、已滿或太碎，放不下整個 entry）。

比較開啟前後（每個引數是一組以空白分隔的 NAME=VALUE，每組各跑一次）：

```
docker compose --profile dev run --rm -v qwen3-27b-5090_kvcache:/cache/kv dev \
    python scripts/tiered_cache_bench.py --spill-dir /cache/kv/bench \
    --compare "SPILL_PINNED_GIB=0 SPILL_ASYNC_IO=0" "SPILL_PINNED_GIB=8 SPILL_ASYNC_IO=1" \
    "SPILL_PINNED_GIB=8 SPILL_PINNED_SLAB_MIB=256 SPILL_ASYNC_IO=1"
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

在 RTX 5090 上以開發容器執行（與 `scripts/prefix_cache_test.py` 相同的方式，需要權重與 `bench/golden/`）。原生 Windows（程式包，無 Docker）改執行 `test.cmd` / `bench.cmd`，見 `docs/deploy-windows.md`。

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

`tiered_cache_test.py` 有七個情境（a–g），各自在獨立的行程中執行（環境變數在引擎建立時讀取）：

| 情境 | 強制走的路徑 | 檢查的計數器 |
| --- | --- | --- |
| a 從 RAM 還原 | 小的 `KV_POOL_TOKENS`：第二段對話的 prefill 逐出第一段的 checkpoint | `spilled` 增加；後續回合 `restored` 增加 |
| b 從 SSD 還原 | 同 a，再把 RAM 中的項目 `spill_export` 到 `SPILL_DIR` | 後續回合從檔案還原（`restored` 增加，且還原前該項目 `on_disk`） |
| c 重啟後匯入 | 第一個行程寫出檔案後結束；第二個行程 `spill_import` 後續談 | 匯入的檔案數；後續回合 `restored` 增加 |
| d 閒置備份重用 | `spill_backup_one` 直到 False；之後的逐出不再複製 | `backed_up` 增加；逐出時 `spilled` 不變；後續回合 `restored` 增加 |
| e 推測解碼 | 同 a，使用草稿模型（環形 checkpoint）；沒有草稿模型時跳過 | 同 a |
| f pinned＋async I/O | 同 a、b，`SPILL_PINNED_GIB=2`、`SPILL_ASYNC_IO=1`（entry 分成多個 pinned 區段） | `pin_hits` 增加 |
| g 背景載入 | 同 a、b，再加 `SPILL_ASYNC_PROMOTE=1`，直接呼叫引擎 API：`spill_match` → `spill_restore_begin`（RAM 回 1、檔案回 2）→ 載入中的 checkpoint 對 `prefix_begin` 不可見 → `spill_restore_poll` 直到回報該 id | `restored_async` 增加、`restore_failed` 不變；續談從上一回合結束處開始 |

每個情境還會檢查續談是否從上一回合結束的位置開始（`cached` 等於 `pos`）。若情境失敗，先確認 `--pool-tokens`：池太大時不會逐出，
檢查會失敗而不是通過，此時把池調小（預設為一段對話加其 checkpoint 再加 1K tokens）。

**PASS 的意義**：該情境的每一段輸出 token（貪婪解碼，每回合 `--tokens` 個）與關閉快取的參考完全相同，而且所有檢查都通過，
表示確實走了該情境的路徑。任何一個 FAIL 則 exit code 為 1。

`tiered_cache_bench.py` 的計時不含引擎建立與 checkpoint 的 prefill。「逐出並複製」的時間是 prefill 時間減去同一段
prefill（沒有東西可逐出）的時間，是估計值。bench 的提示由 golden 文件重複拼接而成，只用於計時，不檢查輸出。

**新的加速旗標只能在 `tiered_cache_test.py` 全部 PASS 之後才預設開啟。** 例如 `SPILL_PINNED_GIB`、`SPILL_ASYNC_IO`
這類選用旗標，先用 `tiered_cache_bench.py --compare` 量測，再以 `tiered_cache_test.py` 驗證精確性；測試未通過前不要寫進
`compose.yaml` 或 `tools/recommend_config.py` 的預設值。
