# 請求軌跡（Request trace）

給「依軌跡重播的快取淘汰模擬器」用的資料：記下每個請求的**大小、命中、時間與前綴雜湊**，之後可在 CPU 上離線比較不同淘汰策略。**預設關閉**，不設定時不寫檔，也沒有額外開銷。

## 開啟

```
REQUEST_TRACE=D:\q27-trace            # 目錄（檔名 trace-…jsonl），或檔名前綴 D:\q27-trace\load1
python frontend/server.py --request-trace D:\q27-trace   # 同義
REQUEST_TRACE_HASH_AGENTS=1           # agent 欄位改為穩定的短雜湊
REQUEST_TRACE_MAX_MIB=512             # 全部軌跡檔的總大小上限（預設 512）
```

檔名為 `<前綴>-YYYYMMDD-HHMMSS.jsonl`（同一秒內再開新檔會加 `_1`、`_2`）。單檔達 64 MiB 就換新檔；總大小超過上限時刪除最舊的檔。每行寫完立即 flush。寫入發生任何錯誤時，記錄器只記一行 log 並自行停用，不影響服務。

## 檔案格式（JSON Lines，一行一個事件）

每個檔的第一行是 `hdr`：

| 欄位 | 說明 |
|---|---|
| `t` | `"hdr"` |
| `version` | 格式版本，目前 1 |
| `engine` | 引擎版本字串（同 `GET /version`） |
| `block_tokens` | 256 |
| `ts` | 檔案開始時間（epoch 秒） |
| `settings` | 與快取有關的設定：所有 `SPILL_*`、`STATE_HOST_CACHE`、`CHECKPOINT_INTERVAL`、`PREFILL_CHUNK`、`MAX_SEQS`、`KV_POOL_TOKENS`（有設定才有）、`kv_pool_pages`、`kv_pool_tokens`（引擎實際的 KV 池）等 |

`req`：請求開始（`prefix_begin` 之後，所以命中與分岔已知）。

| 欄位 | 說明 |
|---|---|
| `t` | `"req"` |
| `ts` | epoch 秒（浮點數） |
| `rid` | 請求編號 |
| `agent` | 客戶端送來的 agent id（`X-Agent-Id` / `user`，含 `/` 後的工作階段部分）；沒送則為 `auto:<8 位十六進位>`。`REQUEST_TRACE_HASH_AGENTS=1` 時，以 `/` 分段各自取 5 位元組雜湊，保留 `/` 結構 |
| `client` | 與狀態頁「機器」欄相同的名稱（`X-Client-Id` / `CLIENT_NAMES` / 位址；本機為 `local`） |
| `prompt_tokens` | 提示詞 token 數（含媒體 token） |
| `media_tokens` | 其中圖片／影片的 token 數 |
| `max_tokens` | 允許的最大輸出 |
| `cached` | 快取供應的 token 數（`prefix_begin` 的回傳） |
| `reusable` | 與最近歷史的最長共同前綴（可重用上限） |
| `diverged_at` | 與最佳歷史分岔的位置，沒有則 `null` |
| `restored_from` | 這次命中從分層快取載回時為 `"ram"` 或 `"ssd"`，否則 `null`（無法判斷時為 `null`） |
| `queued_s` | 排隊秒數 |
| `blocks` | 提示詞每 256 token 一塊（只含完整區塊）的鏈式雜湊，見下 |

`done`：請求結束。`ts`、`rid`、`reason`（`stop` / `length` / `cancelled` / `error`…）、`output_tokens`、`ttft_s`、`prefill_s`、`computed_tokens`（實際預填的 token）、`preempted`（被搶佔次數）、`total_tokens`（提示詞＋輸出）、`end_blocks`：（提示詞＋輸出）中位於「提示詞最後一個完整區塊之後」的區塊雜湊，也就是結尾檢查點涵蓋的部分。未開始的請求 `end_blocks` 為空。

`snap`：有請求在處理時，約每 10 秒一筆（沿用狀態頁的快照節奏，沒有額外計時器）。`ts`、`running`、`waiting`、`kv_free_pages`、`kv_checkpoint_only_pages`、`kv_pool_pages`、`checkpoints`（使用中的檢查點槽），以及累計值 `q27:prefix_cache_evictions_total`、`q27:tier_spilled`、`q27:tier_restored`、`q27:tier_dropped`、`q27:tier_written`（沒有分層快取時省略後四者）。

### 區塊雜湊

`h_k` 取 `blake2b(d_{k-1} + 第 k 塊的 int32 token id（小端位元組）)` 摘要的前 8 位元組（16 個十六進位字元）；`d_k` 為完整摘要，`d_{-1}` 為固定常數 `q27-request-trace-v1`。用的 id 就是交給 `prefix_begin` 的引擎 id（含媒體佔位 id），所以前綴相同的提示詞，前面的雜湊一定相同：較長提示詞的前 k 個雜湊等於較短者。125K token 的提示詞（488 塊）約 3 ms。

## 隱私

只記數量、時間與雜湊，**不記提示詞文字、輸出文字、token id 或任何內容**。雜湊是單向的鏈式雜湊，但相同前綴會得到相同雜湊，所以檔案能看出「哪些請求共用前綴」。agent id 與機器名稱預設照實寫入；若要分享軌跡，設 `REQUEST_TRACE_HASH_AGENTS=1`（機器名稱仍在，分享前自行檢查）。短名稱的雜湊可被字典猜出，不是匿名化。

## 大小

每個請求約每 256 token 一塊，每塊在 JSON 中約 19 位元組。50–125K token 的提示詞是 195–488 塊，約 3.7–9.3 KB；加上 `done` 與其他欄位，平均每個請求約 8 KB。每分鐘 14 個請求約 112 KB／分鐘，即每小時約 6.6 MB；預設 512 MiB 上限約可保留 3 天（約 75 小時）。`snap` 每 10 秒約 250 位元組，可忽略。

## 摘要

```
python tools/trace_summary.py trace-*.jsonl [--json]
```

只用 CPU，印出請求數、各 agent 的請求數、提示詞大小分佈、平均 `cached`／`reusable`、大於 16K 的提示詞完全沒有命中的比例、有 `diverged_at` 的比例與位置分佈、各 agent 相鄰請求的間隔（中位數、p90），以及「可重用但沒命中」的 token 總數。
