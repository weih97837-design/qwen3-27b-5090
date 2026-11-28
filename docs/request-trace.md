# 請求軌跡（Request trace）

給「依軌跡重播的快取淘汰模擬器」用的資料：記下每個請求的**大小、命中、時間與前綴雜湊**，之後可在 CPU 上離線比較不同淘汰策略。**預設關閉**，不設定時不寫檔，也沒有額外開銷。

## 開啟

```
REQUEST_TRACE=D:\q27-trace            # 目錄（檔名 trace-…jsonl），或檔名前綴 D:\q27-trace\load1
python frontend/server.py --request-trace D:\q27-trace   # 同義
REQUEST_TRACE_HASH_AGENTS=1           # agent 欄位改為穩定的短雜湊
REQUEST_TRACE_MAX_MIB=512             # 全部軌跡檔的總大小上限（預設 512）
REQUEST_TRACE_ANCHORS=0               # 關閉內容錨點（new_anchors，見下；開啟軌跡時預設開）
REQUEST_TRACE_ANCHOR_EVERY=128        # 約每幾個 token 一個錨點（預設 128）
REQUEST_TRACE_ANCHOR_K=32             # 每個錨點雜湊的 token 數（預設 32）
```

檔名為 `<前綴>-YYYYMMDD-HHMMSS.jsonl`（同一秒內再開新檔會加 `_1`、`_2`）。單檔達 64 MiB 就換新檔；總大小超過上限時刪除最舊的檔。每行寫完立即 flush。寫入發生任何錯誤時，記錄器只記一行 log 並自行停用，不影響服務。

## 檔案格式（JSON Lines，一行一個事件）

每個檔的第一行是 `hdr`：

| 欄位 | 說明 |
|---|---|
| `t` | `"hdr"` |
| `version` | 格式版本，目前 1 |
| `engine` | 引擎版本字串（同 `GET /version`） |
| `build` | Windows 程式包的版本（`app\VERSION`，例如 `v0.6.2.7-cache-tier`）；在原始碼 checkout 裡沒有這個檔案，值為 `null`。用來分辨不同版本的追蹤檔 |
| `block_tokens` | 256 |
| `anchor_every`, `anchor_k` | 內容錨點的參數（預設 128、32；`REQUEST_TRACE_ANCHORS=0` 時 `anchor_every` 為 0，`req` 沒有 `new_anchors`） |
| `ts` | 檔案開始時間（epoch 秒） |
| `settings` | 與快取有關的設定：所有 `SPILL_*` 與 `CACHE_*`（保留策略 `CACHE_POLICY` 及其常數）、`STATE_HOST_CACHE`、`CHECKPOINT_INTERVAL`、`PREFILL_CHUNK`、`MAX_SEQS`、`KV_POOL_TOKENS`（有設定才有）、`kv_pool_pages`、`kv_pool_tokens`（引擎實際的 KV 池）等 |

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
| `reusable_from` | 這段可重用前綴來自誰的歷史：`"same"`（共用該最長前綴的歷史 entry 之中有一個屬於同一個 agent，即 agent id 的 `/` 之前那段、快取的 owner）、`"other"`（只有其他 agent 的）、`"none"`（沒有可重用前綴）；無法判斷時為 `null`。歷史（最近 64 段）另外記每段的 agent 名稱，記憶體只多 64 個字串。只是量測：快取依內容查找，與 agent 無關 |
| `diverged_at` | 與最佳歷史分岔的位置，沒有則 `null` |
| `restored_from` | 這次命中從分層快取載回時為 `"ram"` 或 `"ssd"`，否則 `null`（無法判斷時為 `null`） |
| `restore` | 這個請求最後一次的還原決策（`worker.py` 的 `RESTORE_OUTCOMES`；見 `docs/tiered-cache.md`「還原決策計數」）：例如 `"started"`、`"joined"`、`"restored"`、`"failed"`、`"timed_out"`、`"not_worth_loading"`、`"no_kv_room"`、`"spill_room_refused"`、`"skipped_busy"`、`"chunk_missing"`、`"damaged"`、`"begin_refused"`、`"lookup_failed"`、`"no_match"`、`"not_checked"`；沒有分層快取或沒有判斷過時為 `null`。舊軌跡沒有這個欄位 |
| `restore_tier` | 該決策所針對 entry 的位置：`"ram"`、`"ssd"`；沒有 entry（`no_match`、`not_checked`）或沒有決策時為 `null` |
| `queued_s` | 排隊秒數 |
| `new_tokens` | `prompt_tokens - reusable`：前綴快取必須自己算的那一段（沒有錨點時省略） |
| `new_anchors` | 那一段（index >= `reusable`）內的內容錨點雜湊清單（12 位十六進位），見「內容錨點」 |
| `blocks` | 提示詞每 256 token 一塊（只含完整區塊）的鏈式雜湊，見下 |

`done`：請求結束。`ts`、`rid`、`reason`（`stop` / `length` / `cancelled` / `error`…）、`output_tokens`、`ttft_s`、`prefill_s`、`computed_tokens`（實際預填的 token）、`preempted`（被搶佔次數）、`total_tokens`（提示詞＋輸出）、`end_blocks`：（提示詞＋輸出）中位於「提示詞最後一個完整區塊之後」的區塊雜湊，也就是結尾檢查點涵蓋的部分。未開始的請求 `end_blocks` 為空。

`snap`：有請求在處理時，約每 10 秒一筆（沿用狀態頁的快照節奏，沒有額外計時器）。`ts`、`running`、`waiting`、`kv_free_pages`、`kv_checkpoint_only_pages`、`kv_pool_pages`、`checkpoints`（使用中的檢查點槽），以及累計值 `q27:prefix_cache_evictions_total`、`q27:tier_spilled`、`q27:tier_restored`、`q27:tier_dropped`、`q27:tier_written`（沒有分層快取時省略後四者）。

### 區塊雜湊

`h_k` 取 `blake2b(d_{k-1} + 第 k 塊的 int32 token id（小端位元組）)` 摘要的前 8 位元組（16 個十六進位字元）；`d_k` 為完整摘要，`d_{-1}` 為固定常數 `q27-request-trace-v1`。用的 id 就是交給 `prefix_begin` 的引擎 id（含媒體佔位 id），所以前綴相同的提示詞，前面的雜湊一定相同：較長提示詞的前 k 個雜湊等於較短者。125K token 的提示詞（488 塊）約 3 ms。

### 內容錨點（位置無關的量測）

前綴快取只能重用**從第一個 token 起完全相同**的內容。要回答「快取必須重算的內容，有多少其實早就在別的請求出現過，只是位置不同或前面的標頭不同」，`req` 的 `new_anchors` 記錄 `reusable` 之後那段的內容定義錨點（前綴那段已知可重用；依歸納，更早出現過的內容都曾在某個較早請求的「新」段記錄過，所以略過）：

- 錨點規則（只依 token id，與位置、前綴無關）：位置 i 的 token 是錨點，當 `((id * 2654435761) & 0xFFFFFFFF) >> 7) % EVERY == 0`（EVERY 預設 128，即平均每 128 token 一個），且 i 之後（含 i）在提示詞內至少有 K 個 token（K 預設 32）。
- 雜湊：以該處起 K 個 token 的 int32 位元組做 blake2b，取前 6 位元組（12 位十六進位）。相同內容放在不同偏移、不同標頭之後，只要錨點之後的 K 個 token 相同，雜湊就相同。
- 成本：125K token 的新段約 15 ms（量測值；只有尾段的請求更少），不逐 token 雜湊，只對約 1/128 的位置雜湊 32 個 token。
- 大小：平均每個請求約 42 個錨點（約 0.6 KB）；完全冷的 125K 提示詞約 970–1,000 個，每個在 JSON 中約 15 位元組，約 14.5 KB。沿用總大小上限（`REQUEST_TRACE_MAX_MIB`）與換檔，沒有另外的單請求上限。

## 隱私

只記數量、時間與雜湊，**不記提示詞文字、輸出文字、token id 或任何內容**。雜湊是單向的鏈式雜湊，但相同前綴會得到相同雜湊，所以檔案能看出「哪些請求共用前綴」。agent id 與機器名稱預設照實寫入；若要分享軌跡，設 `REQUEST_TRACE_HASH_AGENTS=1`（機器名稱仍在，分享前自行檢查）。短名稱的雜湊可被字典猜出，不是匿名化。

錨點只是 32 個 token 視窗的雜湊，不含文字也不含 token id。和任何文字雜湊一樣，已經握有候選文字的人可以做字典比對（把候選內容用同樣規則算一遍，確認某個視窗是否出現過）；除此之外無法反推內容。

## 大小

每個請求約每 256 token 一塊，每塊在 JSON 中約 19 位元組。50–125K token 的提示詞是 195–488 塊，約 3.7–9.3 KB；加上 `done` 與其他欄位，平均每個請求約 8 KB。每分鐘 14 個請求約 112 KB／分鐘，即每小時約 6.6 MB；預設 512 MiB 上限約可保留 3 天（約 75 小時）。`snap` 每 10 秒約 250 位元組，可忽略。

## 摘要

```
python tools/trace_summary.py trace-*.jsonl [--json]
```

只用 CPU，印出請求數、各 agent 的請求數、提示詞大小分佈、平均 `cached`／`reusable`、大於 16K 的提示詞完全沒有命中的比例、有 `diverged_at` 的比例與位置分佈、各 agent 相鄰請求的間隔（中位數、p90），以及「可重用但沒命中」的 token 總數；有 `reusable_from` 的軌跡另外印出各來源（same／other／none）的請求數與可重用 token 數，以及「可重用 token 中來自其他 agent 的比例」。有 `restore` 的軌跡另外印一行 `restore decisions (tier outcome: requests)`：每種「`restore_tier` `restore`」組合的請求數（沒有 tier 時為 `none`）。

### 內容錨點的摘要

`trace_summary.py` 依檔案／時間順序維護「錨點雜湊 → 見過的 agent 集合」。對每個請求，算它的 `new_anchors` 有多少比例已出現在**較早**請求的新段中（任何 agent／同一 agent／其他 agent），並彙總：新 token 總數、估計「可依內容共用」的新 token 數（每個請求的比例 × `new_tokens`，加總；沒有錨點的短尾段不計入分母）、每請求比例的分佈（p10／中位數／p90 與 0%、0–25%、25–75%、75–100%、100% 的直方圖），以及依「`cached` 為 0 或大於 0」與 `reusable_from` 分組的同樣數字。沒有 `new_anchors` 的舊軌跡只印一行說明並略過。

這個數字**代表**：位置無關的重用「最多」能找到多少，是上限。它**不代表**能被前綴快取拿到：換了前綴之後，KV 與遞迴狀態都取決於之前的全部內容，所以逐字相同的內容在不同前綴後面無法精確重用（文獻中有近似方法，本引擎沒有）。誤差來源：每約 128 token 才取樣一個、視窗只有 32 token（視窗內任何一個 token 不同就不算）、純估計。實務上的解法是提示詞排版：共用內容放前面並維持固定順序，會變的內容放最後。

## 模擬器

```
python tools/cache_sim.py trace-*.jsonl --policy legacy,legacy-nofair,smart,lru,sglang_statecost,marconi_utility,ideal
python tools/cache_sim_gen.py synth.jsonl --workload mixed|continuation|diverging [--churn 0.08]   # 合成軌跡
```

把軌跡重播過 VRAM 檢查點（槽與頁）、RAM 層、SSD 層的模型，比較各保留策略（預設 `CACHE_POLICY=smart`，在真實軌跡上驗證預設就是用這個工具；`legacy` 是舊規則）；說明與近似見 `docs/tiered-cache.md`「CACHE_POLICY=smart」與 `tools/cache_sim.py` 開頭。

## 透過連接埠讀取

軌跡檔由**引擎行程**寫入，伺服器行程（`frontend/server.py`）只讀同一個目錄，所以不必登入引擎那台電腦的磁碟：區網上的另一台機器或工具可以直接用 HTTP 取。只有設了 `REQUEST_TRACE` 才有內容；`REQUEST_TRACE_HTTP=0` 關閉這三個端點。

| 端點 | 說明 |
|---|---|
| `GET /admin/trace` | JSON：`enabled`、`directory_name`（只有資料夾名稱，不含完整路徑）、`files`（每個檔的 `name`、`size`、`mtime`、`first_ts`、`last_ts`、`complete`、`complete_bytes`）、`total_bytes`、`max_bytes`（`REQUEST_TRACE_MAX_MIB`）。沒開軌跡時回 200 與 `{"enabled": false}` |
| `GET /admin/trace/<檔名>` | 該檔，`application/x-ndjson`，串流輸出。檔名必須與清單中的某個名稱完全相同（含 `..`、斜線、編碼過的路徑、符號連結一律是 400／404）。支援 `Range: bytes=a-b`（206）與 `?from=<位元組位移>`；標頭 `X-Trace-Offset` 是已送出的最後一個完整行之後的位移，下次輪詢用 `?from=` 接續 |
| `GET /admin/trace/all[?since=<epoch 秒>]` | 全部的檔，由舊到新串接，串流輸出；`since` 會略過 `last_ts` 小於它的檔（用 `?since=` 取最近一段） |

- **只送完整的行。** 最新的檔正被引擎附加；每個回應都在最後一個換行處截斷，寫到一半的行不會出現，`X-Trace-Offset`（也是 `complete_bytes`）就是那個位置。`Range` 的總長也以此為準，所以不會讀到半行。`Range` 的 `X-Trace-Offset` 是 `end + 1`（位元組位移，不保證在行界）。
- **`from`：** 位移等於目前的完整長度時回 200 與空內容（沒有新資料）；超過時回 416，`X-Trace-Offset` 與 `Content-Range: bytes */<長度>` 給出實際長度（檔案被換掉或截短了，從 0 重來）。`from` 與 `Range` 不能同時用（400）。
- **`complete`：** 該檔已封存（不是最新的一個，之後不會再被附加）。最新的檔一律 `false`。`first_ts`／`last_ts` 取自第一行與最後一個完整行的 `ts`，只讀檔頭與檔尾，並以 `(size, mtime)` 快取。
- **壓縮：** 沒有 `Range` 時，請求帶 `Accept-Encoding: gzip` 就以 gzip 串流回應（`Transfer-Encoding: chunked`）。全部都是分塊讀寫（256 KiB），伺服器的記憶體用量與檔案大小無關。`/admin/trace/all` 一律分塊傳輸，沒有 `Content-Length`。
- **輪詢：** 檔案換檔或刪除（超過總上限時最舊的被刪）時，`/admin/trace/<檔名>` 變成 404，改看清單。

```
curl -s http://HOST:8080/admin/trace
curl -s --compressed http://HOST:8080/admin/trace/all -o trace-all.jsonl
curl -s -D- "http://HOST:8080/admin/trace/trace-20260101-000000.jsonl?from=1048576" -o new.jsonl     # 看 X-Trace-Offset
curl -s -r 0-1048575 http://HOST:8080/admin/trace/trace-20260101-000000.jsonl -o head.jsonl

python tools/cache_sim.py http://HOST:8080/admin/trace/all --policy legacy,smart
python tools/trace_summary.py "http://HOST:8080/admin/trace/all?since=1767225600"
python tools/trace_fetch.py http://HOST:8080 --out trace-copy [--since TS] [--follow 60]   # 增量下載到本機
python tools/cache_sim.py trace-copy/*.jsonl --policy legacy,smart
```

`tools/trace_summary.py` 與 `tools/cache_sim.py` 的輸入可以混用檔案路徑與 `http://`／`https://` 網址（逐行串流讀取，gzip，不會把整個檔載入記憶體；`cache_sim` 本來就把請求事件放在記憶體中重播）。網址回 404 時會說明原因（軌跡沒開，或 `REQUEST_TRACE_HTTP=0`）。`trace_fetch.py` 把每個檔以原名存到 `--out`，狀態檔 `.trace_fetch.json` 記錄各檔已取得的位移，重跑或 `--follow SECONDS`（每隔幾秒輪詢，Ctrl-C 結束）只取新的位元組，只寫入完整的行；伺服器已輪替刪除的檔，本機副本保留；伺服器上的同名檔變短（被換掉）則從頭重抓。狀態頁（`/ui`）在軌跡開啟時多一個「追蹤檔」區塊：檔案數、總大小／上限、最新檔名，以及上述兩個連結。

### 存取與隱私

**沒有驗證**，和狀態頁（`/ui`）一樣：任何連得到這個連接埠的人都能下載軌跡，而伺服器可以聽在 `0.0.0.0`。檔案內容是 agent id、機器名稱、長度、時間與雜湊，沒有提示詞或輸出文字（見上面「隱私」），但相同前綴會得到相同雜湊，能看出誰在什麼時候用了多大的提示詞。要限制：

- `REQUEST_TRACE_HTTP=0`：關閉端點（記錄照常寫檔，仍可直接讀磁碟）。
- `REQUEST_TRACE_HASH_AGENTS=1`：agent id 改為短雜湊（機器名稱仍在）。
- 防火牆／綁定位址：只開放信任的網段到這個連接埠（或讓伺服器只聽 `127.0.0.1`）。
