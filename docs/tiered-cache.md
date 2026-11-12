# 分層 KV 快取（VRAM → RAM → SSD）

長上下文的代理（agent）對話會佔用大量 KV 快取。GPU 上的 KV 池（`KV_POOL_TOKENS`，預設 512K tokens）裝不下所有對話時，
引擎會把被逐出的 checkpoint 存到主機 RAM 和 SSD，而不是直接丟棄。之後同一段對話再來時，從較深的 checkpoint 還原，
不必重算整段前綴。

## 概要

流程如下：

1. **VRAM**：活躍的 KV 池。池滿時，最久未用的 checkpoint 被逐出。
2. **RAM**：引擎把被逐出的 checkpoint 存進主機 RAM（`SPILL_RAM_GIB`，預設 64 GiB），而不是丟棄。
3. **SSD**：`frontend/tiercache.py` 在閒置時，把存在 RAM 超過 `SPILL_RAM_RETENTION_S`（預設 3600 秒）的項目寫到
   `SPILL_DIR`（預設 `/cache/kv`，即 Docker 具名磁碟區 `kvcache`）。忙碌時則在超過兩倍保留時間後寫出。
   檔案分成「manifest ＋ 共用的 chunk 檔」（見下方「SSD 層去重」），共用同一段前綴的 entry 只存一份 chunk。
   同一個程式也會執行 SSD 容量上限（`SPILL_SSD_GIB`，以**實體**位元組計，每個 chunk 檔只算一次；從寫入最久的 entry 開始刪，
   優先刪除真的能釋放空間的 entry）。此上限一律執行，即使磁碟已被其他檔案佔滿。
   可用空間保留（`SPILL_SSD_MIN_FREE_GIB`）只在能靠刪除本層自己的檔案達成時才會追求。
   超過 `SPILL_TTL_DAYS` 天的檔案會被刪除，TTL 自寫入 SSD 起算（重用後會重新計時）。每 `SPILL_TICK_S` 秒執行一次。
4. **刪除**：超過 TTL 或超過容量上限的檔案被刪除。

請求到達時，`prefix_begin` 會找出對話前綴最深的命中點，並從 RAM 或 SSD 把該 checkpoint 還原回 VRAM。引擎端的 RAM
上限是最後一道防線，即使 `frontend/tiercache.py` 沒有執行，RAM 也不會無限制成長。

## 閒置時預先備份（VRAM 保留）

引擎完全閒置達 `VRAM_BACKUP_IDLE_S`（預設 60 秒）後，會把 VRAM 中尚未備份的 checkpoint **複製一份到 RAM，原本的
checkpoint 保留在 VRAM**（`engine.spill_backup_one`）。每次只複製一個（大型 checkpoint 約需 0.3–1 秒）；每複製完一個，
都會先檢查是否有新請求，有就先處理請求。只備份「下一批最可能被逐出」的 checkpoint（依 `evict_checkpoint` 的順序：已被同一對話更深的 checkpoint 涵蓋者免備份，其餘依最久未使用，取前 `VRAM_BACKUP_VICTIMS` 個，預設 8），且備份合計不超過 RAM 上限的一半，其餘空間留給真正被逐出的 checkpoint；RAM 超過上限時，備份最先被丟棄（它的 checkpoint 還在 VRAM，不會損失）。沒有可備份的項目後停止，直到下一次活動（請求進出、推理回合）才重新開始。停止時的寫出不受此限（備份全部）。

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

寫出一個項目前會先估計所需時間：`new_disk_bytes`（沒有時用 `ram_bytes`）除以寫入速度（以最近一次 ≥ 64 MiB 的寫出量測；尚未量測時保守地假設 1 GB/s）。
估計時間超過剩餘預算的項目不會開始寫（寫到一半被 `server.py` 強制終止的檔案會作廢，停止也會超時），較小的項目仍會嘗試；略過的項目計入回報的 `left`。

## 參數表

以下全部是 `compose.yaml` 中 `serve` 服務的環境變數，可在 `.env` 或 shell 中覆寫，於容器啟動時生效。

`SPILL_RAM_GIB`、`SPILL_RAM_RETENTION_S`、`SPILL_SSD_GIB`、`SPILL_SSD_MIN_FREE_GIB`、`SPILL_SSD_MIN_TOKENS`、`SPILL_TTL_DAYS`、`VRAM_BACKUP_IDLE_S`、
`SPILL_PINNED_GIB`、`SPILL_ASYNC_IO`、`SPILL_ASYNC_PROMOTE` 與 `SPILL_DIR` 也是啟動設定：可以在 /ui 的「Start settings」
（**分層快取** 區段）和 Windows 托盤程式的設定視窗（**Tiered cache** 群組）直接修改，不必手動編輯 `settings.ini` 的 `[env]`。
引擎程序與 `TierPolicy` 在引擎載入時才讀取這些值，所以要重新啟動引擎才會生效（/ui 按 **Restart**，托盤按 **Apply**）。
環境變數與 `settings.ini` 的值仍是伺服器啟動時的初始值；在 /ui 改的值只持續到伺服器重新啟動，要永久保留請寫入變數
（頁面會列出要加入的行）。RAM 與 SSD 的大小沒有人為上限，由機器決定；`SPILL_TICK_S`、`SPILL_FLUSH_ON_STOP_S` 與
`SPILL_PINNED_SLAB_MIB` 仍只能用變數設定。

| 變數 | 預設 | 說明 |
| --- | --- | --- |
| `KV_POOL_TOKENS` | `524288`（512K） | GPU 上的 KV 池大小（tokens）。`0` 表示使用 VRAM 預算剩下的全部空間 |
| `MAX_SEQS` | `4` | 同時執行的請求數上限 |
| `SPILL_RAM_GIB` | `64` | 主機 RAM 中保存被逐出 checkpoint 的容量（GiB）。`0` 表示關閉分層 |
| `SPILL_DIR` | `/cache/kv` | checkpoint 寫入的目錄（容器內），對應具名磁碟區 `kvcache`。底下有 `entries/`（manifest）與 `chunks/`（共用的 chunk 檔） |
| `SPILL_RAM_RETENTION_S` | `3600` | checkpoint 在 RAM 中保留的秒數，之後才寫入 SSD |
| `SPILL_SSD_GIB` | `100` | SSD 層容量上限（GiB），最舊的檔案先刪除 |
| `SPILL_SSD_MIN_FREE_GIB` | `10` | 磁碟可用空間目標。只在能靠刪除本層自己的檔案達成時才追求，不會刪除其他檔案 |
| `SPILL_SSD_MIN_TOKENS` | `32768` | 短於此 token 數的 checkpoint 不會寫入 SSD：超過 RAM 保留時間後直接從 RAM 丟棄。每個 checkpoint 都帶約 190 MiB 的遞迴狀態，與長度無關；短前綴重算只需約 0.2–0.4 秒，與讀回時間相近，寫入只會浪費 SSD 空間與寫入壽命。`0`：不設下限（舊行為）。已在磁碟上的檔案不受影響；`/metrics` 的 `ssd_skipped_short` 計算被丟棄的筆數 |
| `SPILL_TTL_DAYS` | `7` | 自寫入 SSD 起算（重用後會重新計時），超過此天數的檔案會被刪除 |
| `SPILL_DEDUP` | `1` | `0` 關閉 RAM 層的 chunk 去重（每個 entry 各存完整 KV）。見「RAM 層去重」 |
| `SPILL_TICK_S` | `30` | 分層維護程序的執行間隔（秒） |
| `VRAM_BACKUP_VICTIMS` | `8` | 閒置備份只複製「下一批會被逐出」的前 N 個 checkpoint（`0`：全部）。只能用環境變數設定 |
| `VRAM_BACKUP_IDLE_S` | `60` | 引擎閒置多久後開始把 VRAM checkpoint 複製到 RAM（保留在 VRAM）。`0` 關閉，見上文 |
| `SPILL_FLUSH_ON_STOP_S` | `8` | 停止時備份與寫出的時間預算（秒），上限 8。`0` 關閉，見上文 |

`KV_DTYPE` 維持 `nvfp4`，每個請求的上限 `MAX_TOKENS` 維持 `262144`。

## 容量估算

- 每個 token 的 NVFP4 KV 約 18 KiB。
- 一個 256K tokens 的代理：KV 約 4.5 GiB + 狀態約 0.19 GiB，合計約 4.7 GiB。
- 32 GiB RAM 約可保存 6 個代理。
- 100 GiB SSD 約可保存 21 個代理。
- 512K 的 KV 池約佔 9 GiB VRAM。

## RAM 層去重（預設開啟，`SPILL_DEDUP=0` 關閉）

4–8 個共用同一段長 system prompt／工具定義的代理，或同一段對話的多個 checkpoint，它們的 KV 開頭完全相同。RAM 層因此把 entry 的 KV
切成 **4096 token（64 頁）的 chunk**：整塊 chunk 依內容定址（key 串接：前一塊的 key 加上這一塊的 token id，起點含模型／KV 型別指紋，
128 位元；命中時再比對儲存的 token id，碰撞只會變成 miss，不會用到錯的 KV），相同內容只存一份，各 entry 以**參照計數**指到它；
不足一個 chunk 的尾端（含最後一個未滿的頁）與遞迴狀態仍是各 entry 自己的。chunk 一律依 token 數對齊（`pos / 4096`），所以含未滿頁的區塊不會被共用。

- **RAM 層與 SSD 層各自去重**。RAM 內以 chunk 共用；寫到 SSD 時同一批 chunk 寫成共用的 chunk 檔（見下一節）。從 SSD 還原的 entry
  在 RAM 中是「私有」的（整段線性串流，不去重）：它的 KV 隨即複製進 KV 池。還原（promote）時逐塊複製回 KV 池頁，最後放掉參照。
- **容量**。`SPILL_RAM_GIB` 管的是**實體**位元組（尾端＋狀態＋每個常駐 chunk 只算一次）；SSD 預算與檔案大小仍是各 entry 的**邏輯**大小
  （`spill_list` 的 `ram_bytes`、frontend 的 `ram_logical_bytes`）。超過上限時，先釋放沒人用的 chunk，再逐出「釋放最多記憶體」的 entry
  （釋放量相近時取最久未用者）；與別人共用 chunk 的 entry 釋放得少，所以被保留。
- **統計**。`spill_stats` 在原有欄位後追加 `logical_bytes`、`physical_bytes`、`chunks`、`chunk_hits`、`chunk_misses`、`dedup_saved_bytes`
  （舊引擎沒有這些鍵）。`/stats` 的 tiers 區塊有 `ram_logical_bytes`、`ram_physical_bytes`、`ram_dedup_ratio`（邏輯 ÷ 實體）、
  `ram_dedup_saved_bytes`、`ram_chunks`、`chunk_hits`、`chunk_misses`，/ui 的 RAM 層顯示邏輯大小與去重倍率；`ram_bytes` 即實體用量。
  Prometheus：`q27:tier_ram_logical_bytes` 等。
- 測試：`engine/tests/test_chunk_store.cc`（主機端，ASan／UBSan）與 `scripts/tiered_cache_test.py` 的情境 `h`（共用 6 個 chunk 前綴的
  4 段對話：實體 < 0.6 × 邏輯、`chunk_hits > 0`、還原後 token 與無快取參考完全相同）、`i`（同上，加 pinned 與 async I/O）。
  設計與不變量見 `engine/src/spill.cc` 開頭的 (H)–(L) 與 `engine/src/chunk_store.h`。

## SSD 層去重（chunk 檔 ＋ manifest）

RAM 層去重的同一批 4096 token chunk，寫到 SSD 時也只存一份。`SPILL_DIR` 的結構：

```
$SPILL_DIR/
  chunks/<32 位十六進位 key>-<16 位十六進位雜湊>.kvc    一個 chunk 的 KV，多個 entry 共用
  entries/<id>.q27m                                  一個 entry 的 manifest
```

- **chunk 檔**（`Q27CHK1`）：標頭（指紋、chunk key、父 chunk 的檔名、頁數、每頁位元組數、slice 數、token 數）、chunk 的 token id、KV 位元組
  （[slice][chunk 內頁]，與 RAM chunk 的排列相同）、結尾的 Hash64 校驗。檔名由內容推得，不用記憶體中的 uid（重啟後不穩定）：
  key 是 RAM 層的鏈式 key（含模型／KV 型別指紋與到這個 chunk 為止的所有 token），後半是「父檔名 ＋ 本 chunk token id」的獨立雜湊。
  相同前綴因此在不同 entry、不同行程、重啟前後都是同一個檔案。
- **manifest**（`Q27MAN1`）：標頭、全部 token id、chunk 檔名清單、遞迴狀態、尾端 KV（不足一個 chunk 的部分）、Hash64 校驗。
  `SPILL_DEDUP=0` 或不足 4096 token 的 entry 沒有 chunk，整段 KV 都在尾端（等於舊的整檔，只是換了格式）。
- **寫出順序（崩潰安全）**：先寫缺少的 chunk 檔（`.tmp` → fsync → rename），再 fsync `chunks/`，**最後**才寫 manifest（同樣 `.tmp` → fsync → rename →
  fsync `entries/`）。已存在的 chunk 檔會驗證（大小、標頭、token id、父檔名）才略過；對不上就不共用，該次寫出失敗（entry 被丟棄）。
  Windows 以 `_commit` 取代 fsync，沒有目錄 fsync（只能靠檔案本身的 flush）。
- **參照計數**：引擎維護磁碟上的 chunk 索引，一個 chunk 檔的計數＝列出它的 manifest 數。只有引擎刪檔：刪除／TTL／容量逐出／還原完成時，先刪 manifest，
  再讓其 chunk 檔計數減一，歸零才刪檔。還原中（busy）的 entry 仍持有參照，所以正在被讀的檔不會被刪。
- **啟動**：frontend 對 `entries/*.q27m` 逐一 `spill_import`（索引由 manifest 建立；chunk 檔不存在或大小不對的 manifest 被丟棄），
  全部匯入後呼叫 `spill_sweep` 刪除 `.tmp` 與沒有任何 manifest 參照的 chunk 檔（崩潰留下的）。引擎不會自己清掃（它不知道匯入何時結束）。
  frontend 不會列舉或刪除 `chunks/`，只管 `entries/`。
- **舊檔案**：舊版的 `SPILL_DIR/*.q27kv`（整檔格式）不再被讀取，第一次以新引擎啟動時會全部刪除，並記錄一行 log（它只是快取）。
  舊引擎（沒有 `q27_spill_disk_usage`）仍用舊的平面檔案配置，檔案不會被當成舊檔刪除。
- **降級到舊版引擎**：舊引擎不認得 `entries/` 與 `chunks/`：這兩個目錄裡的檔案不會被計入 SSD 容量、不會被 TTL 或容量逐出清理，也不會被刪除，
  會一直佔用磁碟。降級後請手動刪除 `SPILL_DIR/entries/` 與 `SPILL_DIR/chunks/`（它們只是快取）。
- **匯入失敗與已索引**：`spill_import` 回傳 0 才表示引擎明確拒絕（別的模型／KV 型別、損毀、缺 chunk 檔），frontend 才刪該 manifest；呼叫本身拋出例外時
  檔案保留。已經索引過的路徑回傳既有的 id（不是 0），所以重複呼叫 `start()` 不會刪掉正在使用的 manifest。
- **還原**：讀 manifest 與它列出的所有 chunk 檔到一個私有緩衝區（`DiskRead`，每次還原一條執行緒，只碰自己的緩衝區）；每個檔案驗證標頭、
  token id、大小與 Hash64。缺少或損毀的 chunk 檔會讓這次還原乾淨地失敗：entry（及其 manifest）被丟棄，該 chunk 檔從索引移除並刪除，
  之後的寫出會重新寫它。**I/O 錯誤不是損毀**：開檔失敗（檔案存在但被鎖，例如 Windows 防毒／索引程式）、`fread` 出錯（EIO）只讓這一次還原失敗，
  **不刪** chunk 檔、manifest 與 entry（chunk 0 幾乎被所有 manifest 共用，誤刪會讓整個 SSD 層失效）；同一個 entry 連續 3 次 I/O 失敗才丟棄它。
  只有「檔案不存在」、標頭／token id／Hash64 不符、截斷（讀到 EOF）才算損毀。刪檔失敗（Windows：另一個 `DiskRead` 開著該檔、沒有 `FILE_SHARE_DELETE`）
  時，檔案移到「待刪清單」：仍計入 `spill_disk_usage` 的實體位元組（第 5 個欄位 `pending_files` 為其數量），在下一次刪檔或 `spill_sweep` 時重試；frontend 在
  tick 發現 `pending_files > 0` 時會呼叫 `spill_sweep`。寫 chunk 檔最後的 rename 失敗時，若目標檔已存在且驗證相符（chunk 內容由名稱決定）則視為成功。
  尚未在 RAM 中的 chunk 不會先登記進 RAM 層（避免讀到一半的資料被別的 spill 找到）；也沒有重用 RAM 中已常駐的 chunk 來省略讀取。
- **容量與統計**。`SPILL_SSD_GIB`、`SPILL_SSD_MIN_FREE_GIB` 與 `make_room` 以實體位元組計（`q27_spill_disk_usage`：每個 chunk 檔一次 ＋ manifest）。
  要放入一個 entry 時，需要的空間是它「現在會寫出的位元組」（`spill_list_ex` 的 `new_disk_bytes`：manifest ＋ 還不在 SSD 的 chunk）；
  逐出 entry 釋放的是 `unique_disk_bytes`（它的 manifest ＋ 只有它參照的 chunk 檔）。每次逐出後重新讀清單，因為兄弟 entry 的 unique 與新 entry 的
  `new_disk_bytes` 都會變。大部分內容與別人共用的 entry 排在後面（釋放得少）；挑選逐出對象的邏輯集中在 `TierPolicy._pick_victim`。
  TTL 仍從 manifest 的寫入時間起算。`status()` 多了 `ssd_logical_bytes`、`ssd_physical_bytes`、`ssd_dedup_ratio`，`/stats` 的 tiers 區塊、/ui 的 SSD 層
  （邏輯大小與去重倍率）與 Prometheus（`q27:tier_ssd_logical_bytes`、`q27:tier_ssd_physical_bytes`）同步更新；`ssd_bytes` 現在是實體用量。
- 測試：`engine/tests/test_spill_files.cc`（格式往返、損毀／截斷偵測、崩潰殘留、參照計數，ASan／UBSan／TSan）、`engine/tests/test_spill_read.cc`
  （多檔案 `DiskRead`、共用 chunk 檔的並行讀取，TSan）、`frontend/tests/test_tiercache.py`（共用 chunk 的 make_room、unique 位元組、TTL、舊檔處理、舊引擎回退、狀態欄位）、
  `scripts/tiered_cache_test.py` 的情境 `j`（見下表）。不變量 (M)–(Q) 在 `engine/src/spill.cc` 開頭，檔案格式在 `engine/src/spill_files.h`。

## 加速選項（v0.3.2 起預設開啟）

目前每次 spill（逐出）與 promote（前綴命中時搬回）都是在運算 stream 上以一般（pageable）主機記憶體複製，複製完才繼續，
因此會暫停所有 GPU 工作與引擎執行緒，速度也只有 PCIe 的一半左右。三個選項可改善，**預設都關閉**，行為與先前審查過的版本完全相同；
請先在自己的 GPU 上通過 `scripts/tiered_cache_test.py`（輸出 token 必須與參考完全相同）再開啟。

| 變數 | 預設 | 說明 |
| --- | --- | --- |
| `SPILL_PINNED_GIB` | `8` | spill entry 可使用的 pinned（page-locked）主機記憶體上限（GiB）。0 表示關閉（v0.3.2 起預設 8） |
| `SPILL_ASYNC_IO` | `1` | `1`：KV 頁在專用的 I/O stream 上搬移，不再暫停運算 stream 與引擎執行緒 |
| `SPILL_ASYNC_PROMOTE` | `1` | `1`：命中已 spill 的 checkpoint 時，請求進入「載入快取」狀態，由引擎在背景載入，其他請求照常執行（delay hit） |
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

## 代理識別（X-Agent-Id）

為了之後能依代理（agent）公平地逐出分層快取，伺服器會標記每個請求屬於哪個代理：

- 標頭 `X-Agent-Id: <id>`（優先），或 OpenAI 請求本文的 `"user": "<id>"`；兩者都有時以標頭為準。
- id 最長 128 字元，允許 `A-Za-z0-9._/-`；其他字元換成 `_` 並截斷；空字串視為未提供。
- 公平性的代理 = 第一個 `/` 之前的部分（`manager/task-123` 與 `manager/probe` 都是 `manager`）；完整 id 也會保留。
- 未提供時，由 worker 以提示詞開頭（系統提示詞加第一則使用者訊息）的雜湊產生 `auto:<8 位十六進位>`。限制：開頭相同的代理會共用同一個身分。

worker 記錄每個請求存下的 checkpoint 位置，並以 `Scheduler.agent_of_ids(ids, pos)` 查詢（有上限的字典，最近 4096 筆，鍵為
`agent.prefix_fingerprint(ids, pos)`）。統計頁有「Agents」表（RAM／SSD 位元組目前為空，待逐出機制填入）。

## WSL2 注意事項

- **RAM 層計入 WSL VM 的記憶體。** 預設的 WSL2 記憶體上限不足以容納 32 GiB 的 RAM 層。請在 `%UserProfile%\.wslconfig`
  設定 `memory=`，例如在 96 GB 的機器上，扣掉系統保留 16 GB 後設為 `memory=80GB`（建議配置工具會依主機 RAM 算出這個值），之後執行 `wsl --shutdown` 重新啟動。
- **刪除檔案不會縮小 vhdx。** WSL2 的虛擬硬碟在檔案刪除後不會自動縮小，因此 SSD 上限不會把主機磁碟空間還給系統。
- **使用具名磁碟區。** `kvcache` 位於 WSL2 的 ext4 磁碟上。不要改用 `/mnt/c` 的 bind mount，drvfs 讀寫很慢。
- **權限修正前建立的磁碟區要刪除一次。** 較舊的建置建立的 `kvcache` 磁碟區可能屬於 root，容器無法寫入 SSD 層（`q27:tier_failed` 會增加）。請執行一次 `docker compose down`，再 `docker volume rm qwen3-27b-5090_kvcache`（會刪除 SSD 層的全部快取），然後重新啟動。

## 限制

- 每段對話只保留最深的一個 checkpoint。
- RAM 層與 SSD 層去重都只共用 4096 token 對齊的整塊 chunk；尾端、遞迴狀態不共用。從 SSD 還原的 entry 在 RAM 中不去重。
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

`tiered_cache_test.py` 有十個情境（a–j），各自在獨立的行程中執行（環境變數在引擎建立時讀取）：

| 情境 | 強制走的路徑 | 檢查的計數器 |
| --- | --- | --- |
| a 從 RAM 還原 | 小的 `KV_POOL_TOKENS`：第二段對話的 prefill 逐出第一段的 checkpoint | `spilled` 增加；後續回合 `restored` 增加 |
| b 從 SSD 還原 | 同 a，再把 RAM 中的項目 `spill_export` 到 `SPILL_DIR` | 後續回合從檔案還原（`restored` 增加，且還原前該項目 `on_disk`） |
| c 重啟後匯入 | 第一個行程寫出檔案後結束；第二個行程 `spill_import` 後續談 | 匯入的檔案數；後續回合 `restored` 增加 |
| d 閒置備份重用 | `spill_backup_one` 直到 False；之後的逐出不再複製 | `backed_up` 增加；逐出時 `spilled` 不變；後續回合 `restored` 增加 |
| e 推測解碼 | 同 a，使用草稿模型（環形 checkpoint）；沒有草稿模型時跳過 | 同 a |
| f pinned＋async I/O | 同 a、b，`SPILL_PINNED_GIB=2`、`SPILL_ASYNC_IO=1`（entry 分成多個 pinned 區段） | `pin_hits` 增加 |
| g 背景載入 | 同 a、b，再加 `SPILL_ASYNC_PROMOTE=1`，直接呼叫引擎 API：`spill_match` → `spill_restore_begin`（RAM 回 1、檔案回 2）→ 載入中的 checkpoint 對 `prefix_begin` 不可見 → `spill_restore_poll` 直到回報該 id | `restored_async` 增加、`restore_failed` 不變；續談從上一回合結束處開始 |

| h RAM 去重 | 4 段共用 6 個 chunk（24576 token）前綴的對話，全部溢出到 RAM | 實體 < 0.6 × 邏輯、`chunk_hits > 0`；還原後 token 與參考相同 |
| i RAM 去重＋pinned＋async | 同 h，加 `SPILL_PINNED_GIB`、`SPILL_ASYNC_IO`、`SPILL_ASYNC_PROMOTE`，全部經 `spill_match` / `spill_restore_*` 還原 | 同 h |
| j SSD 去重 | 4 段共用前綴的對話全部 `spill_export` 到 `SPILL_DIR`（兩個行程）：第一個行程寫出，第二個行程（新的 Engine、同一個目錄）`spill_import`、`spill_sweep`，丟棄一筆後從檔案還原其他筆（兩筆走 `prefix_begin` 的同步 promote，一筆走 `spill_restore_begin`） | `spill_disk_usage` 實體 < 0.6 × 邏輯、`.kvc` 檔數少於 chunk 參照數；重啟後仍然成立；丟棄釋放的位元組恰為 `unique_disk_bytes` 且共用的 `.kvc` 仍在；還原的 token 與參考相同；最後一筆還原後沒有殘留檔案 |

每個情境還會檢查續談是否從上一回合結束的位置開始（`cached` 等於 `pos`）。

**池的大小由腳本自動決定**：腳本依引擎的分頁算法計算池，包含 KV 區塊的捨入（每區塊的頁數為 2 MiB 除以每頁位元組數，
NVFP4 為 57 頁、FP8 為 32 頁、BF16 為 16 頁；`Q27_KV_BLOCK` 可覆寫）。從 `--prompt-len`（預設 3000）起逐步加大，直到兩段對話
超過可用頁數（餘 4 頁），使第二段的 prefill 必定逐出第一段的 checkpoint；開頭會印出 `prompt-len A -> P`，以及池的數字
（`max_pages`、`pool_pages`、`kv_block`、`usable pages`、每段對話的頁數）。一般不需要手動調 `--pool-tokens`。

若某個需要逐出的情境失敗，錯誤訊息會附上引擎 `cache_stats()` 的 `evicted`、`pool_pages`、`free_pages`；`evicted` 為 0 時會寫
`no eviction happened: pool too large`，表示池太大、沒有逐出。指定 `--pool-tokens` 時不會調整 `--prompt-len`，只印出數字；
若兩段對話沒有超過可用頁數，會警告「不預期逐出」。

**PASS 的意義**：該情境的每一段輸出 token（貪婪解碼，每回合 `--tokens` 個）與關閉快取的參考完全相同，而且所有檢查都通過，
表示確實走了該情境的路徑。任何一個 FAIL 則 exit code 為 1。

`tiered_cache_bench.py` 的計時不含引擎建立與 checkpoint 的 prefill。「逐出並複製」的時間是 prefill 時間減去同一段
prefill（沒有東西可逐出）的時間，是估計值。bench 的提示由 golden 文件重複拼接而成，只用於計時，不檢查輸出。

**新的加速旗標只能在 `tiered_cache_test.py` 全部 PASS 之後才預設開啟。** 例如 `SPILL_PINNED_GIB`、`SPILL_ASYNC_IO`
這類選用旗標，先用 `tiered_cache_bench.py --compare` 量測，再以 `tiered_cache_test.py` 驗證精確性；測試未通過前不要寫進
`compose.yaml` 或 `tools/recommend_config.py` 的預設值。
