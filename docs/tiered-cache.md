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
`SPILL_PINNED_GIB`、`SPILL_ASYNC_IO`、`SPILL_ASYNC_PROMOTE`、`SPILL_FAIR_AGENTS` 與 `SPILL_DIR` 也是啟動設定：可以在 /ui 的「Start settings」
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
| `SPILL_FAIR_AGENTS` | `1` | `1`：RAM 層或 SSD 層要騰出空間時，從**目前在該層用量最大的 agent** 挑逐出對象（共用 chunk 的位元組平分給引用它的各 agent），再依「被取代 → 從未重用 → 最久未用」。`0`：整層直接用同一個順序，沒有 agent 這一步。VRAM 仍是單純 LRU。見「保留策略」 |
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

引擎端標記：worker 在每次 `prefix_begin` 前呼叫 `set_owner(agent.owner_id(agent))`（`owner_id` 是 agent 名稱的 blake2b 取 63 位元，穩定且不為 0；0 代表未知），
該序列存下的 checkpoint（含被停放、還原後再存的）、由它溢出的 RAM entry、SSD manifest 與還原回來的 checkpoint 都帶著這個 owner。因為是名稱的雜湊，重啟後不需要
對照表，SSD 上舊 entry 的 owner 仍然是同一個 agent。（`Scheduler.agent_of_ids` / `note_checkpoints` 是較早的指紋方案，保留但保留策略不使用：`spill_list` 沒有
token id，指紋字典有上限、重啟即空。）統計頁的「Agents」表有 RAM／SSD 位元組與筆數；不屬於任何已知 agent 的 owner（未知，或前一次執行的 agent）合併顯示為 `unknown`。

## 影片輸入

伺服器接受影片（OpenAI 風格的 `video_url`，與 vLLM 相同）。影片在引擎內由 GPU 的 NVDEC 解碼，抽出的畫面縮放後交給視覺塔；每兩張畫面是語言模型眼中的一張「圖」，前面帶著它的時間（文字 `<12.5 seconds>`）。設定的完整說明見 `docs/configuration.md` 的「Long videos」，給用戶端開發者的版本見 `docs/agent-client-guide.md`。

### 格式與限制

- 容器：MP4／MOV（含 fragmented）、Matroska／WebM；編碼：H.264、HEVC、AV1、VP9（8 或 10 位元，4:2:0；H.264 與 HEVC 也支援 4:2:2，HEVC 4:4:4）。手機影片的旋轉資訊會套用。
- 一支影片最大 **256 MB**（解碼 base64 後的位元組數）；圖片 32 MB。超過回 HTTP 400（`video larger than 256 MiB`）。
- 來源是 `data:video/mp4;base64,...` 資料 URL 或 `http(s)` URL（伺服器自己下載，逾時 120 秒）。其他 URL 回 HTTP 400。
- 一個請求中所有圖片與影片的 token 合計超過 `IMAGE_TOKENS`（預設 131,072）回 HTTP 400（`the images and videos take N tokens, more than IMAGE_TOKENS`）。影片加文字還必須放得進上下文（`MAX_TOKENS`）。
- 引擎以 `VISION=off` 啟動時，帶圖片或影片的請求回 HTTP 400。
- 原生 Windows：影片用 NVIDIA 驅動程式的 `nvcuvid.dll` 解碼，需要安裝驅動程式（見 `docs/deploy-windows.md` 第 8 節）。

### 怎麼送

`POST /v1/chat/completions`，`content` 陣列放 `video_url` 部分與文字：

```json
{
  "model": "qwen3.8-27b",
  "messages": [
    {"role": "user", "content": [
      {"type": "video_url", "video_url": {"url": "data:video/mp4;base64,AAAAIGZ0eXBpc29t..."}},
      {"type": "text", "text": "請依時間順序說明影片裡發生了什麼。"}
    ]}
  ],
  "mm_processor_kwargs": {"fps": 1, "max_frames": 600},
  "max_tokens": 512
}
```

`mm_processor_kwargs`（可省略；也適用於 `/v1/responses`）：

| 鍵 | 值 | 意義 |
| --- | --- | --- |
| `fps` | 大於 0、至多 60 的數字（預設 2） | 每秒取幾張畫面 |
| `num_frames` | 整數，1 到 `VIDEO_MAX_FRAMES` | 直接指定張數（與 `fps` 互斥） |
| `min_frames`、`max_frames` | 整數，1 到 `VIDEO_MAX_FRAMES` | `fps` 算出的張數的下限與上限 |
| `size.longest_edge` | 正整數 | 這支影片所有畫面合計的像素上限，每 token 2,048 像素（`VIDEO_MAX_TOKENS` 乘 2,048 即預設值） |

畫面張數會進位到偶數；不認得的鍵會被忽略；壞值回 HTTP 400。每個請求的 `size.longest_edge` 只改變畫面大小，總 token 數仍受 `IMAGE_TOKENS` 限制。`/v1/messages`（Anthropic 格式）沒有這個欄位，只用預設值。

### 長度怎麼換成 token

預設 `VIDEO_MAX_TOKENS`＝T＝131,072、`VIDEO_MAX_FRAMES`＝F＝2,048（檢查點自己的是 12,288 與 768）。n 是取出的畫面張數（時長 × fps，限制在 4 到 F 之間，進位成偶數）；每兩張畫面（一對）最多 768 token。

| 區間 | 條件 | 每對的 token | 影片總 token | 預設（fps 2） | 檢查點自己的值（fps 2） |
| --- | --- | --- | --- | --- | --- |
| A：每對上限 | n ≤ 2T / 768 | 影片本身大小，至多 768（16:9 的 1080p、4K 為 720：1152×640） | 對數 × 每對，隨長度增加，小於 T | n ≤ 341，約 170 秒以內 | n ≤ 32，約 16 秒以內 |
| B：總 token 上限 | 2T / 768 < n < F | 約 2T / n：影片越長畫面越小 | 約 T | 170 秒到約 17 分鐘 | 16 秒到約 6.4 分鐘 |
| C：張數上限 | 時長 × fps ≥ F | 約 2T / F | 約 T | 約 17 分鐘以上：時間間隔變粗 | 約 6.4 分鐘以上 |

幾個例子（1080p 或 4K 的 16:9）：

| 影片 | 預設（T 131,072、F 2,048） | 檢查點自己的值（T 12,288、F 768） |
| --- | --- | --- |
| 10 秒，2 fps | 10 對 × 720 ＝ 7,200（A） | 同左 |
| 60 秒，2 fps | 60 × 720 ＝ 43,200（A） | 60 × 190（608×320）＝ 11,400（B） |
| 10 分鐘，1 fps | 300 × 405（864×480）＝ 121,500（B） | 300 × 32（256×128）＝ 9,600（B） |
| 10 分鐘，2 fps | 600 × 209（608×352）＝ 125,400（B），每 1.0 秒一對 | 384 × 28（224×128）＝ 10,752（C），每 1.6 秒一對 |
| 1 小時，2 fps | 1,024 × 120（480×256）＝ 122,880（C），每 3.5 秒一對 | 384 × 28 ＝ 10,752（C），每 9.4 秒一對 |

另外每一對還有 8 到 11 個文字 token（時間與開始／結束標記）。短於約 2.8 分鐘的影片兩種設定完全一樣。

### 記憶體

- **IMAGE_TOKENS 的嵌入表**：每 token 10 KiB 的 pinned 主機記憶體，引擎啟動時配置：131,072 時 1.25 GiB。
- **編碼時的縮圖畫面**：每 token 6 KiB 的一般主機記憶體，只在編碼那支影片時佔用：131,072 時約 0.75 GiB。
- **NVDEC 的暫存**：解碼時向 KV 池借用（`kv_lend`），數十到數百 MiB（解析度與編碼而定），解完歸還；借不到時才佔用預算之外的 VRAM。
- **KV**：和文字一樣，每 token 18 KiB（NVFP4 KV）。43,200 token 的影片約 0.77 GiB；T＝131,072 的影片約 2.3 GiB。存成快取 entry 還要加每筆約 190 MiB 的遞迴狀態。
- **原生 Windows**：pinned 記憶體計入 GPU 共用記憶體（約為 RAM 的一半）。`STATE_HOST_CACHE`（8 GiB）、`SPILL_PINNED_GIB`、視覺塔（`VISION=host`，0.86 GiB）與嵌入表（1.25 GiB）加起來要留在其中。RAM 在 32 GiB 以下時把 `VIDEO_MAX_TOKENS` 改成 32768（嵌入表 0.31 GiB）；`tools/recommend_config.py` 會這樣建議，並把影片緩衝從 RAM 層可用量中扣除。

### 快取行為

- 影片的每一對畫面在提示詞裡的 id 由「影片位元組的 sha256、取樣計畫（格點、大小、第幾對、來源畫面）」決定（`reference/qwen27b/image_tokens.py`），所以**同一支影片、同一個取樣計畫，得到同樣的 id**；不同的影片、不同的 fps／`max_frames`，id 不同，不會誤命中。這些 id 走和圖片相同的前綴快取與 chunk 鍵鏈（`kVocab` 以上的 id），所以影片的 KV 與一般文字一樣溢出到 RAM、SSD 並還原。
- **同一支影片再問一次便宜**：落在已快取前綴內的對，不會再解碼、不會再過視覺塔、也不會再 prefill。多回合對話（把上一回合的回覆原樣帶回，見 `docs/agent-client-guide.md`）整支影片都在快取內，只算新的文字。同一支影片換一個問題：最後一則使用者訊息裡最後一段圖片或影片的結尾（含收尾的 `<|vision_end|>`）有自己的檢查點（`Scheduler.checkpoints`，與其他檢查點一樣向下對齊 64 token），所以第二個問題從影片結尾續起，不再解碼、編碼或 prefill 影片的任何一對（對齊造成的最多 63 個 token，可能讓最後一對在續起點之後，只重算那一對）；只有問題文字重新 prefill。條件：影片與問題在同一則最後的使用者訊息、影片在問題之前；影片在較早的訊息裡時，由多回合對話的回合結尾檢查點涵蓋。這個檢查點與其他位置保持 64 token 以上的間隔（太近則不加；與生成提示起點同一個 64 區塊附近時取代它），多個媒體部分只取最後一個的結尾。每個檢查點約佔一筆 190 MiB 的狀態槽，所以一個影片提示詞只多一個。
- 例子：一支 4K 60 秒的影片（預設 2 fps，60 對 × 720 ＝ 43,200 token）。第一次要完整解碼、編碼 60 對並 prefill 約 43.8K token；存下的 entry 約 0.95 GiB，超過 `SPILL_SSD_MIN_TOKENS`（32,768），閒置 `SPILL_RAM_RETENTION_S` 後會寫到 SSD。之後同一個對話（或同一支影片）再來，從 VRAM、RAM 或 SSD 還原該 entry，不需重算那 43K token。
- 取樣計畫不同的同一支影片（例如把 `fps` 從 2 改成 1）是另一份 id，不共用。要讓不同請求共用快取，請固定影片位元組與 `mm_processor_kwargs`。
- 保留策略與公平性不為影片特別處理：entry 屬於送出請求的代理（`X-Agent-Id`），與文字相同。
- `spill_fingerprint`（快取檔案的身分）不含 `VIDEO_MAX_*`、`IMAGE_MAX_PIXELS`、fps：它們都已經反映在 id 裡（格點、大小、取樣計畫）。視覺塔的數值不在 id 裡，所以 `VISION=fp8`（權重 E4M3，嵌入誤差約為 `bf16` 的 3 倍）會讓指紋不同；`host` 與 `bf16` 逐位元相同，指紋不變（既有的快取檔照常使用）。
- 單一請求放不進整個 KV 池（提示詞加輸出預留超過池）時，直接回 HTTP 400，訊息列出需要與池的 token 數與影片佔了多少；其他請求佔住池而暫時放不下時會排隊，統計頁的請求列顯示「waits for: KV room」。

### 建議

- 預設下，約 17 分鐘（fps 2；fps 1 為約 34 分鐘）以內的影片時間間隔不變，只是超過約 170 秒（fps 2）後畫面逐漸變小（B）；再長時間間隔變粗（C）。**超過 10 分鐘的影片請先剪成段落，或調低 `fps`**（例如 `"mm_processor_kwargs": {"fps": 0.5}`），否則 prefill 與記憶體都以十萬 token 計。
- 一支 T 這麼大的影片，prefill 時間與同樣長度的長文字提示詞相同（未針對影片實測）；影片只在第一次付出這個成本。
- 模型在比檢查點預設（768 張、12,288 token）多得多的畫面與 token 上的答題品質，是模型的事，不是引擎的：請用自己的影片比較。
- 要少用記憶體：調低 `VIDEO_MAX_TOKENS`（也同時降低 `IMAGE_TOKENS`）。

## 保留策略（價值導向逐出）

實測（v0.3.1，無去重）：RAM 層 64 GiB 全滿、175 筆；SSD 200 GiB 全滿、624 筆（平均約 320 MB，一半是約 150–190 MiB 的遞迴狀態）；962 筆溢出到 RAM、只有
45 筆被還原（< 5%）。快取留住的大多是不會再用的 entry。因此 RAM 與 SSD 兩層用**同一條**逐出規則（實作：`engine/src/retention.h`、`chunk_store.h` 的
`pick_victim` 給引擎的 RAM 層；`frontend/tiercache.py` 的 `_pick_victim` 給 SSD 層，兩者互相對照測試）。victim 是下列順序的最小值：

1. **用量最大的 agent**（`SPILL_FAIR_AGENTS=1`，預設）：該層裡 agent 的位元組 = 它自己 entry 的 tail／狀態／manifest，加上共用 chunk 的位元組除以「引用該 chunk 的
   不同 agent 數」。完全彈性，沒有保證的最小份額；未知 agent 自成一桶（`unknown`）。同量時 id 小的先。
2. 在該 agent 內，**被取代（superseded）**的 entry 先走。
3. **從未重用（hits = 0）**先於重用過的。`hits` = 該 entry（或它由來的 VRAM checkpoint）被還原或被 `prefix_begin` 選為命中的次數；隨 entry 在 VRAM／RAM／SSD 間移動，
   並存在 SSD manifest 裡。
4. 最後才是舊規則：釋放記憶體最多者，且在最佳值 2 倍範圍內取最久未用（SSD：unique 位元組太少的排後面，其餘最舊寫入先）。

引擎的 RAM 層仍分兩類：閒置備份（VRAM 還有原件）永遠先於唯一副本，上面的順序在各類內套用。

**被取代**：entry 的 token 是另一個 entry（RAM、SSD 或 VRAM 池中的 checkpoint）的嚴格前綴、較深者可以代替它（草稿環形緩衝有效性涵蓋）、長度 > 16384
（`kKeepShort`；更短的是新對話共用的系統提示詞）、**不是分岔點**（所有延伸它的 entry 都在最深者那條鏈上）、也**不是共用前綴**（它與延伸它的 entry 屬於 ≥ 2 個 agent）。
被取代的 entry **不寫入 SSD**：RAM 保留時間到時直接丟棄（`ssd_skipped_superseded`），停止時的 flush 也略過；在 RAM 與 SSD 都最先被逐出。
這與既有的 `spill_subsumed`（深的 checkpoint 溢出時刪掉淺的）與 `ckpt_covered`（VRAM）同一套 `covers()` 邏輯，差別是這裡補上分岔點與跨 agent 的保護。
（既有的 `spill_subsumed` 仍不看分岔點，未更動：它會在較深的 checkpoint 溢出時刪掉淺的，包括分岔點；這個行為只有 GPU 實測才能放心調整。）

**hits 的傳遞**：`prefix_begin` 選中的 checkpoint 命中數 +1，序列（`seq_hits_`，停放時存在 `SeqState`）把它帶給之後存下的更深 checkpoint（worker 每回合結尾都會存一個），
所以持續被使用的對話最深處不會被當成「從未重用」；命中長度 ≤ 16384 token（`kKeepShort`，各對話共用的系統提示詞）時**不**傳遞，否則每個以熱門系統提示詞開頭的新對話都會繼承它的次數。
注意：超過 16384 token 的繼承使延續中的對話在「重用／從未重用」統計（`*_reused_*`／`*_never_*`）中也算「重用過」，即使該 entry 本身從未被還原；`hits` 用於排序，統計上的「還原」請看引擎的 `restored` 計數器（不另存每個 entry 的還原次數，因為會改動 SSD manifest 格式）。
被取代／淘汰的淺 checkpoint 的 hits 隨之消失（只有最深者代表這段對話）。`superseded_flags` 用每 256 token 一塊的串接雜湊（每個 entry 只算一次並快取）比對前綴，
約 740 筆（30–100K token）在 -O2 下約 1.5 ms；一次性的雜湊約 0.2 ms／entry。

**superseded_durable（停止時的保護）**：`spill_meta` 每個 entry 多一欄 `superseded_durable`：與 `superseded` 相同，但「取代它的更深 entry」必須已在 SSD（有 manifest）；VRAM 池中的 checkpoint 與只在 RAM 的 entry 不算。`superseded` 只決定逐出**順序**；是否「直接丟棄而不寫入」（frontend 的 RAM→SSD 保留期、停止時的 `flush()`、引擎的 `spill_enforce_ram`）一律看 `superseded_durable`，所以停止時就算更深的那筆沒來得及寫，淺的也不會被略過而讓整段對話消失。`flush()` 另外在同一組內先寫較深的 entry。快取鍵為 `ret_gen_`（每次 entry／checkpoint 增減、owner／hits／manifest 寫入時遞增）。另外：RAM 不足（OOM）造成的讀檔失敗不計入 `io_fails`；成功讀取會把 `io_fails` 歸零（「連續 3 次」）。

**資料流**：`Checkpoint`／`Spilled` 增加 `owner`、`hits`、`last_hit`。SSD manifest 升為版本 2：標頭後多 16 位元組（owner、hits），舊的版本 1 manifest 仍可讀取與還原
（owner 0、hits 0）；新版寫出的 manifest 舊引擎讀不了（降級時會被當成「不是這個模型的檔案」而刪除）。引擎提供 `spill_meta`（每個 entry 的 owner／hits／superseded）、
`spill_agents`（每個 agent 在兩層的位元組與筆數）與 `set_owner`；沒有這些呼叫的舊引擎會讓 frontend 用預設值（owner 0、無 hits、無被取代），順序退回「最舊寫入先」。
**RAM → SSD**：超過保留時間的 entry 先寫「重用過」的，再依年齡；被取代者丟棄。

**狀態／UI**：`/stats` 的 tiers 區塊每層有重用過／從未重用／被取代的筆數與位元組（`ram_reused_*`、`ram_never_*`、`ram_superseded_*`、`disk_*`），重用率
`ram_reuse_rate` = 從 RAM 還原／溢出數、`ssd_reuse_rate` = 從 SSD 還原／寫出數（引擎計數器 `restored_ram`、`restored_disk`）；`agents` 區塊的 `ram_bytes`、`ssd_bytes`
（及 `ram_entries`、`ssd_entries`）有值，/ui 的 Agents 表與分層快取面板顯示。

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
- 報告也含 `VIDEO_MAX_TOKENS`：RAM 32 GiB 以上為預設的 131072，16 GiB 以上 32768，更少為檢查點自己的 12288；影片緩衝（每 token 16 KiB）會從 RAM 層的可用量扣除。
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

`tiered_cache_test.py` 有十二個情境（a–l），各自在獨立的行程中執行（環境變數在引擎建立時讀取）：

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
| k 保留策略 | 兩個 agent（`set_owner`）：一個有很多大的 entry（含被取代與從未重用者），一個只有幾個；RAM 層壓力（小的 `SPILL_RAM_GIB`）下 | 小 agent 的 entry 全數留在 RAM、大 agent 被逐出的是被取代／從未重用的；`spill_meta` 的 owner／hits／superseded 與 `spill_agents` 位元組一致；還原後 token 與參考相同 |
| l 影片輸入 | 文字前綴＋`bench/vision/colors.mp4`（循環取樣到約 9,000 token）＋問題，共約 12K token：A 問題一，prefill 存下 `Scheduler.checkpoints` 規劃的影片結尾檢查點與問題前最後一個 4096 倍數處的檢查點；B 同一支影片、問題二，從影片結尾檢查點續起（`cached` 不小於影片結尾向下對齊 64 的位置，影片不再解碼與編碼，最多重算最後一對）；C 的 prefill 把兩者逐出到 RAM；A 的續談從 RAM 還原（整支影片在快取前綴內）；RAM 的 entry 寫到 `SPILL_DIR`，B 的續談從檔案還原。需要視覺塔與 NVDEC，否則跳過；與其他情境一樣使用預設的 DSpine 草稿模型 | B 的 `cached` 不小於影片結尾（向下對齊 64）、`cache_stats.hits` 增加；`spilled`、`restored` 增加（RAM 與檔案各一次）；所有輸出 token 與關閉快取的參考完全相同（參考每次都重新解碼並編碼影片） |

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
