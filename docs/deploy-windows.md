# Windows 原生部署（不需要 WSL、不需要 Docker）

這份文件說明如何直接在 Windows 上執行程式包：`start.cmd` 或托盤程式啟動伺服器，`test.cmd` 與 `bench.cmd` 執行分層 KV 快取的真機測試。
托盤與 /ui 設定頁的細節見 `docs/deploy-launcher.md`；分層快取的設定見 `docs/tiered-cache.md`；所有設定名稱見 `docs/configuration.md`（「Windows package」一節）。

> **未驗證**：本文件的 Windows 步驟是依照 `scripts/windows/`、`frontend/launch.py`、`scripts/download_weights.py` 等程式碼撰寫的，撰寫環境是 Linux，沒有在 Windows 實機上執行。文中標示「**未驗證**」的部分尚未確認。

---

## 1. 前置條件

- **NVIDIA RTX 5090**（32 GB VRAM）與支援 CUDA 13 的驅動程式。
- 64 位元 Windows。不需要系統管理員權限。
- **磁碟可用空間約 350 GB**：權重約 23 GB（第一次啟動時下載）；分層快取的 SSD 層預設上限 300 GiB（`SPILL_SSD_GIB=300`），並保留 10 GiB 可用空間（`SPILL_SSD_MIN_FREE_GIB`）；其餘是程式包與解壓縮所需的空間。磁碟較小就把 `SPILL_SSD_GIB` 調小；設成 `0` 則關閉 SSD 層，不佔磁碟（代價：引擎重新啟動後 RAM／VRAM 快取是冷的）。
- 主機 RAM：`docs/deploy-launcher.md` 建議至少 16 GB。RAM 層的 checkpoint 上限預設為 64 GiB（`SPILL_RAM_GIB`），那是上限而非必要用量。`bench.cmd` 預設使用 40 GiB（見第 7 節），**所需 RAM 未驗證**。
- **不需要**：WSL、Docker、Python、CUDA Toolkit、Visual Studio、PyTorch。程式包內含嵌入式 Python 與所需套件，只需要 NVIDIA 驅動程式。

---

## 2. 取得程式包

程式包由 `.github/workflows/windows.yml` 產生，兩種取得方式：

**A. 還沒有 `v*` 標籤時（目前這個分支的情況）**

1. 用瀏覽器登入 GitHub，開啟本 repo 的 **Actions** 頁面。
2. 點選 workflow **windows**。
3. 選擇 branch **main** 上最近一次**成功**（綠色勾勾）的執行。
4. 捲到頁面下方的 **Artifacts**，下載 `qwen3-27b-5090-<版本>-windows-x64`。下載需要登入 GitHub。
5. 下載的檔案是一層 zip，裡面有程式包 `qwen3-27b-5090-<版本>-windows-x64.zip` 與 `.sha256` 校驗檔。先解開外層 zip，再核對 SHA-256（見下）。

Artifact 保留 30 天。版本名稱來自 `git describe`，例如 `v0.3.0-3-g6767c0c`。

**B. 有 `v*` 標籤與 Release 之後**

到 repo 的 **Releases** 頁面，下載同一個 zip 與 `.sha256`。`update.cmd` 只會從 Release 更新；沒有 Release 時它會停止並顯示訊息。

**核對 SHA-256**（在 PowerShell 執行，把路徑換成你的 zip）：

```powershell
(Get-FileHash -Algorithm SHA256 .\qwen3-27b-5090-<版本>-windows-x64.zip).Hash.ToLower()
Get-Content .\qwen3-27b-5090-<版本>-windows-x64.zip.sha256
```

兩行的第一個欄位必須相同。**未驗證**（以上指令未在 Windows 上執行）。

---

## 3. 安裝

1. 把程式包解開到**只含 ASCII 字元**的路徑，例如 `C:\qwen3-27b-5090`。路徑中有中文或其他非 ASCII 字元時，請換一個路徑。
2. 不要解開到 `Program Files`：程式會在資料夾內寫入權重、記錄與 `settings.ini`，需要寫入權限。
3. 解開後的資料夾內容（依 `scripts/windows/package.ps1`）：

   | 檔案或資料夾 | 用途 |
   | --- | --- |
   | `start.cmd` | 在主控台視窗中下載權重並啟動伺服器 |
   | `qwen3-27b-tray.exe` | 托盤程式 |
   | `test.cmd`、`bench.cmd` | 真機測試（第 7 節） |
   | `update.cmd` | 從 Release 更新（第 2 節 B） |
   | `settings.example.ini` | 設定範本；需要改設定時複製成 `settings.ini` |
   | `python\` | 嵌入式 Python 與套件（不需要另外安裝 Python） |
   | `app\` | 伺服器、引擎（`app\engine\build\q27.dll`）與測試腳本 |

不複製 `settings.ini` 時，全部使用預設值。

---

## 4. 第一次啟動

**方法一：`start.cmd`（主控台）**

雙擊 `start.cmd`。視窗中會依序顯示：

1. 下載權重（第一次約 23 GB，由 `scripts\download_weights.py` 執行）。
2. 伺服器在同一個視窗中執行。
3. 瀏覽器自動開啟 http://127.0.0.1:46320/ui（設定 `OPEN_UI=0` 可關閉）。

停止伺服器：在該視窗按 **Ctrl+C**。下載中斷後重新執行會續傳。

**方法二：托盤程式**

雙擊 `qwen3-27b-tray.exe`。右下角（通知區域）會出現圖示；選單的 **Start server** 啟動伺服器。托盤預設不載入模型（`ENGINE_AUTOSTART=0`），第一個請求到達時才載入；閒置 30 分鐘後釋放 VRAM。細節見 `docs/deploy-launcher.md`。

**檔案放在哪裡**（依 `scripts/windows/start_package.py` 與 `frontend/launch.py`）：

| 內容 | 位置 |
| --- | --- |
| 權重（預設 `WEIGHTS=gdnfp4`） | `huggingface\hub\models--juzbox--Qwen3.8-27B-NVFP4-GDN\` |
| 草稿模型（DFlash2，`SPEC` 不是 `none` 時） | `huggingface\hub\models--incoai--Qwen3.8-27B-DFlash2\` |
| 分層快取的 SSD 層（預設開啟；`SPILL_SSD_GIB=0` 時不使用） | `%LOCALAPPDATA%\qwen3-27b-5090\kvcache` |
| 托盤程式的記錄 | `logs\server.log`（舊的為 `server.1.log`） |
| 設定 | `settings.ini`（資料夾內，需要時才建立） |

`huggingface\` 是預設的 `HF_HOME`，相對於程式包資料夾。只有 `settings.ini` 的 `HF_HOME` 能改它；環境變數中的 `HF_HOME` 會被忽略（設計如此，見 `start_package.py`）。

---

## 5. 狀態網頁

伺服器執行後，開啟 **http://127.0.0.1:46320/ui**。這個頁面顯示狀態、統計與引擎設定。管理 API 沒有驗證，預設只接受本機連線（`HOST=127.0.0.1`）。

---

## 6. 調整參數

設定寫在 `settings.ini` 的 `[env]` 區段，每行 `名稱 = 值`。Notepad 存檔的 UTF-8、UTF-16 與 ANSI 格式都可以。改完後重新啟動伺服器（托盤的設定視窗會替你做這件事）。

Windows 原生執行時的預設值（來自 `frontend/launch.py` 與 `frontend/tiercache.py`）：

| 名稱 | 預設 | 說明 |
| --- | --- | --- |
| `KV_POOL_TOKENS` | `0` | GPU 上的 KV 池（token 數）。`0` = 最大：VRAM 預算剩下的全部（預估約 60 萬 token 以上（實際大小以啟動 log 或監控頁為準））；明確設定的數字會覆蓋 |
| `SPILL_RAM_GIB` | `64` | 被逐出的 checkpoint 放在主機 RAM 的上限（GiB）；`0` 關閉分層 |
| `SPILL_DIR` | `%LOCALAPPDATA%\qwen3-27b-5090\kvcache` | SSD 層的資料夾（只在 `SPILL_SSD_GIB` > 0 時使用；SSD 層關閉時資料夾裡既有的檔案不會被讀取或刪除） |
| `SPILL_SSD_GIB` | `300` | SSD 層的上限（GiB），超過時刪除最舊的檔案。預設 300。**`0`：SSD 層關閉**，RAM 層滿時直接丟棄、不寫磁碟，引擎重新啟動後快取是冷的（沒有東西留下來）。可在 `settings.ini` 或托盤的 **SSD cap (GiB)** 設定；明確設過的值（例如 `200` 或 `0`）照舊有效 |
| `WEIGHTS` | `gdnfp4` | 權重版本 |
| `HOST` / `PORT` | `127.0.0.1` / `46320` | 監聽位址與埠 |
| `VIDEO_MAX_TOKENS` | `131072` | 一支影片最多的 token 數（檢查點自己的是 12288）。`IMAGE_TOKENS` 預設跟著它：每 token 10 KiB 的 pinned 主機記憶體，131072 時 1.25 GiB；編碼影片時另有每 token 6 KiB 的縮圖畫面（0.75 GiB）。RAM 32 GiB 以下建議改 `32768` |
| `VIDEO_MAX_FRAMES` | `2048` | 一支影片最多取幾張畫面（檢查點自己的是 768） |
| `SPILL_RAM_RETENTION_S` | `3600` | checkpoint 在 RAM 中保留的秒數，之後（閒置時）才寫入 SSD（只在 SSD 層開啟時有作用） |
| `SPILL_SSD_MIN_FREE_GIB` | `10` | SSD 所在磁碟保留的最小可用空間（GiB）（只在 SSD 層開啟時有作用） |
| `SPILL_SSD_MIN_TOKENS` | `32768` | 短於此 token 數的 checkpoint 不寫入 SSD（`0`：不設下限；只在 SSD 層開啟時有作用） |
| `SPILL_TTL_DAYS` | `7` | 未被重用的 SSD 檔案保留天數（只在 SSD 層開啟時有作用） |
| `VRAM_BACKUP_IDLE_S` | `60` | 引擎閒置多久後把 VRAM checkpoint 備份到 RAM（`0`：關閉） |
| `SPILL_PINNED_GIB` | `8` | pinned 記憶體（GiB）；`0` 關閉（v0.3.2 起預設 8） |
| `SPILL_ASYNC_IO` | `1` | 非同步 I/O；`0` 關閉（v0.3.2 起預設開啟） |
| `SPILL_ASYNC_PROMOTE` | `1` | 非同步還原（delay hit）；`0` 關閉（v0.3.2 起預設開啟） |
| `SPILL_FAIR_AGENTS` | `0` | 預設 `0`：RAM／SSD 層滿時整層一起排序（一位使用者的工作負載上，`0` 的 RAM 重用率約 30%、`1` 約 6%，早期數字）。`1`：先從用量最大的 agent 挑逐出對象（多個租戶共用同一個引擎時用） |
| `SPILL_COLD_FIRST` | `1` | 冷 agent（在該層沒有命中、最新 entry 超過 `SPILL_COLD_AFTER_S`）的 entry 先被逐出；`0` 關閉 |
| `SPILL_COLD_AFTER_S` | `600` | 冷 agent 的門檻（秒） |
| `SPILL_UNUSED_TTL_S` | `0` | 設為正數：從未命中的 entry 超過這麼多秒就從 RAM／SSD 丟棄；`0`（預設）關閉，不會因為沒被命中而丟棄 |
| `SPILL_SSD_WARM_ONLY` | `0` | `1`：從未命中且其 agent 也沒有命中的 entry 不寫入 SSD（沒有 id 的 `auto:` agent 直接丟棄，有 id 的受 `SPILL_PROBATION_GIB` 限制）；`0`（預設）：全部照常寫入（只在 SSD 層開啟時有作用） |
| `SPILL_PROBATION_GIB` | `4` | （只在 SSD 層開啟且 `SPILL_SSD_WARM_ONLY=1` 時有作用）這類「未證明」entry 在 SSD 上合計的容量上限（GiB） |

`SPILL_PINNED_GIB`、`SPILL_ASYNC_IO`、`SPILL_ASYNC_PROMOTE` 自 v0.3.2 起預設開啟；pinned 記憶體在 Windows 上計入 GPU 共用記憶體，記憶體吃緊時可設 `SPILL_PINNED_GIB=0`（見 `docs/tiered-cache.md`）。
`HOST=0.0.0.0` 會讓區網其他電腦連線；第一次啟動時 Windows 防火牆會詢問是否允許 Python，允許前請確認網路環境。

---

## 7. 真機測試

測試需要 VRAM，**先停止伺服器**（托盤選 **Stop server**，或在 `start.cmd` 的視窗按 Ctrl+C）。測試會自行載入引擎。

| 指令 | 做什麼 |
| --- | --- |
| `test.cmd` | 精確性測試：全部情境（a–l）執行 |
| `test.cmd --scenarios a,c` | 只執行指定情境 |
| `test.cmd --scenarios l` | 只執行情境 l（影片輸入：同一支影片的前綴命中、溢出到 RAM 與 SSD 後還原；需要視覺塔與 NVDEC） |
| `python\python.exe app\scripts\vision_api_test.py --base-url http://127.0.0.1:8080` | 對**執行中**的伺服器測試圖片與影片請求（見下）。埠換成 `settings.ini` 的 `PORT`（預設 `46320`；本文件的範例伺服器用 `8080`） |
| `bench.cmd` | 耗時：32K、128K、256K token 的 checkpoint |
| `bench.cmd --sizes 32768` | 只測指定大小 |
| `bench.cmd --compare "SPILL_PINNED_GIB=0" "SPILL_PINNED_GIB=8"` | 比較數個變數組合（每個引數是 `名稱=值`） |

兩個 `.cmd` 都在 `python\python.exe` 上執行 `app\scripts\` 內的腳本，結束時會暫停，視窗不會直接關閉；按任意鍵後才會關閉。

**權重從哪裡讀**：與 `start.cmd` 相同——`settings.ini` 的 `HF_HOME`，否則程式包內的 `huggingface\`。測試的預設 `--weights` 是 `settings.ini`（或環境變數）的 `WEIGHTS`，否則 `gdnfp4`。權重不在預期位置時，腳本會報錯 `... is not in ...hub`，請先確認 `start.cmd` 已下載完成。

**`test.cmd` 的 PASS 代表什麼**：每個情境的輸出 token（貪婪解碼）都與關閉快取的參考完全相同，而且所有檢查都通過，代表確實走了該情境的路徑（細節見 `docs/tiered-cache.md` 「真機測試」）。最後一行顯示 `all passed` 時結束碼為 0；任一情境 FAIL 時為 1。

**`vision_api_test.py`（圖片與影片 API 測試）**：與 `test.cmd` 相反，它需要伺服器**正在執行**（`start.cmd` 或托盤），透過 HTTP 送請求，不載入引擎、只用標準函式庫。在程式包資料夾的命令提示字元執行（不會自動暫停）：

```
python\python.exe app\scripts\vision_api_test.py --base-url http://127.0.0.1:8080
```

`--base-url` 是伺服器的位址與埠（預設 `http://127.0.0.1:46320`）。測試包含：圖表圖片的問答、同一對話第二回合命中前綴快取（圖片不再編碼）、換圖片不命中、兩張圖片、非資料 URL 的 HTTP 400，以及 `bench\vision\colors.mp4`（紅、綠、藍各 3 秒）：答案要依序說出三種顏色、換色約在 3 秒、圖片加影片同一則訊息、不是影片的位元組得到 HTTP 400。腳本最後以結束碼回報成功或失敗。引擎以 `VISION=off` 啟動時這些請求會被拒絕。

**情境 e（推測解碼）**：需要 DFlash2 草稿模型。`SPEC=none` 或草稿未下載時，該情境顯示 `SKIP`，這不是失敗。

**`bench.cmd` 的注意事項**：
- bench 與測試腳本用 `--ssd-gib`（預設 10，只用 10 GiB 的磁碟，與伺服器的 300 無關）明確設定 `SPILL_SSD_GIB`，設成 `0`（SSD 層關閉）會直接失敗。`test.cmd` 同樣有 `--ssd-gib`（預設 10），SSD 情境（b、c、f、g、h、i、j、k、l）在 SSD 關閉時會直接失敗，不會默默通過。
- SSD 檔案放在 `%LOCALAPPDATA%\qwen3-27b-5090\kvcache\bench`；腳本會刪除該資料夾內 `entries\`、`chunks\` 子資料夾的檔案，請不要把它指向其他資料。
- 每個尺寸都會建立兩個引擎，執行時間以分鐘計。耗時的實際數字**未驗證**。
- 預設 `SPILL_RAM_GIB=40`（測試為 2），所以 bench 在 RAM 中保留的 checkpoint 可達 40 GiB。
- 腳本只讀取 `settings.ini` 的 `HF_HOME` 與 `WEIGHTS`；其他 `SPILL_*` 設定由腳本自己決定，不會套用 `settings.ini` 的值。

---

## 8. 疑難排解

- **`The weights could not be downloaded.`**：網路中斷或 Hugging Face 無法連線。重新執行 `start.cmd`，下載會續傳。
- **`python\python.exe` 找不到**：zip 沒有完整解開。請把整個 zip 解開到一個資料夾，不要只執行 zip 內的檔案。
- **`Not enough VRAM to load the model`**：VRAM 不足，見 `docs/deploy-launcher.md` 第 7 節。執行 `test.cmd` 時，若伺服器仍在執行也會發生，先停止伺服器。
- **影片請求回 HTTP 400，訊息含 `NVDEC (libnvcuvid) is not available`**：影片由 NVIDIA 驅動程式的 `nvcuvid.dll`（在 `C:\Windows\System32`）解碼，程式包不含它。安裝或更新 NVIDIA 驅動程式（Game Ready 或 Studio 皆可；訊息為 `lacks entry points` 表示驅動程式太舊）。
- **測試顯示無法載入 q27.dll（`Could not find module` 之類的錯誤）**：確認 `app\engine\build\q27.dll` 存在。
- **托盤程式被 Windows SmartScreen 攔截**：程式未簽章（`README.md` 說明），需確認來源後再執行。嵌入式 Python 由 python.org 簽章。
- **`is not in ...hub`**：權重不在測試使用的 `HF_HOME`。檢查 `settings.ini` 的 `HF_HOME`，並確認 `start.cmd` 的下載已完成。
- **路徑含非 ASCII 字元**：換到 ASCII 路徑（第 3 節）。

---

## 未驗證的項目（一覽）

- 第 2 節 A 的 Actions 與 Artifacts 操作、SHA-256 核對指令。
- 第 4 節的啟動流程與檔案位置（由程式碼推得，未在 Windows 上執行）。
- 第 7 節的 `test.cmd` / `bench.cmd` 實際執行結果、耗時與所需 RAM。
- 影片輸入在原生 Windows 的實際解碼（`nvcuvid.dll` 的載入、WDDM 下 KV 池借出給 NVDEC）、情境 l 與 `vision_api_test.py` 的執行結果：程式碼有 Windows 分支，CI 只驗證 MSVC 編譯與不需 GPU 的 MP4 探測。
- 第 1 節的約 350 GB 與 16 GB RAM 建議（由設定預設值推算）。
