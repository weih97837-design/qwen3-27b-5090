# Windows 托盤程式與 /ui 設定頁（給人操作）

這份文件說明如何在 Windows 上用**托盤程式**（`qwen3-27b-tray.exe`）啟動與停止伺服器，以及如何在瀏覽器的 **/ui** 頁面調整引擎設定。
每一步都會說明你會看到什麼。

> **先知道的事**：本 fork 的 Windows 套件由 `.github/workflows/windows.yml` 自動建置。合併到 main 後，可在 GitHub → Actions → `windows` → 最新一次成功的執行 → Artifacts 下載 zip（需要登入）；打了 `v` 標籤之後則會出現在本 fork 的 Releases。下載與安裝步驟見 `docs/deploy-windows.md`，要自己建置則看第 1 節。
> 如果你只需要用 Docker 在 WSL2 裡跑，請改看 `docs/deploy-cli.md`，不需要這份文件。

---

## 0. 你需要的東西

- 一張 **RTX 5090**（32 GB）與較新的 NVIDIA 驅動（支援 CUDA 13）。
- Windows 64 位元。原生執行**不需要** WSL2 或 Docker，也不需要系統管理員權限（README）。
- 約 25 GB 的磁碟空間放權重（第一次啟動時自動下載，約 23 GB）。
- 至少 16 GB 主機 RAM。

---

## 1. 取得程式（自行建置）

建置需要在 Windows 上裝好下列工具（`scripts/windows/package.ps1` 的說明）。官方下載位置如下：

- **Visual Studio 2022 Build Tools**（或完整版 Visual Studio）：<https://visualstudio.microsoft.com/visual-cpp-build-tools/>。安裝時勾選 **「使用 C++ 的桌面開發」**（C++ 工作負載，含 MSVC 與 `vcvars64.bat`）。
- **CUDA Toolkit 13.3**（`nvcc`；13.x 系列即可，腳本的路徑為 13.3）：<https://developer.nvidia.com/cuda-downloads>，選 Windows、x86_64，使用預設安裝路徑。
- **CMake**：<https://cmake.org/download/>，選 Windows x64 安裝程式，並勾選 **Add CMake to the system PATH**。Visual Studio 的 C++ 工作負載通常也內含。
- **Ninja**：<https://github.com/ninja-build/ninja/releases>，下載 `ninja-win.zip`，解壓後把 `ninja.exe` 放進 PATH。Visual Studio 的 C++ 工作負載通常也內含。
- **Git for Windows**：<https://git-scm.com/download/win>（第 1 步的 `git clone` 需要）。
- **Python 3（含 pip）**：<https://www.python.org/downloads/>，安裝時勾選 **Add python.exe to PATH**。

**如何開啟 x64 Native Tools Command Prompt**：按 Windows 開始鍵，輸入 `x64 Native Tools`，在 **Visual Studio 2022** 資料夾中點選 **x64 Native Tools Command Prompt for VS 2022**。它會先執行 `vcvars64.bat`（設定 MSVC 編譯器的路徑），之後在該視窗執行的指令才找得到編譯器。

（以上網址為撰寫時的官方位置，撰寫時未能線上確認。）

**名詞說明**

- **SHA-256**：檔案的校驗碼，是由檔案內容算出的一串指紋。下載權重時，程式會比對每個檔案的 SHA-256 與官方記錄是否相同；不同代表下載不完整或損壞，需要重新下載。
- **HF_HOME**：Hugging Face 權重的存放資料夾（設定視窗的 **Weights folder (HF_HOME)**，見第 4 節）。權重放在其中的 `hub` 子資料夾，例如 `hub\models--juzbox--Qwen3.8-27B-NVFP4-GDN\`。
- **前綴快取（prefix cache）**：先前已算過的上下文（例如相同的系統提示或同一份文件）會被保留下來重複使用，後續請求不必從頭計算，回應較快。

步驟：

1. 取得原始碼（與 `docs/deploy-cli.md` 第 2 節相同）：

   ```bat
   git clone --branch tiered-kv-cache https://github.com/weih97837-design/qwen3-27b-5090 qwen3-27b-5090
   cd qwen3-27b-5090
   ```

2. 開啟 **x64 Native Tools Command Prompt**（或在 PowerShell 中執行 `vcvars64.bat`），然後設定 CUDA 路徑：

   ```bat
   set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3
   set CUDACXX=%CUDA_PATH%\bin\nvcc.exe
   set PATH=%CUDA_PATH%\bin;%PATH%
   ```

   （路徑來自 `.github/workflows/windows.yml` 的 `CUDA_DIR`。）

3. 執行打包腳本：

   ```bat
   powershell -ExecutionPolicy Bypass -File scripts\windows\package.ps1 -Python python
   ```

   - 這會先編譯引擎 `q27.dll`，再用 `cmake -S windows\tray -B windows\tray\build -G Ninja` 與 `cmake --build windows\tray\build` 編譯托盤程式（`scripts/windows/package.ps1:37-39`），最後組裝套件。
   - **你會看到**：一連串 CMake 與 Ninja 的輸出。第一次需要下載 Python 嵌入版與套件，時間較長。
   - **完成時**：`dist\` 內出現 `qwen3-27b-5090-<版本>-windows-x64.zip`（約 28 MB；版本來自 `git describe`，本分支沒有 `v*` 標籤時會是提交的雜湊值）。

- **失敗時**：`cmake` 找不到 CUDA 或 MSVC，請確認第 2 步的環境變數是在**同一個**命令視窗設定的。

---

## 2. 解壓縮並啟動托盤程式

1. 把 zip 解壓到一個**你有寫入權限**的資料夾（例如 `D:\Qwen27B\`）。**不要**放在 `Program Files`，因為程式會在資料夾內寫入 `settings.ini`、權重與記錄檔。
2. 進入解壓後的資料夾，**雙擊 `qwen3-27b-tray.exe`**。

**你會看到**：

- 右下角系統匣（通知區域）出現一個圖示。如果沒看到，點工作列的 **^**（隱藏的圖示）。
- 把滑鼠移到圖示上，提示文字會顯示 `Qwen3.8-27B: <狀態>`。
- 程式預設**不佔用 VRAM**，直到第一個請求到達時才載入模型（`ENGINE_AUTOSTART=0`）。閒置 30 分鐘後會自動釋放 VRAM（`ENGINE_IDLE_UNLOAD_S=1800`）。

**第一次啟動**會先下載權重（約 23 GB）到資料夾內的 `huggingface\`。提示文字會顯示下載進度，例如 `Downloading the weights: ...`。中斷後重新啟動會續傳。

要讓伺服器一啟動就載入模型：到 **Settings** 把 `ENGINE_AUTOSTART` 設為 1（見第 4 節），或改用 `start.cmd`（見第 6 節）。

---

## 3. 托盤選單

在圖示上**按右鍵**，選單內容如下（順序與文字依 `windows/tray/tray.cc:2014-2023`）：

| 選單項目 | 做什麼 | 什麼時候可用 |
| --- | --- | --- |
| 狀態列（灰色，不能點） | 顯示目前狀態 | 一律 |
| **Start server** / **Stop server** | 啟動或停止伺服器（切換） | 停止中或執行中可點；正在停止時灰色 |
| **Open stats page** | 開啟瀏覽器的 `/ui` 頁面 | 只有伺服器執行中可用 |
| **Settings...** | 開啟設定視窗（第 4 節） | 一律 |
| **Show log** | 開啟記錄視窗；伺服器沒啟動也能看 | 一律 |
| **Start at logon** | 打勾後開機登入時自動啟動托盤程式 | 一律（勾選即為開啟） |
| **Quit** | 結束程式，同時停止伺服器與引擎 | 一律 |

**狀態文字**（狀態列與提示文字）你會看到的範例：

| 顯示文字 | 意思 |
| --- | --- |
| `Stopped` | 伺服器已停止 |
| `Checking the weights` / `Downloading the weights` | 正在檢查或下載權重 |
| `Starting the server` | 伺服器啟動中 |
| `Waiting: the first request loads the model` | 伺服器就緒，模型尚未載入（等第一個請求） |
| `Loading the model` | 模型載入中 |
| `Ready` | 就緒，可以接受請求 |
| `Engine stopped from the stats page` | 在 /ui 頁面按了停止引擎 |
| `Not enough VRAM to load the model` | VRAM 不夠，載入被拒絕（見第 7 節） |

**記錄檔**：位於資料夾的 `logs\` 內，`server.log`（與較舊的 `server.1.log`）。在記錄視窗的選單中可以 **Open log folder** 直接開啟資料夾。

**Start at logon**：會在登錄檔 `HKCU\Software\Microsoft\Windows\CurrentVersion\Run` 寫入值 `qwen3-27b-5090`。取消勾選即移除。如果你搬移了資料夾，程式會在下次啟動時修正路徑。

---

## 4. 設定視窗（Settings...）

選單的 **Settings...** 會編輯資料夾內的 `settings.ini`，不需要開瀏覽器。

- **第一次**：不需要手動建立。按 **Apply** 或 **Open settings.ini** 時，托盤程式會自行建立 `settings.ini`。
- 視窗分為三組（群組與欄位名稱即視窗上的文字，對應變數在括號內）：
  - **Server: changing these restarts it**（改動會重新啟動伺服器）：**Address**（`HOST`，預設 `127.0.0.1`）、**Port**（`PORT`，預設 `46320`）、**Weights folder (HF_HOME)**（權重的資料夾）。
  - **Engine: what it loads**：**Weights**（`WEIGHTS`）、**KV cache type**（`KV_DTYPE`）、**Speculative decoding**（`SPEC`）、**Images (vision tower)**（`VISION`）、**Requests at once**（`MAX_SEQS`）、**Context limit (tokens)**（`MAX_TOKENS`）、**VRAM reserve (GiB)**（`VRAM_RESERVE_GIB`：至少保留不用的 VRAM，預設 1.5；預算 = 可用 VRAM 減去它。保留給桌面與其他程式，並避免 Windows WDDM 把 GPU 記憶體換到系統 RAM 而使推論大幅變慢；調低可讓 KV 池更大，每 1 GiB 約多 58K tokens（NVFP4 KV，18 KiB/token），但有被換出的風險）。
  - **When it holds the VRAM**：**Load the engine**（`ENGINE_AUTOSTART`）、**Unload when idle (seconds)**（`ENGINE_IDLE_UNLOAD_S`，閒置多久後釋放 VRAM）、**Smallest KV pool (tokens)**（`MIN_KV_TOKENS`）。
- 按鈕：**Browse...**（選擇權重資料夾）、**Open settings.ini**（用文字編輯器開啟檔案）、**Apply**（儲存並套用）、**Close**。
- **Apply** 會先告訴你會發生什麼。若會中斷進行中的請求或清空前綴快取，會先詢問。
  - **Apply 會重新啟動伺服器的情況**：只有 **Server: changing these restarts it** 群組（Address、Port、Weights folder）有變動、且伺服器執行中時，Apply 會重新啟動伺服器（訊息 `Saved. The server is restarting.`）。
  - **Engine 群組**（Weights、KV cache type 等）的變動：引擎已載入時會重新載入引擎（會中斷進行中的請求並清空前綴快取，所以會先詢問）；未載入時於下次載入時生效。
  - **`[env]` 區段的變數不在這個視窗中**，按 Apply 不會重新啟動伺服器，見第 4a 節。

**注意**：分層快取的設定（`SPILL_*`）不在視窗的欄位中。要改它們，按 **Open settings.ini** 直接編輯 `[env]` 區段，參見 `docs/tiered-cache.md`。

**要讓其他程式連線**，`Address` 維持 `127.0.0.1` 即可（同一台電腦上的用戶端）。

---

## 4a. 調整分層快取與上下文池

分層快取（RAM 與 SSD）與上下文池的設定不在設定視窗的欄位中。直接在 `settings.ini` 的 `[env]` 區段加入變數即可，**任何環境變數名稱都可以**（名稱不分大小寫）。`settings.ini` 的值優先於程式的環境變數。

```ini
[env]
KV_POOL_TOKENS=524288
MAX_SEQS=4
SPILL_RAM_GIB=32
SPILL_SSD_GIB=100
SPILL_DIR=D:\kvcache
SPILL_RAM_RETENTION_S=3600
SPILL_TTL_DAYS=7
```

- **預設值**：在原生 Windows 上，`frontend/launch.py` 會套用預設值：上下文池 512K tokens（`KV_POOL_TOKENS=524288`）、RAM 分層 32 GiB（`SPILL_RAM_GIB=32`），SSD 層位於 `%LOCALAPPDATA%\qwen3-27b-5090\kvcache`（`SPILL_DIR`）。只有要改的才需要寫入 `[env]`。
- **生效方式**：`[env]` 的變更不經過設定視窗的 **Apply**（Apply 只處理視窗內的欄位，不會重新啟動伺服器）。修改 `settings.ini` 後，在托盤選單按 **Stop server**，再按 **Start server** 才會套用（托盤選單沒有 Restart 項目；結束托盤程式再重新開啟也可以）。
- 其他變數的說明見 `docs/tiered-cache.md` 與 `docs/configuration.md`。

---

## 5. /ui 頁面：啟動、停止、重新啟動與設定

伺服器執行後，在瀏覽器開啟 **http://127.0.0.1:46320/ui**（選單的 **Open stats page** 也會開啟它）。

**你會看到**：

- 頁面上方的引擎區有三個按鈕：**Start**、**Stop**、**Restart**。
- 頁面標題列顯示目前引擎的啟動設定（引擎停止時，顯示下一次啟動會用的設定）。點它會開啟 **Start settings** 對話框。
- 頁面其他區塊：推理 Slot、GPU 與吞吐量、VRAM、KV cache 池、權重、**分層快取**、部署與硬體、請求，以及記錄。

**Start settings 對話框**（可調整的項目與 `docs/configuration.md#start-settings` 相同）：

| 欄位 | 對應變數 | 可選值 |
| --- | --- | --- |
| weights | `WEIGHTS` | `gdnfp4`、`nvidia`、`quasar`（已下載的）或服務啟動時的模型資料夾 |
| kv_dtype | `KV_DTYPE` | `nvfp4`、`fp8`、`bf16` |
| spec | `SPEC` | `dflash2`、`none` |
| vision | `VISION` | `host`、`fp8`、`bf16`、`off` |
| max_seqs | `MAX_SEQS` | 1 到 8 |
| max_tokens | `MAX_TOKENS` | 2,048 到 262,144 |
| vram_reserve_gib | `VRAM_RESERVE_GIB` | 留空（預設 1.5 GiB）或 0 到裝置記憶體的 GiB 數 |

對話框的按鈕：

- **Reset**：還原為目前的設定。
- **Save for the next load**：只記住這些設定，供下一次載入使用，**不會**載入引擎。
- **Start**：以這些設定啟動（或重新啟動）引擎。
- **Close**：關閉，不儲存。

**你會看到**：每個選項都會顯示它能提供的最長上下文與 KV 池大小，在載入前就能判斷是否放得下。

**注意**：

- **停止或重新啟動引擎會中斷進行中的請求，並清空前綴快取。** 重新啟動會重新載入權重。
- 在 /ui 做的改動**只在伺服器重新啟動前有效**。要永久保留，請把對應的變數寫入 `settings.ini`（頁面會列出要加入的行），或用第 4 節的設定視窗。

---

## 6. 不用托盤程式：start.cmd

如果你偏好主控台視窗，可以雙擊 `start.cmd`（同一個資料夾內）。

- **你會看到**：一個主控台視窗，裡面是下載（第一次）與伺服器的輸出；伺服器在 http://127.0.0.1:46320 監聽，瀏覽器會開啟 /ui（`OPEN_UI=1`）。
- **關閉視窗 = 停止伺服器**。
- `start.cmd` 會讀取 `settings.ini`，而且會直接載入引擎（`ENGINE_AUTOSTART` 預設為 1）。

---

## 7. 常見問題

| 你看到的 | 原因與處理 |
| --- | --- |
| 提示文字是 `Not enough VRAM to load the model` | VRAM 不夠。關閉其他使用 GPU 的程式（遊戲、其他 AI 工具），或在 Settings 降低 `MAX_TOKENS`、調低 `VRAM_RESERVE_GIB`。 |
| 視窗或記錄顯示 `Port 46320 is in use by another server.` | 46320 埠已被其他伺服器佔用（托盤程式不會啟動第二個）。在 Settings 改 **Port**，或先停止另一個伺服器。 |
| Windows 跳出安全性警告 | 程式未簽章，README 說明 Windows 可能會警告。自行建置的 zip 沒有上游 Release 的 SHA-256 可比對，請確認是你自己建置的檔案。 |
| `The weights could not be downloaded. The log has the reason.` | 下載失敗。看 **Show log**，中斷後重新啟動即可續傳。 |
| `The server exited with code N.` | 伺服器異常結束。看 **Show log** 的最後幾行。 |
| 托盤圖示沒有出現 | 看隱藏圖示區（**^**）。若程式已在執行，第二次啟動不會做任何事。 |
| 拔掉網路線或搬移資料夾後權重找不到 | 權重在資料夾的 `huggingface\` 內；搬移資料夾後請確認 `HF_HOME` 設定，或把整個資料夾一起搬。 |

---

## 8. 連接 OpenAI 相容用戶端

以下是人類操作的步驟。API 的細節與指令版本見 `docs/deploy-cli.md` 第 8 節。

1. **確認伺服器就緒**：在 /ui 頁面看到引擎狀態為可用（托盤提示為 `Ready`，或 `Waiting: the first request loads the model`）。
2. **用戶端設定**：Base URL 設為 `http://127.0.0.1:46320/v1`，API key 填任意值，模型名稱用 `http://127.0.0.1:46320/v1/models` 查到的名稱（預設 `qwen3.8-27b`）。
3. **並行**：用戶端同時送出的請求數最多為 **Requests at once**（`MAX_SEQS`，預設 4）時可並行執行；超過的請求排隊等候。
4. **確認**：送出一則請求得到回應；在 /ui 頁面可以看到請求的進行。

---

## 9. 更新

- `update.cmd` 查詢的 Release 來源是 `-Repo`，預設為本 fork `weih97837-design/qwen3-27b-5090`（`scripts/windows/update.ps1:32`）。本 fork 從 `v0.3.0-cache-tiered` 起有 Windows Release，`update.cmd` 會裝最新一版。查不到 Release 時（例如沒有網路），它會停止並顯示中英文說明；這時請依下列步驟，用自行建置的 zip 更新（`update.cmd -Zip <zip 路徑>` 也可以安裝它，但不檢查 SHA-256）。**不要**用 `-Repo mkyt/qwen3-27b-5090`：那會換成上游程式碼，不適用於本分支。
- 要更新本分支：
  1. 用托盤選單 **Quit** 結束程式。
  2. 在資料夾中刪除 `python\` 與 `app\`（舊版檔案不能留在新版旁邊，`docs/configuration.md#updating`）。
  3. 重新在 Windows 上執行第 1 節的建置，把新的 zip 解壓到同一資料夾（zip 的名稱含版本，資料夾內沒有版本）。
  4. 雙擊 `qwen3-27b-tray.exe`。

  `settings.ini`、`huggingface\` 權重與記錄檔會保留。
