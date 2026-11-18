# 在一張 RTX 5090 上跑 Qwen3.8-27B 的推論伺服器（繁體中文簡介）

這是一個在自己電腦上執行的 AI 語言模型伺服器。它在一張 NVIDIA RTX 5090（32 GB）上執行 [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)，
提供 OpenAI 與 Anthropic 相容的 API，可以輸入文字、圖片與影片。Claude Code、Codex 這類程式碼代理（agent）工具可以把它當成後端。

英文完整說明在 [README.md](README.md)。本文是給朋友看的簡介與快速開始。

## 這個 repo 是什麼

本 repo 是 [mkyt/qwen3-27b-5090](https://github.com/mkyt/qwen3-27b-5090) 的 **fork**，授權為 MIT（[LICENSE](LICENSE)，保留原作者版權聲明）。

**上游（mkyt）提供的：**

- 引擎本體：從頭用 C++ / CUDA 寫的推論引擎與 GPU 核心（kernel），只針對這個模型與 RTX 5090 調整。
- 推測解碼（speculative decoding），含 DSpine 草稿模型支援。
- OpenAI / Anthropic 相容的 HTTP 伺服器、圖片輸入、前綴快取。
- 速度與長上下文的測量數字（在 400 W 功率限制下量測）：見 [README.md](README.md) 與 [docs/benchmarks.md](docs/benchmarks.md)。本文不重複這些數字。

**這個 fork 加上的：**

| 項目 | 白話說明 |
| --- | --- |
| 分層快取 | KV 快取放不下 VRAM 時，依序放到 RAM、再到 SSD（VRAM → RAM → SSD），回來時不必重算。RAM 與 SSD 層會去除重複內容；引擎閒置時先備份；空間不足時依 Agent 公平淘汰（用量最大的先被清）；從未被重用過的先於重用過的被清。細節：[docs/tiered-cache.md](docs/tiered-cache.md)（繁體中文） |
| 多機器計數 | 多台電腦共用一台伺服器時，請求可帶 `X-Client-Id`（或伺服器設定 `CLIENT_NAMES` 對照 IP），監控頁看得到誰在用。**只計數，不限制**，排隊方式不變 |
| Agent 識別 | 請求可帶 `X-Agent-Id`，監控頁與快取淘汰以此分辨不同的代理 |
| 影片輸入整合 | 影片設定可調（預設一支影片最多 131,072 token、2,048 張畫面）；影片結尾存檔點，同一支影片換問題時不重算影片 |
| 繁體中文監控頁 | `/ui`，有針對窄螢幕的版面（手機排版） |
| 不同的預設值 | 同時 8 個 slot（`MAX_SEQS=8`）、KV 池自動取最大（`KV_POOL_TOKENS=0`）、VRAM 預留 1 GiB（`VRAM_RESERVE_GIB=1.0`）、RAM 層 64 GiB（上限）、SSD 層上限預設 100 GiB |
| Windows 原生程式包 | 不需要 WSL 或 Docker；`start.cmd`、托盤程式與 `update.cmd` 更新 |
| 托盤程式設定視窗 | 在 Windows 托盤選單的 **Settings...** 改設定，不必手動編輯檔案 |

**版本命名：** 這個 fork 的版本是「上游版本.修訂-cache-tier」。例如 `v0.6.2.1-cache-tier` 是以上游 `v0.6.2` 為基礎的第 1 個修訂。

> Windows 程式包的安裝、更新與分層快取測試（`test.cmd`）已在 RTX 5090 的 Windows 電腦上實機跑過；文件裡個別標示「未驗證」的步驟是較早期的寫法，見 [docs/deploy-windows.md](docs/deploy-windows.md)。

## 需求

| 項目 | 需求 |
| --- | --- |
| GPU | RTX 5090（32 GB，SM120）。其他 GPU 不支援 |
| 驅動 | 支援 CUDA 13 的 NVIDIA 驅動（上游以 617.42 測試） |
| 作業系統 | Windows 10/11（64 位元，原生，不需要系統管理員權限）；或 Linux / WSL2 加 Docker 與 NVIDIA Container Toolkit |
| 網路 | 第一次啟動會從 Hugging Face 下載權重 |

**磁碟**

- 權重約 23 GB（預設的 gdnfp4 權重約 19.5 GB，加上草稿模型約 3.5–4.1 GB）。見 [README.md](README.md) 與 [docs/deploy-cli.md](docs/deploy-cli.md)。
- SSD 快取預設最多再用 100 GiB（`SPILL_SSD_GIB`），並保留 10 GiB 以上的可用空間（`SPILL_SSD_MIN_FREE_GIB`）；資料夾在 `%LOCALAPPDATA%\qwen3-27b-5090\kvcache`。
- [docs/deploy-windows.md](docs/deploy-windows.md) 估計 Windows 上約需 150 GB 可用空間（文件自己標示為由預設值推算）。不想佔這麼多，把 `SPILL_SSD_GIB` 調小。
- Windows 程式包約 35 MB（不含權重）。把它解壓到有寫入權限、路徑只有 ASCII 的資料夾（不要放 `Program Files`）。

**RAM**

| 項目 | 數字 |
| --- | --- |
| 最低建議 | 16 GB（要把下面列出的設定調低）；用預設設定，建議 32 GB 以上 |
| 分層快取的 RAM 層 | 預設 `SPILL_RAM_GIB=64`。這是**上限**，不是必要用量，用多少佔多少 |
| pinned（鎖定）記憶體，預設值 | `STATE_HOST_CACHE` 8 GiB、`SPILL_PINNED_GIB` 8 GiB、視覺塔 0.86 GiB、圖片影片嵌入表 1.25 GiB（`IMAGE_TOKENS` 131,072 時） |
| Windows 的 pinned 上限 | pinned 記憶體計入 GPU 共用記憶體，約為 RAM 的一半，上述四項加起來要留在其中。四項預設值相加約 18 GiB（預估，由文件數字相加） |

RAM 不多的機器（32 GiB 以下）請調低這些設定（設定方法見下一節）：

| 設定 | 調整 |
| --- | --- |
| `SPILL_RAM_GIB` | 調小，例如 16 或 32；`0` 關閉分層快取 |
| `STATE_HOST_CACHE` | 預設 8（GiB）；調小，`0` 關閉前綴快取 |
| `SPILL_PINNED_GIB` | 預設 8；吃緊時設 `0` |
| `IMAGE_TOKENS` / `VIDEO_MAX_TOKENS` | RAM 32 GiB 以下建議 `VIDEO_MAX_TOKENS=32768`（`IMAGE_TOKENS` 會跟著，嵌入表約 0.31 GiB） |

## 快速開始（Windows 程式包）

1. 到 [Releases](https://github.com/weih97837-design/qwen3-27b-5090/releases/latest) 下載 `qwen3-27b-5090-<版本>-windows-x64.zip` 與同名的 `.sha256`。
2. 核對 SHA-256（PowerShell），兩行的第一個欄位必須相同：

   ```powershell
   (Get-FileHash -Algorithm SHA256 .\qwen3-27b-5090-<版本>-windows-x64.zip).Hash.ToLower()
   Get-Content .\qwen3-27b-5090-<版本>-windows-x64.zip.sha256
   ```

3. 解壓縮到只含 ASCII 字元的路徑，例如 `C:\qwen3-27b-5090`。
4. 啟動，二選一：
   - 雙擊 `start.cmd`：在主控台視窗執行，會立刻載入模型；關閉視窗或按 Ctrl+C 即停止。
   - 雙擊 `qwen3-27b-tray.exe`：托盤程式，右下角出現圖示。預設要到第一個請求才載入模型（不佔 VRAM），閒置 30 分鐘後釋放 VRAM。
5. 第一次啟動會下載權重到資料夾內的 `huggingface\`（約 23 GB，中斷後重新啟動會續傳）。
6. 完成後開啟監控頁 <http://127.0.0.1:46320/ui>（`start.cmd` 會自動開啟瀏覽器；托盤選單的 **Open stats page** 也可以）。
   預設埠是 `46320`；要改，在 `settings.ini` 的 `[env]` 寫 `PORT=<埠號>`，或用托盤 **Settings...** 的 **Port**。

程式未簽章，Windows 第一次可能跳出安全性警告（[README.md](README.md)）。

**記錄檔：** 托盤程式的記錄在資料夾的 `logs\server.log`（較舊的是 `server.1.log`），托盤選單 **Show log** 可開啟。`start.cmd` 的輸出就在它的視窗裡。

**更新：** 在程式包資料夾執行 `update.cmd`。它會下載最新 Release、核對 SHA-256、結束托盤程式、替換程式，並保留 `settings.ini`、權重與記錄。

| 指令 | 作用 |
| --- | --- |
| `update.cmd` | 更新到最新 Release |
| `update.cmd -Check` | 只顯示目前版本與最新版本 |
| `update.cmd -Tag <版本>` | 裝指定版本，也可以退回舊版 |

用 `start.cmd` 啟動的伺服器要自己先關掉。如果從 PowerShell 7 啟動 `update.cmd` 卻無法核對雜湊，改在 `cmd.exe` 執行，先打 `set PSModulePath=` 再執行 `update.cmd`
（`update.cmd` 本身已經清掉這個變數，這是額外的保險；做法見 `scripts/windows/update.cmd` 與 `update.ps1` 的註解）。

**Docker / Linux：** 見 [README.md](README.md) 的 Quick start 與命令列版 [docs/deploy-cli.md](docs/deploy-cli.md)（繁體中文）。簡短版：

```bash
scripts/download-weights.sh
docker compose up -d --build serve
docker compose logs -f serve       # 等到出現 "engine ready"
```

## 設定

設定是變數（名稱全部大寫）。三個地方可以改：

- **托盤程式的設定視窗：** 托盤選單 **Settings...**，按 **Apply** 套用。
- **網頁 `/ui` 的「啟動設定」：** 只在伺服器重新啟動前有效。頁面會列出要寫進檔案的那幾行。
- **`settings.ini` 的 `[env]` 區段：** 每行 `名稱=值`。把 `settings.example.ini` 複製成 `settings.ini`。Docker 版則是 `.env`。

`settings.ini` 或托盤明確設定的值，會覆蓋程式的預設值（例如 `MAX_SEQS`、`KV_POOL_TOKENS`）。沒寫就用預設值。
視窗沒有的欄位，改完 `settings.ini` 後在托盤選單先 **Stop server** 再 **Start server**。

| 變數 | 預設 | 意思 |
| --- | --- | --- |
| `MAX_SEQS` | `8` | 同時執行的請求數（slot 數），1 到 8 |
| `KV_POOL_TOKENS` | `0` | GPU 上 KV 池的大小（token）。`0` 是自動取最大（預估約 60 萬 token 以上，實際數字看啟動記錄或監控頁）；填數字則固定 |
| `VRAM_RESERVE_GIB` | `1.0` | 至少留著不用的 VRAM（GiB）。若這張顯卡也接螢幕或有別的程式用 GPU，文件建議調到 1.5–2，避免 Windows 把 GPU 記憶體換到 RAM 而變慢 |
| `SPEC` | `dspine` | 推測解碼：`dspine`、`dflash2` 或 `none`（見下） |
| `SPILL_RAM_GIB` | `64` | RAM 層上限（GiB）；`0` 關閉分層快取 |
| `SPILL_SSD_GIB` | `100` | SSD 層上限（GiB），超過時先刪最舊的 |
| `SPILL_SSD_MIN_FREE_GIB` | `10` | SSD 所在磁碟至少保留的可用空間（GiB） |
| `VIDEO_MAX_TOKENS` | `131072` | 一支影片最多的 token 數（模型檢查點自己的值是 12,288） |
| `VIDEO_MAX_FRAMES` | `2048` | 一支影片最多取幾張畫面（檢查點自己的值是 768） |
| `CLIENT_NAMES` | 空 | 把來源 IP 對應成機器名稱，例如 `192.168.1.11=PC-A,192.168.1.12=PC-B` |
| `HOST` | `127.0.0.1` | 預設只接受本機連線。`0.0.0.0` 讓區網的其他電腦連線（**沒有登入驗證**，只在信任的網路使用，不要把埠轉到網際網路） |

**`SPEC` 怎麼選：** 一次只跑一個請求，`dspine`（預設）比較快。同時跑很多請求（四個或八個）且內容是長篇文字或推理時，`dflash2` 稍快（在文件的測試中快約 3–6%）。
依據在 [docs/benchmarks.md](docs/benchmarks.md)。

完整的設定清單：[docs/configuration.md](docs/configuration.md)（英文）。托盤設定視窗的每個欄位：[docs/deploy-launcher.md](docs/deploy-launcher.md)。
`tools/recommend_config.py` 會依機器的 GPU、RAM 與磁碟建議參數；它在 Linux / WSL 主機或 Docker 開發容器執行，Windows 程式包沒有附它（[docs/tiered-cache.md](docs/tiered-cache.md) 的「建議配置工具」）。

## 使用

- **OpenAI 相容：** base URL `http://127.0.0.1:46320/v1`，API key 填任意值，模型名稱 `qwen3.8-27b`。
- **Anthropic 相容**（例如 Claude Code）：`ANTHROPIC_BASE_URL=http://127.0.0.1:46320`、`ANTHROPIC_MODEL=qwen3.8-27b`。
- **Codex：** 供應商設 `base_url = "http://127.0.0.1:46320/v1"`、`wire_api = "responses"`。
- 其他電腦連線時，把 `127.0.0.1` 換成伺服器電腦的位址（先設 `HOST=0.0.0.0`，見上表）。

文字：

```bash
curl -s http://127.0.0.1:46320/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model": "qwen3.8-27b", "messages": [{"role": "user", "content": "Hello!"}]}'
```

影片（`video_url`，來源是 `data:` 資料 URL 或 http(s) 網址；一支最多 256 MB；MP4 / MOV、Matroska / WebM；H.264、HEVC、AV1、VP9）：

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

把這段 JSON 用 POST 送到 `/v1/chat/completions`。影片預設每秒取 2 張畫面；超過 10 分鐘的影片請先剪短，或把 `fps` 調低。其他細節見 [docs/agent-client-guide.md](docs/agent-client-guide.md)。

**兩個建議帶的標頭**（兩者都是標籤，不是登入驗證）：

| 標頭 | 用途 |
| --- | --- |
| `X-Agent-Id: <名稱>` | 代理的身分。監控頁的 Agents 表依它分組，RAM／SSD 快取滿時依它公平淘汰。名稱要固定；子任務寫成 `名稱/子id`（以第一個 `/` 之前為代理） |
| `X-Client-Id: <機器名稱>` | 機器名稱，監控頁的「機器」表依它計數。最長 64 字元，建議用英數字。沒帶時用來源 IP（或 `CLIENT_NAMES` 對照的名稱） |

伺服器沒有 API key，也不限制並行或速率，多的請求只是排隊。限制並行是各用戶端自己的責任。詳見 [docs/agent-client-guide.md](docs/agent-client-guide.md)（繁體中文）。

**監控頁 `/ui`**（繁體中文，手機瀏覽器也開得起來，要先讓伺服器能被區網連到）：推論 Slot 與每個請求的 tok/s、GPU 與 VRAM、KV 快取池、分層快取、Agent 表、機器表、進行中的請求、紀錄。
也可以在這裡啟動、停止、重新啟動引擎，以及改啟動設定。**停止或重啟引擎會中斷進行中的請求並清空前綴快取。**

`/metrics` 提供 Prometheus 格式的指標。

## 疑難排解

| 狀況 | 做法 |
| --- | --- |
| `update.cmd` 無法核對雜湊 | 從 PowerShell 7 啟動時，改在 `cmd.exe` 先執行 `set PSModulePath=` 再 `update.cmd`。沒有 Release 或沒有網路時它會停止並顯示訊息 |
| `Port 46320 is in use by another server.` | 埠被別的程式佔用。在托盤 **Settings...** 改 **Port**，或在 `settings.ini` 寫 `PORT=<別的埠>`，或先關掉另一個伺服器 |
| 影片請求回 HTTP 400，訊息含 `NVDEC (libnvcuvid) is not available` | 影片由 NVIDIA 驅動程式的 `nvcuvid.dll` 解碼，程式包不含它。安裝或更新 NVIDIA 驅動程式（訊息為 `lacks entry points` 代表驅動太舊） |
| `Not enough VRAM to load the model`，或 KV 池比預期小 | 關閉其他用 GPU 的程式。引擎只在載入時看一次可用 VRAM。可調高 `VRAM_RESERVE_GIB`（避免被換出）、調低 `MAX_SEQS` 或 `MAX_TOKENS`。先停止引擎再開別的 GPU 程式。見 [docs/configuration.md](docs/configuration.md) 的「Too little VRAM」 |
| 權重下載中斷（`The weights could not be downloaded.`） | 看記錄找原因，重新執行 `start.cmd` 或重啟托盤程式，下載會續傳 |
| Windows 安全性警告 | 程式未簽章，確認下載來源與 SHA-256 後再執行 |
| 路徑含中文，或 `python\python.exe` 找不到 | 重新解壓縮到 ASCII 路徑，並把整個 zip 完整解開 |

更多見 [docs/deploy-windows.md](docs/deploy-windows.md) 第 8 節與 [docs/deploy-launcher.md](docs/deploy-launcher.md) 第 7 節。

## 文件索引

| 文件 | 內容 | 語言 |
| --- | --- | --- |
| [README.md](README.md) | 完整說明、速度與長上下文數字、API 功能 | 英文 |
| [docs/deploy-windows.md](docs/deploy-windows.md) | Windows 原生部署、真機測試 | 繁體中文 |
| [docs/deploy-launcher.md](docs/deploy-launcher.md) | 托盤程式與 `/ui` 設定頁的操作步驟 | 繁體中文 |
| [docs/deploy-cli.md](docs/deploy-cli.md) | Docker / 命令列部署 | 繁體中文 |
| [docs/tiered-cache.md](docs/tiered-cache.md) | 分層快取、保留策略、影片輸入、Agent 識別 | 繁體中文 |
| [docs/agent-client-guide.md](docs/agent-client-guide.md) | 給呼叫伺服器的程式：標頭、前綴穩定、影片 | 繁體中文 |
| [docs/configuration.md](docs/configuration.md) | 所有設定 | 英文 |
| [docs/benchmarks.md](docs/benchmarks.md) | 效能測量 | 英文 |
| [docs/architecture.md](docs/architecture.md)、[docs/decisions.md](docs/decisions.md) | 設計與取捨 | 英文 |
| [docs/releases/](docs/releases/) | 本 fork 較早版本的說明；新版本的說明看 [Releases 頁面](https://github.com/weih97837-design/qwen3-27b-5090/releases) | 繁體中文 |

## 測試

Windows 程式包的真機測試在資料夾內：先停止伺服器，再執行 `test.cmd`（分層快取的精確性測試，每個情境的輸出 token 必須與關閉快取的參考完全相同）。
`test.cmd --scenarios a,c` 只跑指定情境；`bench.cmd` 量測耗時。細節見 [docs/deploy-windows.md](docs/deploy-windows.md) 第 7 節。

開發者的單元測試（需要 Python 與 pytest）：`python -m pytest frontend/tests tools/tests -q`。

## 授權與致謝

- 程式碼：MIT（[LICENSE](LICENSE)），版權屬原作者 Masahiro Kiyota（mkyt），本 fork 保留該聲明。
- 模型權重（Qwen3.8-27B、NVIDIA 的 NVFP4 檢查點、DSpine、DFlash2、gdnfp4 版本）是 Apache-2.0，從 Hugging Face 下載，**不隨本 repo 或安裝包散布**；其他檢查點有各自的授權。
- Windows 程式包內含 Python 與其套件，各有自己的授權（在程式包裡）。
- 上游作者：Masahiro Kiyota（mkyt）。第三方：[DSpine](https://huggingface.co/tencent/Qwen3.8-27B-DSpine-b16)（tencent）、NVIDIA nv-codec-headers、libwebm。
