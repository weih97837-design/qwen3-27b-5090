# 給代理程式開發者：呼叫這個伺服器的指南

這份文件給「寫代理（agent）程式、呼叫本伺服器」的開發者。伺服器相容 OpenAI（`/v1/chat/completions`、`/v1/responses`）與 Anthropic（`/v1/messages`）格式，模型名稱 `qwen3.8-27b`（請求可以填任何名稱）。要讓分層 KV 快取（VRAM → RAM → SSD，見 `docs/tiered-cache.md`）發揮作用，用戶端需要做到三件事：**報上自己的身分**、**讓提示詞前綴位元組穩定**、**正確地送影片**。

位址：Docker 與 Windows 程式包預設 `http://127.0.0.1:46320`（Windows 的 `PORT`、Docker 的 `PORT` 可改）。下面的例子用這個位址。

## 1. 報上代理身分（X-Agent-Id）

伺服器用代理身分決定：統計頁「Agents」表的歸屬，以及 RAM／SSD 層滿時從誰那裡逐出（用量最大的代理先被逐出，小的代理不會被大的擠掉；`SPILL_FAIR_AGENTS`）。契約（細節見 `docs/tiered-cache.md` 「代理識別」）：

- HTTP 標頭 `X-Agent-Id: <id>`（優先），或 OpenAI 請求本文的 `"user": "<id>"`；兩者都有時以標頭為準。
- id 最長 128 字元，允許 `A-Za-z0-9._/-`；其他字元換成 `_` 並截斷；空字串視為沒有提供。
- 公平性的代理 = 第一個 `/` 之前的部分。`manager/task-123` 與 `manager/probe` 都是代理 `manager`；完整 id 也會記在記錄裡，方便追蹤是哪個子任務。
- 沒提供時，伺服器用提示詞開頭（系統提示詞加第一則使用者訊息）的雜湊產生 `auto:<8 位十六進位>`；開頭相同的不同代理會被當成同一個。

建議：

- 每個**獨立的代理或服務**用一個固定的名稱（例如 `coder`、`reviewer`、`nightly-report`），子任務寫成 `名稱/子id`。名稱要**穩定**：重啟後同一個名稱仍對應同一批 SSD 上的 entry。
- 不要把每個請求都當成新代理（隨機 UUID 放在 `/` 之前）：那會讓每個請求在統計與逐出上都是「新來的小代理」，失去公平性的意義。把 UUID 放在 `/` 之後。
- 範例：

```bash
curl http://127.0.0.1:46320/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'X-Agent-Id: reviewer/pr-1842' \
  -d '{"model":"qwen3.8-27b","messages":[{"role":"user","content":"hi"}],"max_tokens":16}'
```

OpenAI SDK 以 `default_headers={"X-Agent-Id": "reviewer/pr-1842"}` 設定，或在請求加 `user="reviewer/pr-1842"`。

## 2. 讓提示詞前綴位元組穩定

前綴快取比對的是**token id 序列的最長共同前綴**。你的請求只要和上一個請求在某個位置開始不同，那個位置之後的所有東西都要重新計算。代理的提示詞動輒幾萬 token，所以這是最大的效能因素。

**不要做：**

- 在系統提示詞或早期訊息裡放會變的東西：時間戳記、今天的日期、隨機 id、請求編號、目前工作目錄之類每次不同的字串。要放就放在**最後一則使用者訊息的末尾**。
- 每次重新排序工具（`tools`）：工具清單與 JSON schema 的序列化要固定（固定順序、固定的鍵順序）。工具定義在提示詞最前面，它變動會使整段前綴失效。
- 修改、重排、刪減或「整理」歷史訊息（例如把舊的工具輸出截短、改寫早期的使用者訊息、重新縮排 JSON）。要壓縮歷史時，接受一次完整重算，然後在新的基礎上繼續只往後加。
- 改動空白（修剪行尾空白、換行 `\n` 與 `\r\n` 互換、全形半形轉換）。伺服器依 token 計，一個空白不同就不同。
- 讓兩個請求用不同的 `enable_thinking`／`reasoning_effort` 之類的模板參數：它們會改變渲染出來的提示詞。

**要做：**

- **只往後附加（append-only）**：每一回合的 `messages` = 上一回合的 `messages` ＋ 助理的回覆 ＋ 新的使用者／工具訊息。
- **原樣送回助理的回覆**：包括 `content`、`tool_calls`（`arguments` 照伺服器回的字串，不要重新序列化）與推理（回應的 `reasoning_content`，也可以叫 `reasoning`；聊天模板會渲染前幾回合的推理，前綴快取只有在你把它帶回去時才命中）。串流時把片段原樣串起來，不要 `strip()`。
- 工具結果（`role: "tool"`）照原樣放，不要重新格式化。
- 同一個代理的並行子任務若共用長前綴（系統提示詞＋工具＋共同的上下文），讓它們**位元組相同**。伺服器會讓後到的請求等先到的請求算過這段前綴（並在那裡存檢查點），而不是各算一遍。

**怎麼確認有沒有命中：**

- 回應的 `usage.prompt_tokens_details.cached_tokens`（OpenAI 格式；Responses 格式是 `input_tokens_details.cached_tokens`）是這個請求從快取接上的 token 數。多回合對話的第二回合起，它應該幾乎等於上一回合的長度。
- 統計頁（`/ui`）的 Log 表有 `Cached (hit)` 欄；若請求偏離了先前的某段序列，該欄會有 `≠`，伺服器記錄也會寫出 `diverges from its best earlier sequence at token N` 與前後的文字：那就是你的前綴在哪裡變了。

## 3. 影片輸入（video_url）

### 怎麼送

OpenAI 格式的 `content` 陣列放 `video_url` 部分（與 vLLM 相同；`{"type":"video","video": url}` 也接受）。來源是 `data:video/mp4;base64,...` 資料 URL，或 `http(s)` URL（由伺服器下載，逾時 120 秒）：

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

Python（標準函式庫）：

```python
import base64, json, urllib.request

raw = open("clip.mp4", "rb").read()
url = "data:video/mp4;base64," + base64.b64encode(raw).decode()
body = {"model": "qwen3.8-27b", "max_tokens": 512,
        "messages": [{"role": "user", "content": [
            {"type": "video_url", "video_url": {"url": url}},
            {"type": "text", "text": "影片裡有哪些顏色？"}]}],
        "mm_processor_kwargs": {"fps": 1}}
req = urllib.request.Request("http://127.0.0.1:46320/v1/chat/completions", json.dumps(body).encode(),
                             {"Content-Type": "application/json", "X-Agent-Id": "demo/video"})
print(json.load(urllib.request.urlopen(req))["choices"][0]["message"]["content"])
```

- 容器與編碼：MP4／MOV、Matroska／WebM；H.264、HEVC、AV1、VP9。手機影片的旋轉會套用。
- 上限：一支影片 256 MB（超過回 HTTP 400）。一個請求所有圖片與影片的 token 合計不得超過 `IMAGE_TOKENS`（預設 131,072），否則 HTTP 400；影片加文字還要放得進上下文（`MAX_TOKENS`，預設 262,144）。
- `mm_processor_kwargs`（在請求本文最上層；`/v1/chat/completions` 與 `/v1/responses` 有，`/v1/messages` 沒有）：`fps`（大於 0、至多 60）、`num_frames`（與 `fps` 互斥）、`min_frames`、`max_frames`（1 到伺服器的 `VIDEO_MAX_FRAMES`，預設 2048）、`size.longest_edge`（這支影片所有畫面合計的像素上限，每 token 2,048 像素）。壞值回 HTTP 400。
- 引擎以 `VISION=off` 啟動時帶影片的請求回 HTTP 400。

### 長度怎麼變成 token（預設 T＝131,072 token、F＝2,048 張）

| 影片（1080p 或 4K，16:9） | 2 fps（預設） | 1 fps（`"fps": 1`） |
| --- | --- | --- |
| 10 秒 | 7,200 token | 3,600 token |
| 60 秒 | 43,200 token | 21,600 token |
| 3 分鐘 | 126,000 token（每對 700 token，畫面開始變小） | 64,800 token |
| 10 分鐘 | 125,400 token（每對 209 token，608×352） | 121,500 token（每對 405 token，864×480） |
| 1 小時 | 122,880 token，每 3.5 秒一對（C） | 122,880 token，每 3.5 秒一對（C；fps 在 C 區間不改變任何東西） |

規則：每兩張畫面（一對）最多 768 token（1080p 與 4K 的 16:9 為 720）。約 2.8 分鐘（2 fps）以內影片隨長度線性增加；再長則總 token 維持在約 131K，畫面變小；超過約 17 分鐘（2 fps，F＝2,048 張）後畫面張數不再增加，時間間隔變粗（`2 × 時長 / F` 秒）。每一對另有 8 到 11 個文字 token。完整的公式與例子見 `docs/configuration.md` 「Long videos」，記憶體與快取見 `docs/tiered-cache.md` 「影片輸入」。

### 建議

- **超過 10 分鐘的影片請先剪成段落**，或調低 `fps`（`{"fps": 0.5}`），否則一支影片就是十萬 token 等級的提示詞：prefill 慢、佔 KV 與主機記憶體。
- **先縮小再送**：模型看到的每張畫面至多約 1152×640，4K 原檔只是白白增加上傳量與解碼時間。用 ffmpeg 轉成 720p、較高壓縮的 H.264（例如 `ffmpeg -i in.mp4 -vf scale=-2:720 -crf 28 out.mp4`）再送。無狀態的聊天 API 每一回合都會重送整段歷史，包括影片本身。
- **同一支影片要位元組相同、`mm_processor_kwargs` 也相同**：影片在提示詞裡的 id 由它的位元組與取樣計畫決定，位元組或 `fps` 一變就是另一支影片，不共用快取。多回合對話把含影片的第一則使用者訊息與其後的歷史**原樣**帶回，影片就留在快取前綴裡，不會再解碼或編碼。
- 同一支影片換一個問題：伺服器會在最後一則使用者訊息裡影片（或圖片）結尾存一個檢查點，第二個問題從那裡續起，影片完全不重算，只算問題文字。做法：**把影片放在問題之前、同一則使用者訊息裡**（如上面的例子）；每次問題之間**影片位元組保持相同**（`mm_processor_kwargs` 也相同）；追問請當成新的回合（帶回上一回合的回覆），或當成新的訊息、重送同一個影片部分。問題在影片之前，或影片不在最後一則使用者訊息裡，就沒有這個檢查點（較早訊息裡的影片只靠上一回合結尾的檢查點）。
- 影片放在使用者訊息裡、**不要放在系統提示詞**（系統提示詞要在所有請求間共用）。
- 超過 `IMAGE_TOKENS`、單一請求放不進 KV 池，會得到 HTTP 400 與說明（要縮短影片，或請管理者調高 `VIDEO_MAX_TOKENS`／`IMAGE_TOKENS`）；KV 池暫時被其他請求佔滿則會排隊等待，不是錯誤。
- 比檢查點預設（768 張、12,288 token）多得多的畫面上，模型的答題品質是模型的事，不是引擎的：請用自己的影片比較。

## 4. 錯誤與重試

| 狀態 | 意義 | 做法 |
| --- | --- | --- |
| 400 | 請求有問題：太長（超過上下文）、影片太大、超過 `IMAGE_TOKENS`、壞的 `mm_processor_kwargs`、非資料／http(s) URL、單一請求放不進 KV 池 | 不要原樣重試；依訊息縮短或改請求 |
| 503 | 引擎沒有載入（已停止，或閒置卸載後正在載入） | 稍後重試；第一個請求會把閒置卸載的引擎載入並等待 |
| 連線中斷（串流途中） | 用戶端斷線會取消該請求 | 重送（前綴多半仍在快取，會很快） |

所有請求都建議開串流（`"stream": true`），長提示詞的 prefill 期間伺服器會送 `: keep-alive` 註解行，避免中間的代理逾時。

## 5. 檢查清單

- [ ] `X-Agent-Id`（或 `user`）固定，子任務放在 `/` 之後。
- [ ] 系統提示詞與工具清單位元組固定，沒有時間戳記與隨機值。
- [ ] 歷史只往後附加；助理的回覆（含 `reasoning_content` 與工具呼叫）原樣帶回。
- [ ] 影片：已縮小、位元組與 `mm_processor_kwargs` 固定、超過 10 分鐘先剪段或降 `fps`。
- [ ] 看 `cached_tokens` 與統計頁的 `Cached (hit)`，確認第二回合起幾乎全命中。
