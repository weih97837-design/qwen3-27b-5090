# Configuration

All settings are environment variables of `frontend/server.py`, with the same names and defaults however it is started
(the exceptions are in the first table below, and `HOST`). The startup log prints the memory plan. Task IDs such as `P7-13` refer to
[decisions.md](decisions.md).

## How settings work

**Keep a setting in one file; try one on the stats page.** A value in the file applies from the next start of the
server. A change on the stats page or through the admin API applies to the running server and is gone when it restarts.

| | Docker (`serve`, `serve.sh`, `serve.cmd`) | Windows package: `start.cmd` | Windows package: tray app |
| --- | --- | --- | --- |
| The file | `.env` next to `compose.yaml` | `settings.ini` next to the program | the same `settings.ini` |
| Changing the file | An editor | An editor | **Settings** in the tray menu ([below](#settings-window)), or an editor |
| Order | The shell's variables, then `.env`, then the default | `settings.ini`, then the environment, then the default | the same as `start.cmd` |
| Address and port | `HOST`, `PORT`: the address published on the host | `HOST`, `PORT` | `HOST`, `PORT` |
| Where the weights are | `./hf` (a symlink moves them; `HF_HOME` is `/hf` in the container) | `HF_HOME`, by default `huggingface\` in the folder | the same as `start.cmd` |
| Getting the weights | Before the start: `scripts/download-weights.sh` | Downloaded at each start if missing | the same as `start.cmd`, with the progress in the tray |
| `ENGINE_AUTOSTART` default | 1: loads at the start | 1: loads at the start | **0: loads at the first request** |
| `ENGINE_IDLE_UNLOAD_S` default | 0: stays loaded | 0: stays loaded | **1800: unloads after 30 minutes** |

What it takes for a setting to change depends on who reads it, and when:

| Takes effect | Settings | Change it for now | Keep it |
| --- | --- | --- | --- |
| At once | `ENGINE_IDLE_UNLOAD_S` | Stats page, `POST /admin/engine/idle` | The file |
| At once | `MIN_SEQ_TPS`, `ADMIT_KV`, `KV_RESERVE_TOKENS`, `DECODE_SHARE` | `POST /admin/concurrency` (`DECODE_SHARE` also in the Stats page's start settings, 「預填時給解碼的時間比例」) | The file |
| At once | `target_seqs` | `POST /admin/concurrency` | Cannot be kept: there is no variable (`MAX_SEQS` is the upper limit that can) |
| When the engine restarts | `WEIGHTS`, `KV_DTYPE`, `SPEC`, `VISION`, `MAX_SEQS`, `MAX_TOKENS`, `VRAM_RESERVE_GIB`, `IMAGE_TOKENS`, `VIDEO_MAX_TOKENS`, `VIDEO_MAX_FRAMES` | Stats page ([Start settings](#start-settings)), `POST /admin/engine` | The file. The page lists the lines to add, for these and for `ENGINE_IDLE_UNLOAD_S` |
| When the server restarts | Everything else, such as `HOST`, `PORT`, `HF_HOME`, `ENGINE_AUTOSTART`, `MIN_KV_TOKENS`, `MAX_TOKENS_FIT`, `STATE_HOST_CACHE`, `PREFILL_CHUNK` | Not possible | The file |

- Restarting the engine reloads the weights, fails the requests in flight and empties the prefix cache. Restarting the
  server does the same if the engine is loaded, and also closes the stats page's connection for a moment.
- `WEIGHTS` with another tokenizer takes a restart of the server ([Other checkpoints](#other-checkpoints)).
- With Docker, restart the server with `docker compose up -d serve` (or `scripts/serve.sh up`): it recreates the
  container when a value changed. `docker compose restart` keeps the old values. With `start.cmd`, close it and run it
  again. The tray app's settings window does what its changes need when you apply them.
- Only variables listed in the `serve` service of `compose.yaml` reach the container. That is all of this page except
  `HOST` and `PORT` (published by compose, above), the Windows package's, and most of
  [Development switches](#development-switches): add those in `compose.override.yaml`.

## Using it from other machines

By default Docker and the Windows package accept connections from this PC only. To use the server from other machines on
your network, set `HOST` to `0.0.0.0`:

| | What to do |
| --- | --- |
| Windows package, tray app | **Settings**, then **Address** `0.0.0.0` and **Apply**, or `HOST=0.0.0.0` in `settings.ini` |
| Windows package, `start.cmd` | `HOST=0.0.0.0` in `settings.ini`, then run it again |
| Docker on native Linux | `HOST=0.0.0.0` in `.env` next to `compose.yaml`, then `docker compose up -d serve`. Nothing else is needed |
| Docker under WSL2 | The same, and also one of the two settings below: WSL2 does not take connections from the network by default |

Use `0.0.0.0` rather than the PC's own address: the tray app and `scripts/serve.sh` check the server on `127.0.0.1`.

**Security:** there is no authentication. Anyone who can reach the port can use the model, and can stop the engine and
change its settings through the admin API and the stats page. It can also make the server fetch image and video URLs
from the server's network (including addresses only that machine can reach). Do this only on a network you trust, and
do not forward the port to the internet.

POST requests that a browser sends from a page of another site (an `Origin` header other than the server's own) get
403, so a web page you visit cannot use the server through your browser, even when it listens only on `127.0.0.1`.
API clients send no `Origin` and are not affected, nor is the built-in UI. Behind a reverse proxy, the proxy must pass
`Host` or set `X-Forwarded-Host`. `TRUSTED_ORIGINS` allows other origins.

**Windows Firewall:** the first start of the Windows package with `0.0.0.0` asks whether to allow Python; allow it for
private networks. If you closed that prompt, add a rule in an administrator PowerShell:

```powershell
New-NetFirewallRule -DisplayName "Qwen3.8-27B" -Direction Inbound -Protocol TCP -LocalPort 46320 -Action Allow
```

**Linux firewalls:** Docker publishes ports with its own iptables rules, ahead of `ufw` and `firewalld`'s usual rules, so
the port opens without a firewall rule and those tools' rules do not close it. To limit who can connect, use Docker's
`DOCKER-USER` chain or a firewall in front of the PC.

**WSL2** ([Microsoft's documentation](https://learn.microsoft.com/en-us/windows/wsl/networking)), either of:

- Mirrored networking (Windows 11 22H2 or later): add `networkingMode=mirrored` under `[wsl2]` in `%USERPROFILE%\.wslconfig`,
  run `wsl --shutdown`, and allow the port in an administrator PowerShell:

  ```powershell
  New-NetFirewallHyperVRule -Name "Qwen3.8-27B" -DisplayName "Qwen3.8-27B" -Direction Inbound -VMCreatorId '{40E0AC32-46A5-438A-A0B2-2B479E8F2E90}' -Protocol TCP -LocalPorts 46320
  ```

- A port proxy, in an administrator PowerShell, plus the Windows Firewall rule above. The WSL address can change when WSL
  restarts; run it again then.

  ```powershell
  netsh interface portproxy add v4tov4 listenport=46320 listenaddress=0.0.0.0 connectport=46320 connectaddress=(wsl hostname -I).Trim()
  ```

**On the other machine,** use the PC's address in place of `127.0.0.1`: the API at `http://<address>:46320/v1` (Anthropic
clients: `http://<address>:46320`) and the stats page at `http://<address>:46320/ui`. With another `PORT`, change the port
everywhere above.

## Serving

| Variable | Default | Meaning |
| --- | --- | --- |
| `HOST` | `frontend/server.py` run directly: 0.0.0.0. Docker and the Windows package: 127.0.0.1 | Address to listen on. `0.0.0.0`: the network ([Using it from other machines](#using-it-from-other-machines)). With Docker, the address published on the host: the server in the container always listens on all its addresses |
| `PORT` | 46320 | Port of the API and the web UI. With Docker, the port published on the host; in the container it stays 46320 |
| `TRUSTED_ORIGINS` | empty | Web origins (`scheme://host[:port]`, comma-separated; `*`: any) whose pages may send POST requests from a browser, besides the server's own. The server sends no CORS headers, so such a page can only send them, not read the answers ([Using it from other machines](#using-it-from-other-machines)) |
| `SERVED_MODEL_NAME` | `qwen3.8-27b` | Name in `/v1/models` (requests may use any model name) |
| `MAX_TOKENS` | 262144 | Context limit of one request. If the KV pool cannot hold it, `MAX_TOKENS_FIT` lowers it; the value in effect is in the startup log (`max_tokens ... lowered to`) and in `/v1/models` (`max_model_len`) |
| `MAX_TOKENS_FIT` | 1 | 1: start with `MAX_TOKENS` lowered to what the KV pool holds (a multiple of 1,024). 0: fail to start instead (the service then restarts every 60 s). The engine library (`q27lib.Engine`) defaults to 0 |
| `MAX_SEQS` | 8 | Requests the engine can run at once (1–8; 8 is the engine's maximum and this fork's default since the pool is uncapped); more wait in a queue. The admission target `target_seqs` starts at `MAX_SEQS`, so up to 8 requests decode together: total throughput rises, each request's own speed drops; set `target_seqs` in /ui to run fewer. How many actually run is decided while running (`ADMIT_KV`, `MIN_SEQ_TPS`, `target_seqs` of the admin API). Each request beyond the first has a 189.6 MiB parking area at the top of the KV pool, lent to the KV cache while unused (P7-13), so `MAX_SEQS=8` costs 7 parking areas (1.3 GiB, but lent to the KV cache while unused) and about 0.25 GiB of fixed per-request buffers; `STATE_HOST_CACHE` (8 GiB of pinned checkpoint slots) is unchanged. An explicit `MAX_SEQS` in the environment, `.env`, `settings.ini` or the tray overrides the default. KV pool with gdnfp4, NVFP4 KV and DFlash2: 620,096 tokens with 4, 605,504 with 8 (DSpine's tables take about 3,600 tokens of it: 601,856 with 8) |
| `ADMIT_KV` | 1 | 1: a waiting request starts only when the KV pool can take it: free pages plus pages held only by prefix-cache checkpoints must cover the remaining prompts of the running requests, `KV_RESERVE_TOKENS` of output per request, and one more parking area. With nothing running a request always starts. 0: start by count only (a request that later runs out of KV is suspended) |
| `KV_RESERVE_TOKENS` | 4096 | Output tokens per request that `ADMIT_KV` keeps room for (or the request's remaining `max_tokens`, if smaller) |
| `MIN_SEQ_TPS` | 0 | Lower bound of tokens per second per request; 0 disables it. Another request does not start if it would push each below this. The prediction uses the measured speed at each number of running requests (decode turns, about the last 10 s); unmeasured counts are scaled from smaller ones by the step time |
| `DECODE_SHARE` | 1.5 | Seconds of decoding given to running requests per second of prefill, so output does not stall during a long prefill. A larger value keeps the decoding requests faster while another request prefills, but a long prompt then takes longer to prefill (about 1 + `DECODE_SHARE` times its prefill time). It was 0.5 before v0.6.2.5: on an RTX 5090 with 3-4 slots at 40-115K context, decoding fell to 2-10 tok/s per slot during a prefill at 0.5 and held 80-100 tok/s at 1.5. Changeable live (`POST /admin/concurrency`, Stats page) |
| `SCHED_QUANTUM_MS` | 50 | Length of a decode turn (with `BATCH_DECODE=0`, the interval between requests' turns; a switch costs about 1 ms) |
| `BATCH_DECODE` | 1 | 1: requests in decode advance together in one batched speculative step (P6-10; about 2.3x aggregate at 4 requests; each request's tokens are the same as alone). 0: requests take turns |
| `PREFILL_CHUNK` | 2048 | Prefill chunk in tokens (8,192 is only 2.5% faster at 32K and needs 4x the activations, P5-4) |
| `KV_DTYPE` | `nvfp4` | `nvfp4`, `fp8` or `bf16`. NVFP4 holds about 1.7x the tokens of FP8 (589,888 against 339,968 at 262K with gdnfp4), prefills as fast or faster (+1 / +6 / +9% at 19K / 73K / 194K, P7-8), and costs a little KL against the BF16 original (0.0470 against 0.0445 with FP8 KV); GSM8K, the Japanese set and needle retrieval are the same. BFCL `parallel` is about 3 points lower than FP8 on short prompts and the same after 32K / 96K of agent context (P4-11, P7-8, P7-9) |
| `WEIGHTS` | `gdnfp4` | `gdnfp4`: GDN projections in NVFP4, attention projections FP8 ([juzbox/Qwen3.8-27B-NVFP4-GDN](https://huggingface.co/juzbox/Qwen3.8-27B-NVFP4-GDN), pinned revision, `scripts/download-weights.sh`; PQ-14). `nvidia`: NVIDIA's checkpoint as is (GDN / attention projections FP8). `quasar`: the QUASAR-QAT checkpoint, every projection NVFP4, weaker at parallel tool calls (`scripts/download-weights.sh --quasar`; its BF16 `lm_head` is replaced by NVIDIA's NVFP4 one through `Q27_LM_HEAD_CKPT`). Or another NVFP4 checkpoint of Qwen3.8-27B on the Hugging Face Hub, e.g. a fine-tune: `ORG/NAME` (its main branch) or `ORG/NAME@REVISION`, downloaded with `scripts/download_weights.py --weights ORG/NAME` (which first checks that the engine takes it) or `scripts/download-weights.sh --repo ORG/NAME` (see [Other checkpoints](#other-checkpoints)). Or a model directory inside the container. From Python: `qwen27b.weights.engine_target(...)`; scripts take `--weights` |
| `SPEC` | `dspine` | `dspine` (speculative decoding with the DSpine draft, `tencent/Qwen3.8-27B-DSpine-b16`: blocks of up to 16, adaptive width), `dflash2` (the DFlash2 draft, blocks of 8) or `none`. Unset with only DFlash2 downloaded (a checkout from before DSpine became the default), the server uses DFlash2 and logs a warning; `scripts/download-weights.sh` gets DSpine |
| `Q27_DRAFT_BLOCK` | `16` | DSpine: rows of a single sequence's draft block (the last token and the drafted ones), 2..16 (DSpine was trained on 16; fewer rows verify faster). Batched steps (several sequences decoding) always draft 8 per sequence |
| `Q27_DRAFT_WINDOW` | `2048` | DSpine: the draft's context window (its attention and context ring), 256..8192. DSpine was trained with 8,192, but at a 32K context that accepted no more tokens than 2,048 and made each step 0.2-0.3 ms longer; its ring takes 160 MiB per sequence at 8,192, 40 MiB at 2,048 (as DFlash2's) |
| `Q27_SPEC_ADAPT` | `1` | Adaptive verify width when the draft's block is longer than 8 (DSpine): a sequence verifies `Q27_SPEC_ADAPT_LOW` (8) rows while the moving average of its tokens per step (`Q27_SPEC_ADAPT_ALPHA`, 0.125) is below `Q27_SPEC_ADAPT_DOWN` (4.0), and the full block again once it reaches `Q27_SPEC_ADAPT_UP` (4.5). Only past steps decide the width, so the output distribution is unchanged. `0`: always the full block |
| `Q27_SPEC_ADAPT_BATCH` | `1` | DSpine, requests that step together: each verifies 4, 8, 12 or 16 rows, chosen from its own tokens per batched step at the rows it ran (a moving average per width; untried widths are extrapolated) and a model of the step's time (the GEMMs step up every 16 rows of the total), within the row budget (8 per request of `MAX_SEQS`). Masked (structured output) requests keep 8. Only past steps decide the rows, so the output distribution is unchanged. Needs `Q27_SPEC_ADAPT`. `0`: 8 rows each |
| `Q27_SPEC_ADAPT_HYST` | `0.01` | `Q27_SPEC_ADAPT_BATCH`: a lane's rows change only when that raises the step's expected tokens per ms by more than this fraction (each new split of the rows re-captures the step's CUDA graph, about 1.2 ms) |
| `Q27_SPEC_ADAPT_PERIOD` | `8` | `Q27_SPEC_ADAPT_BATCH`: batched steps between the decisions (between them the rows stay) |
| `Q27_SPEC_ADAPT_BATCH_ALPHA` | `0.03125` | `Q27_SPEC_ADAPT_BATCH`: rate of the lanes' moving averages. Slower than a single request's (`Q27_SPEC_ADAPT_ALPHA`) because the rule compares lanes: with 1/8, requests of one kind looked different by chance and traded rows at a loss (four sampled reasoning requests 5% slower) |
| `Q27_DRAFT_QUANT` | `fp4` | Draft linear layers: `fp4`, `fp8` or `bf16` (same acceptance, FP4 is fastest) |
| `Q27_DRAFT_VOCAB` | `bench/draft-vocab/vocab-128k.txt` | Candidate vocabulary of the draft (`tools/draft_vocab.py`); `none` for the full vocabulary |
| `STATE_HOST_CACHE` | 8 | GiB of pinned host memory for prefix-cache checkpoints (8 = about 42 checkpoints, twice the 21 of 4; 0 disables the cache; the server sets 8 when unset, the engine library defaults to 0). The slots are not allocated at startup (pinning takes about 0.45 s per GiB on WSL2): one by one while the server is idle, or when a checkpoint needs one |
| `KV_POOL_TOKENS` | 0 | KV pool size in tokens. The default 0 is the maximum: all the VRAM budget leaves (free VRAM minus `VRAM_RESERVE_GIB`, less weights, state and activations), an estimated 600K+ tokens on an RTX 5090 with `MAX_SEQS=8` and DSpine (not measured; the startup log and the stats page show the real size); the parking areas are pages of that pool. The pool therefore depends on the free VRAM at each start: read the real size from the stats page (`pool_pages` of the engine's cache stats), not from the setting, which shows as "auto (max)". With `STATE_HOST_CACHE=0` (no prefix cache) 0 means a pool of `MAX_TOKENS` only. A number (e.g. 524288 = 512K, about 9 GiB of VRAM with NVFP4 KV) fixes it: it is the KV cache with every parking area in use, and the pool is that plus the parking areas (P7-13). An explicit value in the environment, `.env`, `settings.ini` or the tray overrides the default. Checkpoints evicted from the pool go to the spill tiers ([tiered-cache.md](tiered-cache.md)). With MAX_TOKENS_FIT=1 (the default) a pool the VRAM budget cannot hold is lowered to what fits (logged), and MAX_TOKENS with it if needed. |
| `SPILL_RAM_GIB` | 64 | Host RAM for KV checkpoints evicted from the GPU pool, in GiB. 0: off (the evicted checkpoints are dropped). Counts against the WSL2 VM's memory ([tiered-cache.md](tiered-cache.md#wsl2-注意事項)) |
| `SPILL_PINNED_GIB` | 8 | Pinned host memory (GiB) for spill entries, pinned in 1 GiB slabs while the server is idle (`SPILL_PINNED_SLAB_MIB`). Faster copies. An entry is stored in segments (each inside one slab), so entries larger than a slab are pinned too; an entry the pool cannot hold completely uses normal memory. Part of the `SPILL_RAM_GIB` budget: the whole pinned pool counts toward it (reserved host memory, even when unused), and the value is clamped at startup to below it, leaving room for one checkpoint (at least 0.25 GiB). 8 by default since v0.3.2 (verified on an RTX 5090); 0: off. Counts against Windows shared GPU memory ([tiered-cache.md](tiered-cache.md)) |
| `SPILL_ASYNC_IO` | 1 | 1: checkpoint pages are copied on a separate CUDA stream, so a spill or restore no longer stalls the GPU and the engine thread. The CPU is only freed for entries in pinned memory (copies to/from pageable memory block the CPU), so use it together with `SPILL_PINNED_GIB`. On by default since v0.3.2; 0: off |
| `SPILL_ASYNC_PROMOTE` | 1 | 1: a request whose prompt matches a checkpoint spilled to RAM or SSD waits in state `restoring` while the engine loads it in the background (a thread reads the file, the copy runs on the I/O stream), and the requests behind it keep running; with 0 the match is promoted synchronously inside `prefix_begin`, stalling the engine for the whole load. A load starts only when it is clearly faster than computing the prompt again (estimated load time < 0.8 × recompute time) and gives up waiting after 30 s. Needs `SPILL_RAM_GIB` > 0; use with `SPILL_PINNED_GIB` and `SPILL_ASYNC_IO=1` (without async I/O the copy itself is synchronous). On by default since v0.3.2; 0: off |
| `SPILL_FAIR_AGENTS` | 1 | 1 (default): the victim is taken from the agent that currently uses the most bytes of that tier (bytes of a chunk shared by several agents are split between them; no guaranteed minimum share, so an idle agent's entries are the first to go when it is the biggest). 0: when the RAM tier or the SSD tier must free space, the victim is chosen over the whole tier (superseded checkpoints, then never-reused before reused, then the oldest), with no per-agent step. On one user's real workload the RAM reuse rate was about 30% with 0 against about 6% with 1 (the per-agent step evicts the busiest agent's entries, which are the ones most likely to be reused); these are early numbers from a single workload, so try both on yours. Within that agent the order is: superseded checkpoints (a deeper checkpoint of the same conversation exists, and it is not a branch point or a prefix shared by several agents), then entries never reused before reused ones, then the oldest. The VRAM pool stays plain least-recently-used either way. The agent is the `X-Agent-Id` / `user` of the request (see "Agent identity"). Read by the engine process (RAM tier) and by `frontend/tiercache.py` (SSD tier); see `docs/tiered-cache.md` "Retention" |
| `SPILL_COLD_FIRST` | 1 | 1: when a tier must free space, the entries of COLD agents are evicted before anything else (an agent is cold in a tier when none of its entries there has ever been hit and its newest entry there is older than `SPILL_COLD_AFTER_S`; among cold agents the one using the most bytes goes first, then the usual order inside it). 0: no such step. Read by the engine (RAM) and `frontend/tiercache.py` (SSD) |
| `SPILL_COLD_AFTER_S` | 600 | Seconds the newest entry of an agent without hits must be old before the agent counts as cold (a conversation that has just started is not cold). 0: an agent without hits is cold at once |
| `SPILL_UNUSED_TTL_S` | 0 | Seconds; 0 (default): off, a never-hit entry is not dropped for being unused. Above 0: an entry that has had no hit for this many seconds since it was created in RAM / written to the SSD is dropped from RAM and the SSD by the tier policy tick, whatever the free capacity (one-shot agents must not linger). Entries that were hit keep the normal rules; the shared short prefixes (16,384 tokens or less), idle backups and entries being restored are never dropped this way. Counted as `expired` in the tier counters. (With the SSD tier off it applies to RAM only.) Note that a value below `SPILL_RAM_RETENTION_S` (3600) drops a never-hit entry from RAM before it would be written to the SSD. (It was 1800 in v3 until this default changed to 0.) |
| `SPILL_SSD_WARM_ONLY` | 0 | 1: an entry is written from RAM to the SSD only when it was hit or its agent has had a hit in some entry ("warm"). A never-hit entry of an agent without any hit ("unproven") is dropped instead when the agent has no explicit id (`auto:` ids, requests without `X-Agent-Id`), and for an agent with an explicit id it is written only within `SPILL_PROBATION_GIB`. 0 (default): write everything, as before the warm-only rule. Counted as `not_written`. Only matters with the SSD tier on (`SPILL_SSD_GIB` above 0) |
| `CACHE_POLICY` | legacy | The retention policy of the RAM and SSD tiers and of the VRAM checkpoint choice. `legacy` (default): the `SPILL_*` rules above and below, exactly as before. `smart`: a value-based GreedyDual order (prefill seconds a checkpoint saves over its parent, per GiB it costs, times (1 + `CACHE_HIT_WEIGHT` x hits), x `CACHE_JUNCTION_FACTOR` for a branch point / a prefix shared by two agents, x `CACHE_SUPERSEDED_FACTOR` for a checkpoint a deeper one of its chain stands in for, plus the tier's watermark at its last use), leaf-first, recent entries protected; no per-agent or cold-agent step (`SPILL_FAIR_AGENTS`, `SPILL_COLD_FIRST`, `SPILL_SSD_WARM_ONLY` and `SPILL_PROBATION_GIB` are ignored), superseded checkpoints are spilled instead of dropped, and an entry goes to the SSD only when its priority there is above the SSD's watermark. Read by the engine (RAM tier, VRAM checkpoints) and `frontend/tiercache.py` (SSD). Validate it on your own traces with `tools/cache_sim.py` first; see `docs/tiered-cache.md` "CACHE_POLICY=smart" |
| `CACHE_PREFILL_T0_US` | 54 | `smart` only: prefill microseconds per token at depth 0 (the gain model T(a, b) = t0 (b - a) + k (b^2 - a^2) / 2000; default fitted to an RTX 5090 with DFlash2: ~17,000 tok/s at 30K, ~13,600 tok/s at 125K) |
| `CACHE_PREFILL_K_US` | 0.31 | `smart` only: additional prefill microseconds per token per 1,000 tokens of depth (the k of the model above) |
| `CACHE_SUPERSEDED_FACTOR` | 0.25 | `smart` only: the value factor of a checkpoint whose deeper continuation exists (it is kept and spilled with this reduced value, not dropped) |
| `CACHE_JUNCTION_FACTOR` | 2 | `smart` only: the value factor of a branch point or of a prefix shared by entries of two or more agents |
| `CACHE_HIT_WEIGHT` | 1 | `smart` only: value x (1 + this x hits). Hits count the reuses of a conversation's whole lineage (inherited by every deeper checkpoint), so 1 keeps long conversations' tips the longest; on the synthetic workloads of `tools/cache_sim.py` 0 does better (see `docs/tiered-cache.md`) |
| `CACHE_MIN_RESIDENT_S` | 300 | `smart` only: a RAM / SSD entry that came to the tier less than this many seconds ago is not a victim unless every candidate is that young (then the oldest goes). Not used for the VRAM checkpoints |
| `SPILL_PROBATION_GIB` | 4 | The SSD room (logical GiB) all unproven entries of agents with an explicit id may take together; when it is full the oldest unproven entry is dropped first. Entries of warm agents do not count. Only matters with `SPILL_SSD_WARM_ONLY=1` (it is 0 by default, so this quota is unused by default), and unused with the SSD tier off |
| `SPILL_DIR` | `/cache/kv` | Directory in the container where spilled checkpoints are written (only with `SPILL_SSD_GIB` above 0: with `SPILL_SSD_GIB=0` the server starts the engine with an empty `SPILL_DIR`, and a folder an earlier run left there is not touched). `compose.yaml` mounts the named volume `kvcache` there |
| `SPILL_RAM_RETENTION_S` | 3600 | Seconds a checkpoint stays in RAM before `frontend/tiercache.py` writes it to SSD (only when idle). Only matters with the SSD tier on |
| `SPILL_SSD_GIB` | 300 | Cap of the SSD tier in GiB; the oldest files are deleted first, and `SPILL_SSD_MIN_FREE_GIB` (10) keeps that much of the disk free whatever the cap, so 300 is a ceiling, not a reservation. **0: the SSD tier is off.** Nothing is written to, read from or deleted in `SPILL_DIR` (the server starts the engine with an empty `SPILL_DIR`, which makes it drop the victims of a full RAM tier instead of writing them; no write at a stop, no import at a start). Nothing survives an engine restart then; see [The SSD tier: on by default, how to turn it off](#the-ssd-tier-on-by-default-how-to-turn-it-off) |
| `SPILL_SSD_MIN_FREE_GIB` | 10 | The SSD tier never fills the disk below this much free space. Only matters with the SSD tier on |
| `SPILL_SSD_MIN_TOKENS` | 32768 | Spilled checkpoints shorter than this many tokens are never written to the SSD: past `SPILL_RAM_RETENTION_S` they are dropped from RAM instead. Every entry carries a ~190 MiB recurrent state whatever its length, and a short prefix recomputes in about 0.2-0.4 s, about the time a read takes. 0: no minimum (every entry is written). Entries already on disk are unaffected. Only matters with the SSD tier on (`SPILL_SSD_GIB` above 0) |
| `SPILL_TTL_DAYS` | 7 | Files not reused for this many days are deleted. Only matters with the SSD tier on |
| `SPILL_TICK_S` | 30 | Seconds between the tier maintenance passes |
| `VRAM_BACKUP_VICTIMS` | 8 | The idle backup copies only one of the next N checkpoints the pool would evict (0: any); the copies also stay within half of `SPILL_RAM_GIB`. Environment only |
| `VRAM_BACKUP_IDLE_S` | 60 | Seconds the engine must be idle before it copies VRAM checkpoints to RAM while keeping them in VRAM (idle backup, [tiered-cache.md](tiered-cache.md#閒置時預先備份vram-保留)). Needs `SPILL_RAM_GIB` above 0. 0: off |
| `SPILL_FLUSH_ON_STOP_S` | 8 | Seconds a stop (idle unload, restart, shutdown) may spend backing up VRAM checkpoints (half of it) and writing RAM entries to `SPILL_DIR` (with the SSD tier off nothing is done at a stop: there is nothing to write to, and the cache is cold at the next start). Cut to 8: `server.py` gives the engine 15 s to exit on an unload or restart and 10 s on shutdown, then terminates it. 0: off |
| `CHECKPOINT_INTERVAL` | 4096 | Distance between checkpoints inside a long prompt (a multiple of 64) |
| `VISION` | `host` | Image input. `host`: the vision tower's BF16 weights (0.86 GiB) stay in pinned host memory and are streamed to the GPU one layer ahead of the computation; the KV pool loses 192 tokens, small images take +13 ms (the copy is about 20 ms), 1024x1024 and larger +1–4 ms; output identical to `bf16`. `fp8`: E4M3 weights with per-row scales on the GPU, widened per layer (0.43 GiB; embedding error about 3x that of `bf16`, answers can change). `bf16`: BF16 weights on the GPU (fastest, 0.86 GiB). `off`: requests with images get HTTP 400 (the engine library's default). Activations and the weight staging (90 MiB) share the prefill's buffers |
| `IMAGE_MAX_PIXELS` | 4194304 | Images are resized to at most this many pixels (2048x2048 = 4,096 tokens, 370 ms in the vision tower) |
| `IMAGE_TOKENS` | 131,072 | Image and video tokens per request (embeddings in pinned host memory, 10 KiB per token: 1.25 GiB at 131,072; 160 MiB at 16,384). More: HTTP 400. Unset or empty: 16,384, or `VIDEO_MAX_TOKENS` when that is more, so 131,072 by default. The engine library's own default (`q27lib.Engine`) is 16,384. See [Host memory](#host-memory-of-images-and-videos) |
| `VIDEO_MAX_TOKENS` | 131,072 (the checkpoint's own: 12,288) | Tokens of a video at most (the pixels of all its sampled frames together, 2,048 per token; the checkpoint's `size.longest_edge`), besides the time stamps of its pairs. A pair of frames takes at most 768 tokens all the same, so a video of up to 170 s at 2 frames per second (341 frames) keeps its size and longer ones get smaller frames; with the checkpoint's 12,288 that is 32 s. A request's `mm_processor_kwargs.size.longest_edge` replaces it for that request ([Long videos](#long-videos)) |
| `VIDEO_MAX_FRAMES` | 2,048 (the checkpoint's own: 768) | Frames sampled from a video at most (2 per second until then), and the most a request's `num_frames` / `min_frames` / `max_frames` may ask for |
| `VRAM_RESERVE_GIB` | 1.0 | The minimum VRAM to keep unused, in GiB: the budget of the engine is the free VRAM minus this. It is for the desktop and other programs, and to stay below the point where Windows (WDDM) starts to page GPU memory to system RAM without an error, which slows inference about 7 times. The default is 1.0 GiB because the display is expected on the iGPU, so the dGPU needs little for the desktop (it was 1.5 GiB, the old automatic budget on an RTX 5090; each 0.5 GiB less is about 29K more tokens of KV pool in auto mode). If the dGPU also drives a display or other GPU programs run on it, raise it to 1.5–2 to stay clear of WDDM paging. Lowering it gives the KV pool more room (each 1 GiB is about 58K tokens of NVFP4 KV at 18 KiB per token) at the risk of that paging; a negative or unreadable value counts as 0. The log shows `reserve R GiB, budget B GiB`. The old `VRAM_BUDGET_GIB` (legacy, prefer `VRAM_RESERVE_GIB`) still works as an upper bound that can only lower the budget |
| `Q27_DEVICE_FREE_BYTES` | set by the server | Free VRAM with other programs counted (NVML), which the engine's budget uses if it is below what `cudaMemGetInfo` says |
| `MIN_KV_TOKENS` | 0 | Do not start the engine if the KV pool would hold fewer tokens (other programs are using the GPU, or a low budget). 0: start with the KV pool that fits ([Too little VRAM](#too-little-vram)) |
| `Q27_LOAD_THREADS` | 8 | Threads that read the weight files ahead of the uploads (the bytes loaded are the same with any number). On WSL2 one read stream from a Windows drive (`/mnt/d`, virtiofs) gets 1.2–1.5 GB/s; 4 to 16 threads reach the filesystem's 2.6–2.8 GB/s, and the engine starts in about 10 s instead of 18–21 s. A copy of the weights on the Linux file system, `cache/weights/<org>--<name>/<revision>/` (`cp -L` of the HF snapshot; used before the HF cache when present; the serve container sees it when `compose.override.yaml` mounts `./cache/weights:/work/cache/weights:ro`), starts in about 3.3 s while it is in the page cache |
| `SERVE_TAG` | — | Not a setting on this branch. The serve image is the local `local/qwen3-27b-5090-serve:tiered-kv-cache` (compose: `pull_policy: build`), built from this checkout by `scripts/serve.sh up --build`. Do not `docker compose pull`: the upstream image `ghcr.io/mkyt/qwen3-27b-5090` has no tiered cache |
| `ENGINE_AUTOSTART` | 1 | 0: the server starts with the engine stopped (no VRAM in use). Requests get 503 until it is started at `/ui` or with `POST /admin/engine`; with `ENGINE_IDLE_UNLOAD_S` above 0, the first generation request loads it instead |
| `ENGINE_IDLE_UNLOAD_S` | 0 | Seconds without a request after which the engine stops and frees its VRAM; the next generation request loads it again and waits for it ([Unloading when idle](#unloading-when-idle)). 0: off, the engine stays ready |
| `STATS_SAMPLE_S` | 1 | Seconds between the GPU samples (NVML: utilization, power, clocks, temperature, memory) of the stats page `/ui`; the last 10 minutes are kept. 0: no sampling (the page still shows requests, VRAM and the log) |
| `API_DUMP_DIR` | unset | Debugging: write the body of every POST to this directory, created if missing (to see what a client sends; may contain personal data). In the Windows package, a full path in `settings.ini` |

### The SSD tier: on by default, how to turn it off

`SPILL_SSD_GIB` defaults to **300**: the SSD tier is on, with a 300 GiB cap (`SPILL_SSD_MIN_FREE_GIB`, 10, keeps that much of the disk free whatever the cap). The other tiered-cache defaults: `SPILL_FAIR_AGENTS=1` (the per-agent step is on), `SPILL_SSD_WARM_ONLY=0` (every entry may be written to the SSD) and `SPILL_UNUSED_TTL_S=0` (a never-hit entry is not dropped for being unused). `SPILL_PROBATION_GIB` only matters with `SPILL_SSD_WARM_ONLY=1`.

- **Values you set yourself still win.** An explicit `SPILL_SSD_GIB` in `settings.ini`, `.env`, the environment, the tray's Settings window or the stats page is used as before: someone who set `SPILL_SSD_GIB=200` keeps a 200 GiB cap, and someone who set `SPILL_SSD_GIB=0` keeps the tier off. Only an unset (or empty) variable takes the default.
- **To turn the SSD tier off**, set `SPILL_SSD_GIB=0`. The server then starts the engine with an empty `SPILL_DIR`, so the engine drops the victims of a full RAM tier instead of writing files, and the policy has no directory: no write, no read, no directory scan, no write at a stop (`SPILL_FLUSH_ON_STOP_S`), no import at a start. It takes effect when the engine restarts. The stats page then shows "SSD 層：已關閉（SPILL_SSD_GIB=0）".
- **The cost of turning it off:** nothing survives on disk, so after an engine restart (a stop, a settings change, an idle unload, an update) the RAM tier and the VRAM prefix cache are cold, and the first request of each conversation prefills its whole prompt again. Entries pushed out of a full RAM tier are lost instead of waiting on the SSD.
- **An existing SSD folder is left untouched while the tier is off** (the `kvcache` volume, or `%LOCALAPPDATA%\qwen3-27b-5090\kvcache` on Windows): its files are not read and not deleted, and keep taking their disk space. Delete them yourself (`docker volume rm`, or delete the folder) to get the space back; if you turn the tier on again later with the same folder, the files of the same model are imported again.
- `SPILL_SSD_MIN_TOKENS`, `SPILL_SSD_MIN_FREE_GIB`, `SPILL_SSD_WARM_ONLY`, `SPILL_PROBATION_GIB`, `SPILL_TTL_DAYS`, `SPILL_RAM_RETENTION_S` and `SPILL_FLUSH_ON_STOP_S` only matter with the SSD tier on (`SPILL_PROBATION_GIB` also needs `SPILL_SSD_WARM_ONLY=1`).

### Attention numerics

| Variable | Default | Meaning |
| --- | --- | --- |
| `Q27_ATTN_FP8` | empty | Prefill attention MMAs, read per call. Empty: `2` with FP8 KV, `3` with NVFP4 KV (`2` without the staging buffer). FP8 KV: `2` QK^T and P·V in FP8, `1` P·V only, `0` BF16. NVFP4 KV: `3` QK^T on native FP4 MMAs (Q quantized to NVFP4, K as cached, V from the E4M3 staging buffer; P7-8), `2` K and V rounded to E4M3 for FP8 MMAs, otherwise exact in f16 |
| `Q27_ATTN_FP4_STAGE` | 1 | NVFP4 KV prefill: `1` converts the context to an E4M3 staging buffer once per layer and chunk (mode 3: V only, 1 KiB per token of `max_tokens`, 256 MiB at 262K; with `Q27_ATTN_FP8=2` at startup K and V, 2 KiB per token, bit-identical to per-CTA conversion). `0`: convert in every CTA (mode 2). Sized at startup, read per call |
| `Q27_ATTN_TC_F8` | empty | Decode / verify attention MMAs, read per call. Empty or `1`: FP8 with FP8 and NVFP4 KV (P7-7; FP8 KV since P3-36), `0`: BF16 / f16 (a 16-row verify then takes two launches with FP8 KV). FP8 means Q in E4M3 with per-row scales, NVFP4 KV re-rounded to E4M3, P in E4M3: 8% faster speculative steps at 131K and at 8 x 32K, no change on golden or needle retrieval |

## Long videos

A video is sampled at 2 frames per second (`fps`, a request's `mm_processor_kwargs` can change it), as the checkpoint's
processor does, but with a larger budget than the checkpoint's own (12,288 tokens, 768 frames, where a 10-minute 1080p
video gets 224 x 128 pixels a frame): `VIDEO_MAX_TOKENS` (T below) is 131,072 and `VIDEO_MAX_FRAMES` (F) 2,048 here.
Videos up to about 2.8 minutes (341 frames at 2 fps) are the same with either: each pair of frames takes at most 768
tokens. Set the checkpoint's numbers to get its behaviour back:

```bash
VIDEO_MAX_TOKENS=12288 VIDEO_MAX_FRAMES=768 docker compose up -d serve
```

The defaults are in one place, `MEDIA_DEFAULTS` in `frontend/launch.py` (`compose.yaml`, `settings.example.ini` and
these tables say the same; `frontend/tests/test_launch.py` compares them). The same three are start settings of the
stats page (`image_tokens`, `video_max_tokens`, `video_max_frames`, under "影片與圖片") and rows of the tray's
Settings window; empty means the defaults. The server's own processor reads `VIDEO_MAX_*` from the settings of the
engine that is running, so a restart with changed values applies them.

Or per request, within `IMAGE_TOKENS` and `VIDEO_MAX_FRAMES`:
`"mm_processor_kwargs": {"fps": 2, "max_frames": 2048, "size": {"longest_edge": 268435456}}`
(`size.longest_edge` is T in pixels: 2,048 per token; `num_frames` takes a frame count instead of `fps`).

### How a video becomes tokens

| Step | Rule | Depends on |
| --- | --- | --- |
| 1. Frames n | duration x fps, within 4..F (and the video's own frame count), rounded up to an even count | duration, fps, F |
| 2. Pairs | n / 2. The language model sees each pair of frames as one image, after its time as text | n |
| 3. Pixels per frame | min(768 tokens' worth, T x 2,048 / n) | T, n |
| 4. Frame size | the aspect ratio kept, both sides multiples of 32, within step 3; a smaller video is not enlarged | resolution, aspect ratio |
| 5. Tokens | (height / 32) x (width / 32) per pair, times the pairs | steps 1-4 |

Each pair also takes 8 to 11 tokens of text (its time, `<12.5 seconds>`, and the vision start / end). The model
sees time in steps of a pair: 2 / fps seconds, or 2 x duration / F once the frames reach F.

### Three regimes

| Regime | When | Tokens per pair | Tokens of the video | Time step of the pairs |
| --- | --- | --- | --- | --- |
| A: the per-pair cap | n <= 2T / 768 (341 frames by default, 32 with the checkpoint's T) | 768 (1152 x 640 for 16:9), or the video's own size | pairs x 768: grows with duration x fps, below T | 2 / fps s |
| B: the token bound | 2T / 768 < n < F | about 2T / n: smaller for longer videos and higher fps | about T | 2 / fps s |
| C: the frame bound | duration x fps >= F | about 2T / F | about T | 2 x duration / F s: coarser for longer videos |

- **Duration**: in A the tokens grow with it; in B the frames get smaller; in C the frames get farther apart.
- **fps**: sets the number of frames only. In B and C the total stays about T, so a higher fps trades frame size
  for finer time; in C it changes nothing.
- **T**: sets where A ends (T / 384 frames) and the frame size in B and C; videos in A do not change.
- **F**: sets where C starts (F / fps seconds) and its time step. Raising F without T gives more, smaller frames.
- **Resolution**: matters only when the video is smaller than step 3 allows (640 x 360 takes at most 220 tokens a
  pair); larger videos, 1080p or 4K, end up the same size, only the aspect ratio counts.
- The totals fall a little short of T because both sides are rounded down to multiples of 32 (10,752 of 12,288).

### Examples (1080p unless noted; frame size in parentheses)

| Video | Default (T 131,072, F 2,048) | The checkpoint's (T 12,288, F 768) |
| --- | --- | --- |
| 10 s, 2 fps | 10 pairs x 720 (1152 x 640) = 7,200 (A) | the same, 7,200 (A) |
| 60 s, 1 fps | 30 x 720 = 21,600 (A) | 30 x 390 (832 x 480) = 11,700 (B) |
| 60 s, 2 fps | 60 x 720 = 43,200 (A) | 60 x 190 (608 x 320) = 11,400 (B) |
| 60 s, 4 fps | 120 x 720 = 86,400 (A) | 120 x 91 (416 x 224) = 10,920 (B) |
| 10 min, 1 fps | 300 x 405 (864 x 480) = 121,500 (B) | 300 x 32 (256 x 128) = 9,600 (B) |
| 10 min, 2 fps | 600 x 209 (608 x 352) = 125,400; every 1.0 s (B) | 384 x 28 (224 x 128) = 10,752; a pair every 1.6 s (C) |
| 1 hour, 2 fps | 1,024 x 120 (480 x 256) = 122,880; every 3.5 s (C) | 384 x 28 = 10,752; a pair every 9.4 s (C) |
| 640 x 360, 60 s, 2 fps | 60 x 220 (640 x 352, its own size) = 13,200 (A) | 60 x 190 = 11,400 (B) |

For a long video, put it in B or C, where the total reaches T: raise T for detail in each frame, F (and fps) for
finer time. Lower T and F (or the request's `fps`) for a machine with little memory or to keep the prefill short.

### Host memory of images and videos

The engine's pinned host memory is the sum of: the media embeddings (`IMAGE_TOKENS` x 10 KiB: 1.25 GiB at 131,072),
the prefix cache's state checkpoints (`STATE_HOST_CACHE`, 8 GiB), the tiered cache's pinned slabs (`SPILL_PINNED_GIB`,
8 GiB) and the vision tower's weights (`VISION=host`, 0.86 GiB): about 18 GiB with the defaults, of which the media
table is 1.25 GiB. Resized frames (6 KiB per token, 0.75 GiB at 131,072) are ordinary host memory, held only while a
video is encoded. On native Windows pinned memory counts against the GPU's shared memory (about half of the RAM), and
Windows may page it out under pressure, which slows the engine: on a machine with 32 GiB of RAM or less, lower
`VIDEO_MAX_TOKENS` (32,768 takes 0.31 GiB of table), `SPILL_PINNED_GIB` or `STATE_HOST_CACHE` rather than leaving all
three at their defaults. `tools/recommend_config.py` does the first for machines with less than 32 GiB of RAM.

### Limits and costs

- `IMAGE_TOKENS` follows `VIDEO_MAX_TOKENS` unless it is set: the embeddings take 10 KiB of pinned host memory per
  token (1.25 GiB for 131,072), allocated when the engine starts. A request with more tokens of images and videos
  gets HTTP 400. A video of 131,072 tokens plus its text also needs a context (`MAX_TOKENS`, 262,144 by default) and
  KV pool room: a request that cannot fit the pool at all gets HTTP 400 at once instead of waiting.
- The tokens of the videos, their pairs' text included, must fit the context: 1,024 pairs add about 10K tokens of
  text.
- A frame takes at least about 4,300 pixels (about 4 tokens), so with a very high F and a low T (more than about
  0.48 T frames) the total goes above T. The defaults never get there.
- The resized frames are held in host memory while the video is encoded: 3 bytes per pixel, 6 KiB per token (about
  0.8 GiB for 131,072 tokens).
- The vision tower's time grows with the pairs and the prefill's with the tokens, as for a long text prompt; the
  pairs already in the prefix cache are not decoded or encoded again on the next turn.
- How well the model answers on many more frames or tokens than the checkpoint's defaults is the model's, not the
  engine's: compare on your own videos.

## Concurrency

- Up to `MAX_SEQS` requests run at once. Prefill goes to the request with the least prompt left; requests in decode
  advance together in batched steps (with `BATCH_DECODE=0` they take turns every `SCHED_QUANTUM_MS`).
- New requests wait until the KV pool can take them (`ADMIT_KV`). If the KV cache still runs out, the latest request
  (including the one that ran out) is suspended to a checkpoint and goes back to the head of the queue. When requests
  that arrive together share a prefix, the later one waits until the earlier one has passed it.
- Admin API (no authentication; Docker and the Windows package listen on 127.0.0.1 unless `HOST` says otherwise): `GET /admin/concurrency` returns the settings
  and the state (running, waiting, why requests wait, measured tok/s per number of running requests);
  `POST /admin/concurrency` changes `target_seqs` (1 to `MAX_SEQS`; lowering it does not stop running requests),
  `min_seq_tps`, `admit_kv`, `kv_reserve` and `decode_share` (0 to 10: seconds of decoding given to running requests per
  second of prefill; higher keeps decoding from slowing during a prefill but makes long prompts prefill slower; NaN,
  infinity, negative or above 10 are refused with 400). Changes survive an engine restart, not a service restart.
  Example: `curl -s -X POST localhost:46320/admin/concurrency -d '{"target_seqs": 4}'`.
  The answer (and `GET`) has the keys `target_seqs`, `engine_seqs`, `admit_kv`, `kv_reserve`, `min_seq_tps`, `decode_share`,
  `rates`, `running`, `waiting` and `blocked`, e.g. `{"target_seqs": 4, "engine_seqs": 8, "admit_kv": true,
  "kv_reserve": 4096, "min_seq_tps": 0.0, "decode_share": 1.5, "rates": {"1": 96.0}, "running": 1, "waiting": 0,
  "blocked": null}`.
- `/metrics` (Prometheus): among others `q27:target_seqs`, `q27:decode_share`, `q27:admission_blocked{reason=...}`,
  `q27:admission_waits_total{reason=...}`, `q27:cached_tokens_total`, `q27:reusable_tokens_total`.

## Stats page

`http://localhost:46320/ui` shows the server live in the browser (no authentication, like the admin API):

- GPU utilization, power against the limit, clocks and temperature (NVML), and the aggregate decode and prefill
  tokens per second, over the last 1 to 10 minutes.
- Device memory by owner: weights (by part of the model), the KV pool (pages of running requests, pages only
  prefix-cache checkpoints hold, parked sequence state, free), sequence state, activations, the CUDA context and
  runtime (CUDA graphs, libraries), other programs and the driver, and free memory, plus the pinned host pools. The
  pool sizes come from the engine's memory plan. The bar always shows the whole device from NVML, updated every
  second: without an engine, other programs and free memory; with one, what other programs take or release after it
  started is the change of the memory in use beyond the engine's own allocations (`cudaMemGetInfo`).
- Each running or waiting request: its context (tokens restored from the prefix cache, prefilled, still to run,
  generated, with the message boundaries), current and average prefill and decode tokens per second, tokens per
  speculative step, TTFT, time queued.
- A log of the finished requests (prompt, cached tokens and hit rate, output, TTFT, prefill / decode tok/s, finish
  reason, the start of the last user message) and of engine events (admission waits, preemptions, errors,
  restarts), the last 500 entries.

The page also starts, stops and restarts the engine (a start takes about 3.3 s with the weights in the page cache, about 10 s
reading them from a Windows drive under WSL2: see `Q27_LOAD_THREADS`). The engine process holds all the VRAM; the HTTP server
holds none, so stopping the engine frees the GPU for other work while the service and the page stay up. Requests in
flight fail with 503, and so do new ones until the engine is started again. The prefix cache is lost. The same from
a script: `curl -s -X POST localhost:46320/admin/engine -H 'Content-Type: application/json' -d '{"action": "stop"}'`
(`start`, `stop` or `restart`; `GET /admin/engine` returns the state: `stopped`, `loading`, `ready`, `restarting`
after a crash, `stopping`). The JSON content type is required, so that another web page cannot send the action as a
form.

### Unloading when idle

With `ENGINE_IDLE_UNLOAD_S` above 0 the engine stops after that many seconds without a generation request, so that the
GPU is free for other work, and the next generation request starts it and waits for it instead of getting 503. It is off by
default: the first request after an unload takes the load time longer (see above), and the prefix cache is lost, so
that request and the next ones with its context prefill from the start. Polling `/health`, `/metrics`, `/v1/models` or the stats page does not count as
a request and does not keep the engine loaded.

- `/health` answers 200 with `{"status": "unloaded"}` while unloaded (with `refused`, the reason, after a load that found
  too little VRAM) and `{"status": "loading"}` during such a load: the service can take requests.
  `GET /admin/engine` has `state: "stopped"`, `unloaded: true` and `idle_unload_s`.
- The memory plan is made when the engine loads, from the VRAM free then ([Too little VRAM](#too-little-vram)).
- A request waits at most 5 minutes for the load, then gets 503. Clients with a shorter timeout see their own timeout.
- `ENGINE_AUTOSTART=0` starts the server in this unloaded state: no VRAM in use until the first request.
- A stop from the stats page or `POST /admin/engine` stays until a start from there: requests get 503, as without this
  setting.
- The stats page tells these states apart ("引擎已卸載": the first request loads it; "引擎已停止": requests get
  503 until a start), shows the time left before an unload and why a load was refused, and changes the time under
  **閒置時卸載** in the start settings. From a script:
  `curl -s -X POST localhost:46320/admin/engine/idle -H 'Content-Type: application/json' -d '{"idle_unload_s": 600}'`
  (0: off). It applies at once, the time running from then, and lasts until the service restarts.

### Too little VRAM

The engine plans its memory once, when it loads, from the VRAM free at that moment, with other programs counted (NVML; CUDA
alone does not see them, on WSL2 or on Windows), and allocates nothing beyond that budget. NVML catches up with a program's
allocation within about 2 s, so a load in that moment can count too little. If other programs are using the GPU then, what
happens depends on the settings:

| Setting | Effect |
| --- | --- |
| Default | The KV pool takes what is left. If it cannot hold `MAX_TOKENS`, `MAX_TOKENS` is lowered as well (`MAX_TOKENS_FIT=1`) |
| `MIN_KV_TOKENS=N` | The engine does not start if the KV pool would hold fewer than N tokens (for scale: about 600K with nothing else on the GPU, nvfp4, the default `KV_POOL_TOKENS=0`) |
| `MAX_TOKENS_FIT=0` | The engine does not start if the KV pool cannot hold `MAX_TOKENS` |

For scale, with the defaults (`KV_POOL_TOKENS=0`, gdnfp4, nvfp4, `MAX_SEQS=8`), by the memory other programs use: nothing, a KV pool of about 600K
tokens; 6 GiB, about 230K and `MAX_TOKENS` lowered to about 230K; 8 GiB, 120K; from about 9 GiB the engine does not start, unless `MAX_SEQS` is lower. These are this fork's defaults (`KV_POOL_TOKENS=0`, `MAX_SEQS=8`); a fixed `KV_POOL_TOKENS` is lowered to what fits with `MAX_TOKENS_FIT=1`, while 0 always takes what is left.

An engine that does not start allocates no pools. With `ENGINE_IDLE_UNLOAD_S` set, the requests waiting for it get 503 with the
reason (also `refused` in `GET /admin/engine`), and the next request tries again; there are no retries in between. Without it,
the server retries as after a crash, up to every 60 s.

**This covers the load only, by design.** The engine does not watch the GPU afterwards. A program that starts allocating VRAM
while the engine is loaded can push memory to system RAM, its own or the engine's, without an error. Before starting other GPU
work, unload the engine (the stats page, `POST /admin/engine`, or `ENGINE_IDLE_UNLOAD_S`).

`GET /ui/stream` is the page's event stream (server-sent events) and `GET /ui/state` returns one snapshot as JSON.

### Start settings

The page's header shows the running engine's start settings (with the engine stopped, the next start's); clicking
them opens "啟動設定" (Start settings), which choose what the next start or restart of the engine uses, in place of these
variables:

| Setting | Variable | Values |
| --- | --- | --- |
| `weights` | `WEIGHTS` | `gdnfp4`, `nvidia`, `quasar` (those downloaded), or the model directory the service started with |
| `kv_dtype` | `KV_DTYPE` | `nvfp4`, `fp8`, `bf16` |
| `spec` | `SPEC` | `dspine`, `dflash2`, `none` |
| `vision` | `VISION` | `host`, `fp8`, `bf16`, `off` |
| `max_seqs` | `MAX_SEQS` | 1 to 8 |
| `max_tokens` | `MAX_TOKENS` | 2,048 to 262,144 (lowered to what the budget holds, `MAX_TOKENS_FIT`) |
| `vram_reserve_gib` | `VRAM_RESERVE_GIB` | `null` (the default, 1.0 GiB) or a number of GiB from 0 up to the device's memory |
| `image_tokens` | `IMAGE_TOKENS` | `null` (16,384, or `video_max_tokens` when that is more: 131,072 by default) or 1 to 262,144: image and video tokens of one request. Pinned host memory, 10 KiB per token, allocated when the engine starts (it changes the plan's host memory) |
| `video_max_tokens` | `VIDEO_MAX_TOKENS` | `null` (131,072; the checkpoint's own: 12,288) or 1 to 262,144: tokens of one video. The server's own bound: no pool changes. While a video is encoded its resized frames take 6 KiB of host memory per token |
| `video_max_frames` | `VIDEO_MAX_FRAMES` | `null` (2,048; the checkpoint's own: 768) or 2 to 16,384: frames sampled from a video, and the most a request may ask for |
| `spill_ram_gib` | `SPILL_RAM_GIB` | A number of GiB, 0 or more (64 by default; 0: no tiered cache) |
| `spill_ram_retention_s` | `SPILL_RAM_RETENTION_S` | Seconds, 0 or more (3600) |
| `spill_ssd_gib` | `SPILL_SSD_GIB` | GiB, 0 or more (300; 0: the SSD tier is off) |
| `spill_ssd_min_free_gib` | `SPILL_SSD_MIN_FREE_GIB` | GiB, 0 or more (10) |
| `spill_ssd_min_tokens` | `SPILL_SSD_MIN_TOKENS` | Tokens, 0 or more (32768; 0: no minimum; a fraction is rounded down) |
| `spill_ttl_days` | `SPILL_TTL_DAYS` | Days, 0 or more (7) |
| `vram_backup_idle_s` | `VRAM_BACKUP_IDLE_S` | Seconds, 0 or more (60; 0: off) |
| `spill_pinned_gib` | `SPILL_PINNED_GIB` | GiB, 0 or more (8; 0: off; the engine clamps it below `SPILL_RAM_GIB`) |
| `spill_async_io` | `SPILL_ASYNC_IO` | `true` or `false` (`true`) |
| `spill_async_promote` | `SPILL_ASYNC_PROMOTE` | `true` or `false` (`true`) |
| `spill_fair_agents` | `SPILL_FAIR_AGENTS` | `true` or `false` (`true`) |
| `spill_cold_first` | `SPILL_COLD_FIRST` | `true` or `false` (`true`) |
| `spill_cold_after_s` | `SPILL_COLD_AFTER_S` | Seconds, 0 or more (600) |
| `spill_unused_ttl_s` | `SPILL_UNUSED_TTL_S` | Seconds, 0 or more (0: off) |
| `spill_ssd_warm_only` | `SPILL_SSD_WARM_ONLY` | `true` or `false` (`false`) |
| `spill_probation_gib` | `SPILL_PROBATION_GIB` | GiB, 0 or more (4) |
| `cache_policy` | `CACHE_POLICY` | `"legacy"` or `"smart"` (`"legacy"`; its constants `CACHE_*` are plain variables) |
| `spill_dir` | `SPILL_DIR` | A folder for the spilled checkpoints; empty: no SSD tier |

The tiered cache settings ("分層快取" in the page, and a group of the tray app's settings window) have no upper limit, since
the machine decides what fits: the page only warns when the RAM tier is more than the machine's RAM. The engine process
and `TierPolicy` read them when the engine loads, so they apply with a restart of the engine (the page's **重新啟動**,
the tray's **Apply**); the other tier variables (`SPILL_TICK_S`, `SPILL_FLUSH_ON_STOP_S`, `SPILL_PINNED_SLAB_MIB`) stay
variables. The environment (or `settings.ini` `[env]`) still gives the values the server starts with.

Under every choice the page shows the longest context of one request and the KV pool all requests share if only that
setting changed, and for the settings as chosen the device pools against the budget. These come from the engine's own
memory planner (`q27_plan`: the pools of an engine start, nothing allocated or loaded, about 10 to 80 ms each), so
they are the numbers the startup log then prints. The free device memory they assume is the running engine's (before
its pools; a restart sees the same), or with the engine stopped the free memory NVML reports now less an engine's
CUDA context (about 1.2 GiB on WSL2, measured at each start).

The settings stay for later starts and restarts (also after a crash) until the service restarts, which starts from
the variables again: to keep them, set the variables, which `GET /admin/engine` lists in `keep_env` and the page shows
(in `settings.ini` for the Windows package). Changes of `/admin/concurrency` stay too (`target_seqs` only while `max_seqs` is the same).
The same from a script:

- `GET /admin/engine`: the state, `running` (the running engine's settings), `settings` (the next start's),
  `at_start` (those the service started with), `keep_env`,
  `weights` (null for each profile that can be loaded, else why not) and `device` (the free and total memory the
  plans assume).
- `POST /admin/engine/plan` with `{"settings": {...}}` (changes to the next start's settings): `plan` (`max_tokens`,
  `kv_tokens`, `budget`, `device` and `pools` in bytes, `fits`, or `error`), `variants` (the plans with each
  setting's other values), `weights` and `at_start`. `frontend/plan.py` answers the same without a server.
- `POST /admin/engine` with `{"action": "start" | "restart", "settings": {...}}`: starts with those changes, if they
  fit (HTTP 400 otherwise; 409 for `start` while the engine runs). With `"action": "save"` (the page's **Save for the
  next load**) they only stay for the next load, while the engine is stopped or unloaded (409 otherwise): it does not
  load, and whether they fit is decided when it does. Example:
  `curl -s -X POST localhost:46320/admin/engine -H 'Content-Type: application/json' -d '{"action": "restart", "settings": {"kv_dtype": "fp8", "max_seqs": 4}}'`.

## Agent identity

The server tags each request with the AI agent it belongs to, so that the stats page can show the cache per agent and a
later change can evict the tiered cache fairly between agents. Nothing changes for a client that sends nothing.

- Send `X-Agent-Id: <id>` (preferred), or the OpenAI field `"user": "<id>"` in the body. The header wins when both are set.
- An id is at most 128 characters of `A-Za-z0-9._/-`. Other characters become `_` and the rest is cut off; an empty id counts as none.
- The agent is the part before the first `/`: `manager/task-123` and `manager/probe` are both agent `manager`. The full id
  (session or sub-agent) is kept too.
- Without an id the worker derives `auto:<8 hex digits>` from a hash of the prompt up to the end of the first user message
  (the system prompt and that message). Limitation: agents that open with the same system prompt and first message share an
  identity, and one that edits its system prompt gets a new one. Send an id when it matters.

The stats page shows an **Agents** table (requests running and waiting, requests, prompt and cached tokens; RAM and SSD
bytes are placeholders) and an Agent column in the Log. In `/ui/state` each request row has `agent` and `agent_id`, and the
`agents` block lists the aggregates.

- `AGENT_STATS` (default 1024): how many agents the per-agent counters keep; the least recently seen go first. Each entry is
  four integers (a few hundred bytes), so even 1024 is well under 1 MiB. Agents cut off from the table keep their tier bytes
  (the engine tags checkpoints with the agent's owner id), which the table cannot name: they show as one row
  **其他 N 個 agent（超出統計表）** with their count and RAM / SSD bytes. Owner 0 (checkpoints without an agent) is its own row,
  **未知（owner 0）**. The two rows add up to what the old single "unknown" row showed. Raise `AGENT_STATS` when the
  overflow row is not empty.

## Client counting

The server counts the requests of each client machine, so you can see who uses it. It is counting only: admission and
ordering stay as they are (one queue), and nothing is limited, capped or rejected here, and there are no API keys. Limits
are the job of the client machines' own agent programs (see `docs/agent-client-guide.md`).

Who a request comes from, in this order:

1. The header `X-Client-Id: <name>` (at most 64 characters: letters of any script, so a Chinese name works, digits and `. _ - /`; other characters become `_`; empty counts as none. HTTP headers carry bytes: send a non-ASCII name UTF-8 encoded; a client library that only accepts latin-1 header values should use an ASCII name).
2. The setting `CLIENT_NAMES` (environment, empty by default): a comma-separated list of `address=name` or `network/prefix=name`,
   e.g. `192.168.1.11=PC-A,192.168.1.12=PC-B,192.168.2.0/24=Lab`. An exact address is a `/32` (or `/128`); the longest matching
   prefix wins. Entries that do not parse are ignored.
3. The peer's IP address itself.

A loopback peer (127.0.0.0/8, `::1`: the stats page, the tray app, tests on the server's own machine) is `local`.
Behind a reverse proxy: the first entry of `X-Forwarded-For` replaces the peer address, but only when the immediate peer is a
loopback or private address and the header is present. A public peer cannot choose its address that way. The header
`X-Client-Id` is trusted as sent (there is no authentication): it is a label, not an identity check.
Any client on a private network can also choose its address through `X-Forwarded-For`: the counts are for looking, not for
security. With Docker, source addresses are kept when the port is published normally; a userland proxy or a rootless setup
makes every machine appear as the bridge gateway (a 172.x address): then give each machine an `X-Client-Id`.

Counted per client since the server's engine started (the table keeps at most 256 clients; the least recently seen go first):
requests, requests running and waiting (queued or restoring), prompt tokens, cached tokens (and the hit ratio), output tokens,
media tokens, errors, cancelled requests, the average queue wait (arrival until the prefill begins) and when the client was last seen.

Where to read it: `/ui/state` has a `clients` list (`client`, `running`, `waiting`, `waiting_tokens`, `requests`,
`prompt_tokens`, `cached_tokens`, `hit`, `output_tokens`, `media_tokens`, `avg_wait_s`, `errors`, `cancelled`, `first_seen_s`,
`last_seen` in seconds ago); each request row has `client`, and so does the finished-request record in the Log. `/metrics` has
labelled series `q27:client_requests_total{client="PC-A"}`, `q27:client_running`, `q27:client_waiting`,
`q27:client_prompt_tokens_total`, `q27:client_cached_tokens_total`, `q27:client_output_tokens_total`,
`q27:client_media_tokens_total`, `q27:client_errors_total`, `q27:client_cancelled_total`, `q27:client_queue_wait_seconds_total`
and `q27:client_queue_wait_count` (the average wait is their ratio). Docker: `CLIENT_NAMES` in `compose.yaml`.

The stats page (`/ui`) shows this as a **機器** (Machines) table right after the Agents table (running, waiting and the queued prompt tokens, requests, prompt and cached tokens with the hit ratio, output tokens, average queue wait, errors and cancelled, last seen; it is hidden when the server sends no `clients`), and the Log has a 機器 column with the record's `client`.

### Engine time split (引擎時間分配)

The stats page has a **引擎時間分配** card (hidden when the server sends no `engine_time`) showing where the engine loop's wall-clock
time goes, as one stacked bar for the last 60 s plus the share since the engine started. Prefill and decode take turns on the GPU
(they never run at the same time), so the shares add up to 100%. Kinds: `prefill` and `decode` (turns, single and batched; what
ran inside them is taken out), `restore` (a synchronous tier restore inside `prefix_begin`, while the loop is blocked; the
background restores of `SPILL_ASYNC_PROMOTE` are not loop time), `spill` (the tiered cache's tick and the idle backups), `media`
(image / video encode and the wait for their decoding), `idle` (waiting for a request or a restore) and `other` (the rest: admission,
polling, sending). Not separable: KV page writes and checkpoint saves the engine does inside a prefill or decode call stay in
those two. `/ui/state` has `engine_time` (`recent` and `total`: `wall`, `seconds`, `share` per kind, `prefill_tokens`,
`decode_tokens`); `/metrics` has `q27:engine_seconds_total{kind="prefill|decode|restore|spill|media|idle|other"}` (cumulative seconds
since the engine started).

## Request trace

Off by default. `REQUEST_TRACE=<directory or file name prefix>` (or `--request-trace`) records one JSON line per request start, request end
and every 10 s while busy, with counts and hashes only (no prompt or output text, no token ids), for a trace-driven simulator of the cache's
eviction policies. `REQUEST_TRACE_HASH_AGENTS=1` replaces agent ids by a short hash; `REQUEST_TRACE_MAX_MIB` (default 512) caps the total
size of the files (64 MiB each, the oldest are deleted). About 8 KB per request with 50-125K token prompts. Fields, privacy and
`tools/trace_summary.py`: [request-trace.md](request-trace.md) (Traditional Chinese).

## Prefix cache

The server takes checkpoints at message boundaries and at the end of each generation; the next request resumes from
the deepest match. The log line `[api] prompt N tokens, cached C of R reusable` and the `q27:cached_tokens_total` /
`q27:reusable_tokens_total` metrics show how well it hits. Clients must send the previous turns' reasoning back, since
the chat template renders it.

## Other checkpoints

`WEIGHTS=ORG/NAME[@REVISION]` serves another NVFP4 checkpoint of Qwen3.8-27B from the Hugging Face cache. The engine
takes the layouts it already reads: the modelopt one of NVIDIA's checkpoint (projections in FP8 or NVFP4) and the
compressed-tensors one of QUASAR, with NVFP4 activations (W4A4) and the model's dimensions. A BF16 `lm_head` (kept by
QUASAR and its fine-tunes) is quantized to NVFP4 at load, with the block scales chosen for the smallest error (on
QUASAR: KL to the BF16 model 0.0473, against 0.0471 with NVIDIA's NVFP4 `lm_head`). The compressed-tensors
"mixed-precision" layout of unsloth's and Red Hat's checkpoints loads as well: FP8 weights with per-output-channel
scales (the GDN and attention projections, the last eight MLPs, unsloth's `lm_head`) run as they are, with the
activations quantized per token at run time as the checkpoint specifies (`gemm_fp8_scaled`, `skinny_fp8_scaled`);
an FP8 `lm_head` gives the logits itself, and the DFlash2 draft reads an NVFP4 copy of it made at load (with the
draft vocabulary, only to build its own head: the copy then lives in the KV pool during the load and takes no VRAM
afterwards). Weight-only checkpoints
(NVFP4A16), GGUF files, other engines' formats and checkpoints that mix in other types are refused with the tensor at
fault. `scripts/download_weights.py` runs the engine's checks on the safetensors headers before it downloads anything
(it needs the engine library, `Q27_LIB`).

The stats page lists the downloaded checkpoints of the model under the start settings. The server keeps the tokenizer
of the weights it started with, so it starts an engine only on weights whose tokenizer has the same ids (vocabulary,
merges, added tokens); for a checkpoint with another one, start the server with `WEIGHTS` set to it. A
`tokenizer.json` that another transformers version rewrote (e.g. Huihui's abliterated checkpoint: another
pre-tokenizer pattern) counts as the same, and the server keeps its own. Prompts are rendered with the chat template
of the weights the engine runs (e.g. unsloth's, which merges several system messages and takes conversations without
a user message), switched with the engine. The DFlash2 draft was trained on the original
model: it stays correct on a fine-tune (the target model checks every token), but accepts fewer tokens the further
the fine-tune drifts.

## Windows package

The package is portable: what it writes stays in its folder (the one with `qwen3-27b-tray.exe`), so it needs a folder
you can write to, not Program Files. The exception is the start at logon, a value in the registry, only if you tick it.

| Path in the folder | What |
| --- | --- |
| `settings.ini` | Your settings (below); the package does not ship one, so an update does not touch it |
| `huggingface\` | The weights, 23 GB (`HF_HOME`) |
| `logs\` | The tray app's `server.log`, and one older file, `server.1.log`: a new one at each start of the server and beyond 16 MiB |
| `python\`, `app\`, `qwen3-27b-tray.exe`, `start.cmd`, `settings.example.ini`, `LICENSE` | The package; an update replaces them |

### Settings

Copy `settings.example.ini` to `settings.ini` and set any variable of this page in its `[env]` section ([How settings
work](#how-settings-work)). Both the tray app and `start.cmd` read it, when the server starts; the log says when
`settings.ini` overrides the environment. Relative paths are from the folder. The stats page does not write the file:
what you change there lasts until the server restarts, and the page shows the lines to add here to keep it. Only the
tray app's settings window writes it, when you click **Apply**, and only the lines of what you chose or typed there, also
a value equal to the default, which then stays when a later version changes the default (comments and other lines stay,
and it is saved as UTF-8).

`HF_HOME` is the exception: without it in `settings.ini` the weights are in the folder, whatever the environment says, so
that a variable of the whole PC does not move them out. To share an existing cache, write it there in full, for
example the one earlier versions downloaded to:

```ini
[env]
HF_HOME=C:\Users\me\.cache\huggingface
PORT=8001
```

Windows limits most paths to 260 characters and the weights' are about 140 below the folder: keep the folder's path under
about 110, or `HF_HOME` elsewhere.

| Variable | Default | Meaning |
| --- | --- | --- |
| `OPEN_UI` | 1 | `start.cmd`: 1 opens the web UI in the browser at the start, 0 does not |
| `WEIGHTS` | gdnfp4 | Weights downloaded before the start and served: a profile or `ORG/NAME[@REVISION]` ([Other checkpoints](#other-checkpoints)) |
| `HF_HOME` | `huggingface` in the folder | Hugging Face cache the weights are downloaded to and loaded from (23 GB) |
| `HF_TOKEN` | (none) | Sent with the downloads if set (needed for gated repositories; otherwise it only raises the rate limit) |

### Tray app

`qwen3-27b-tray.exe` does the same as `start.cmd` without a console: it runs `scripts/download_weights.py`, then the
server, in a job object, so that the server and the engine end with the app, even if the app is killed. A second start
of the app does nothing. If another server answers on the port, the app reports it and does not start one.

Its defaults differ from `start.cmd`: `ENGINE_AUTOSTART=0` and `ENGINE_IDLE_UNLOAD_S=1800`, so the service is up from the
start but holds VRAM only between the first request and 30 minutes after the last one ([Unloading when idle](#unloading-when-idle)).

The output goes to the log (the menu's **Show log**, which also works when the server did not start). **Start at
logon** is the value `qwen3-27b-5090` under `HKCU\Software\Microsoft\Windows\CurrentVersion\Run`; the app corrects its
path if the folder moved.

### Settings window

**Settings** in the tray menu edits `settings.ini`, without a browser:

| Group | Settings | Apply |
| --- | --- | --- |
| Server | `HOST`, `PORT`, `HF_HOME` | Restarts the server, which downloads the weights if the new `HF_HOME` does not have them |
| Images and videos | `IMAGE_TOKENS`, `VIDEO_MAX_TOKENS`, `VIDEO_MAX_FRAMES` (empty: the default) | Reloads the engine if it is loaded; otherwise the next load uses them |
| Tiered cache | `SPILL_RAM_GIB`, `SPILL_RAM_RETENTION_S`, `SPILL_SSD_GIB`, `SPILL_SSD_MIN_FREE_GIB`, `SPILL_SSD_MIN_TOKENS`, `SPILL_TTL_DAYS`, `VRAM_BACKUP_IDLE_S`, `SPILL_PINNED_GIB`, `SPILL_ASYNC_IO`, `SPILL_ASYNC_PROMOTE`, `SPILL_FAIR_AGENTS`, `SPILL_COLD_FIRST`, `SPILL_COLD_AFTER_S`, `SPILL_UNUSED_TTL_S`, `SPILL_SSD_WARM_ONLY`, `SPILL_PROBATION_GIB`, `CACHE_POLICY`, `SPILL_DIR` | Reloads the engine if it is loaded; otherwise the next load uses them |
| Engine | `WEIGHTS`, `KV_DTYPE`, `SPEC`, `VISION`, `MAX_SEQS`, `MAX_TOKENS`, `VRAM_RESERVE_GIB` | Reloads the engine if it is loaded (weights with another tokenizer: restarts the server); otherwise the next load uses them |
| When it holds the VRAM | `ENGINE_IDLE_UNLOAD_S` | Applies at once |
| When it holds the VRAM | `ENGINE_AUTOSTART`, `MIN_KV_TOKENS` | Restarts the server |

- The window says what **Apply** will do before you click it, and asks first if that fails requests in flight or empties
  the prefix cache. The values are saved even if the server then refuses them (for example, too little VRAM free now).
- Each engine choice shows the longest context and the KV pool it would leave, from the same planner as the stats page;
  the line below the choices sums up the settings as chosen. Without a server for that folder, `frontend/plan.py` gives
  the same answer (no GPU, nothing allocated).
- **Browse...** chooses the **Weights folder** in Explorer's dialog: the folder with `hub\` in it (choosing `hub\` itself
  counts as the folder above). **Weights** lists the checkpoints of the model there, also those of other repositories
  ([Other checkpoints](#other-checkpoints)), and follows the folder at once, before you apply it, with the estimates of
  that folder's weights. If the chosen weights or the draft are not there, the window says so (the draft's absence
  also means no estimate until `SPEC` is `none`) and asks first, as the server would download them.
- To share one cache with WSL, set `HF_HOME` to it on the Windows drive. `huggingface_hub` or `hf download` under WSL with
  a 9p mount of the drive leaves links (LX symlinks) that Windows programs cannot read; the window names those
  checkpoints, and **Fix the links** replaces the links of this model's checkpoints with hard links to the same files
  (`python\python.exe app\scripts\download_weights.py --fix-links`, with `HF_HOME` set). Both systems then read them as
  plain files, and nothing is copied or downloaded. It needs the links' files on an NTFS volume. A virtiofs mount
  (`virtiofs=true` in `.wslconfig`) and `scripts/download_weights.py` leave nothing to fix.
- All of them can be changed with the server stopped or failed, to repair a wrong port or folder.
- If the stats page changed something that `settings.ini` does not have, the window lists it, and **Use these** copies it
  into the window.
- Any other variable: **Open settings.ini**. The window reads the file when it opens.

### Updating

From v0.3.1, run `update.cmd` in the package's folder. It downloads the latest release's zip (an interrupted download
resumes when run again), checks its SHA-256 (the `update.cmd` of v0.6.2 and later stops when it cannot get the
SHA-256, instead of installing an unchecked zip), quits the tray app of this folder, replaces `python\`, `app\` and the
files next to them, and starts the tray app again if it was running. `settings.ini`, the weights and the log stay. If a
step fails, the old version is put back. A server started with `start.cmd` has to be closed by hand.

| Command | |
| --- | --- |
| `update.cmd` | The latest release (nothing to do when `app\VERSION` already has its version) |
| `update.cmd -Check` | Only show the installed version and the latest release's |
| `update.cmd -Tag v0.3.1` | That release, also an older one |
| `update.cmd -Force` | Install again when the version is the same |
| `update.cmd -Zip <file>` | A zip already downloaded (not checked against a SHA-256) |

By hand (and from v0.3.0, which has no `update.cmd`): quit the tray app, delete `python\` and `app\` (files of the old
version must not stay among the new ones), and unzip the new package over the folder (the zip's name has the version,
the folder in it does not).

## Development switches

All default to the faster setting. Except those marked static, they can be switched within one process with
`Engine.configure()` (`bench_spec.py --ab`).

| Variable | Meaning |
| --- | --- |
| `Q27_PROFILE=1` | GPU time per operation, also inside CUDA graphs (`Engine.profile_dump()`) |
| `Q27_NO_GRAPH=1` / `Q27_NO_PDL=1` / `Q27_NO_SKINNY=1` | Disable CUDA graphs / programmatic dependent launch / the skinny GEMMs |
| `Q27_SKINNY_CHECK=1` | Compare the skinny GEMMs with the prefill GEMMs and log the difference |
| `Q27_SKINNY_EXACT=1` | Split K for every row count as decode did before v0.6.2 (2 waves rounded up; a row's result stays independent of M) |
| `Q27_SKINNY_M1_UP=1` | The pre-v0.6.2 default: one-row calls (plain decode) split K into 2 waves rounded up, wider calls rounded down. Plain decode about 1% faster, but a token's result then depends on M, and greedy speculative decoding no longer reproduces `SPEC=none` exactly |
| `Q27_SKINNY_EXT_REDUCE_M` | Rows from which the skinny GEMM's split-K reduction runs as a separate kernel (default 1: M >= 2; at M = 1 the last CTA reduces; same result; P7-4) |
| `Q27_DRAFT_ATTN_TC=0` | The draft's attention on CUDA cores (default: tensor cores; P7-4) |
| `Q27_SKIP=norm,silu,gdn,attn,gemm,gemm4,gemm8,draft,dctx` | Skip a kind of kernel to measure its share of the time (`gemm4` / `gemm8`: FP4 / FP8 GEMMs only; `draft` / `dctx`: the draft model / its context update in a speculative step; the output is garbage) |
| `Q27_DECODE_MEMCPY=1` | Decode inputs and outputs by copies instead of mapped pinned memory |
| `Q27_ATTN_TC=0` | Decode / verify attention on CUDA cores (tree verification always uses them) |
| `Q27_ATTN_TC_DEC=0` | Decode (T = 1) attention with one warp per token (same result) |
| `Q27_ATTN_TC8_GRP=0` | With the FP8 MMAs (`Q27_ATTN_TC_F8`), compute T <= 2 in one warp instead of four (same result; static) |
| `Q27_KV_BLOCK=N` | Pages per block of the KV pool (P7-13; default the smallest count whose slice is >= 2 MiB: 57 for NVFP4, 32 for FP8, 16 for BF16). 1 lays pages out one by one (long-context decode 1.5% slower). Read at startup |
| `Q27_PAGES_HIGH=1` | Hand out KV pages from the top (tests, P7-13): the KV cache then occupies the parking areas, so every park moves pages. Read at startup |
| `Q27_KV_VMM=0` | Allocate the KV pool in one `cudaMalloc` instead of through CUDA's virtual memory management: then no KV pool memory is lent to video decoding, and NVDEC's surfaces are taken beyond the VRAM budget. Read at startup |
| `Q27_ATTN_PREFILL_TC=0` | Prefill attention on CUDA cores (static) |
| `Q27_GDN_CHUNK=0` / `1` | GDN in prefill: 0 = sequential (static); 1 = the chunk prep writes W and U (P5-2, read per call). Default 2: the prep writes only (I + L)^-1 and the chunk kernel forms delta itself (P5-14, prefill +1.9% at 19K) |
| `Q27_AB_TC=1` | Prefill in_proj_a / b (T > 8): one block per 32 tokens over all of K (before P5-16, read per call). Default: K split in 4 over a cluster, partial sums added in order (prompts of 128 / 512 tokens +7.4 / +6.4%) |
| `Q27_AB_NORM=0` | Prefill: the input RMSNorm writes its fp32 rows for in_proj_a / b (before P5-17, read per call). Default: it writes only 1 / rms per token and in_proj_a / b normalizes the residual itself (bit-identical; 19K +0.5%) |
| `Q27_MLP_FUSED=0` | Prefill MLP: gate \| up as a BF16-output GEMM, then SiLU and NVFP4 quantization in a separate kernel. Default 1: one GEMM (`gemm_mma.cu`) whose epilogue writes silu(gate) * up as the down GEMM's NVFP4 activations (prefill +2% at 19K / 73K, P5-12) |
| `Q27_MLP_BF16=0` | With `Q27_MLP_FUSED=0`: fp32 outputs for the prefill MLP's gate \| up |
| `Q27_SPEC_PATHS=K` | Paths of tree verification (default 1 = a chain; 2 / 4 accept more but take longer and break even) |
| `Q27_LOOKUP=1` | Lookup drafting (off: no gain with the 8-row chain) |
| `Q27_SPEC_PIPELINE=0` | Enqueue a speculative step after reading the previous result (default 1: before) |
| `Q27_SKINNY_WAVES` / `Q27_SKINNY_FIT` / `Q27_SKINNY_L2_CHUNKS` / `Q27_ATTN_WAVES` / `Q27_ATTN_MIN_KEYS` / `Q27_ATTN_PREFETCH` / `Q27_ATTN_TC_SPLITS` | Tuning |
| `Q27_LIB` / `Q27_TRACE` | Another build of `libq27.so` (e.g. for tracing; the default is `/build/engine/libq27.so`, on Windows `engine/build/q27.dll`) / where a `-DQ27_TRACE` build writes its timeline |
