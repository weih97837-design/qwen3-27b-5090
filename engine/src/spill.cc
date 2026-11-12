// Tiered cache (SPILL_RAM_GIB > 0, docs/tiered-cache.md): checkpoints the KV pool gives up go to host memory instead
// of being dropped, and from there to files (frontend/tiercache.py decides when; the RAM cap here is the backstop).
// prefix_begin brings the deepest spilled match back into the pool before its own lookup, so everything after that
// (the shared full pages, the copied partial page, the plan of the prefill) is the unchanged cache-hit path.
//
// All of this runs on the engine's thread, like the rest of Model (no locks).
//
// Two optional speed-ups, both off by default (Options::spill_pinned_gib, Options::spill_async_io):
//
// SPILL_PINNED_GIB: an entry's KV bytes live in one contiguous region of a pinned slab (PinnedPool) when one has room,
// else in pageable memory as before. Slabs are pinned only while idle (spill_grow_pinned). Pinned memory roughly doubles
// the PCIe speed of the copies; it changes nothing else (same host layout, same file format).
//
// SPILL_ASYNC_IO: the KV copies run on io_s_ instead of s_ and the CPU does not wait for them. Invariants:
//  (1) io_s_ READS or WRITES pool pages only after io_begin(): an event recorded on s_ after all work queued so far
//      (which may have written the pages, or still read pages that were just freed and handed out again) and waited
//      on by io_s_.
//  (2) A D2H copy of pages that are then freed (eviction): spill_copy records `done` on io_s_ and makes s_ wait for it
//      (cudaStreamWaitEvent, GPU side only) before returning, so no s_ work queued later can overwrite those pages
//      before they were read. An H2D copy into newly allocated pages likewise: s_ waits for its `done` before the
//      resumed sequence's kernels. Both directions share io_s_, so a H2D into a page that an earlier D2H still reads
//      is ordered after that read by the stream itself.
//  (3) The CPU does not touch an in-flight entry (Spilled::inflight): spill_wait (cudaEventSynchronize on `done`) runs
//      first in spill_write (so export and the RAM cap), spill_erase (so drop, subsume, the RAM cap's victims),
//      spill_promote, and for the idle backup before spill_backup_one returns. So an idle backup (live entry, its
//      pages stay in the pool and may be moved or freed later) is never in flight once it is visible to evict_checkpoint
//      or reserve_area; only entries of evicted checkpoints (pages already free, s_ already ordered) can be.
//      The promoted entry's host memory is read by the H2D copy after the entry is erased: it is parked in retired_
//      with its event and freed by spill_reap (the next spill call, and prefix_begin after its cudaStreamSynchronize(s_)
//      which also covers the copy, as s_ waited for it).
//  (4) The state part of an entry (ckpt_bytes_) stays a memcpy from the pinned slot after cudaStreamSynchronize(cs_).
//  Bytes of a checkpoint's last, partial page past its pos may change under a running D2H (the sequence that saved
//  it can keep writing there); they are never read (consumers copy the page and overwrite those rows).
//  With SPILL_ASYNC_IO=0 io_s_ is null, spill_async() is false, and every code path below is the synchronous one.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <new>
#include <sstream>
#include <system_error>

#include "model.h"

namespace q27 {

using namespace dims;

namespace {

constexpr char kMagic[8] = {'Q', '2', '7', 'K', 'V', 'C', '1', '\0'};
constexpr uint32_t kVersion = 1;

struct SpillHeader {
  char magic[8];
  uint32_t version;
  uint32_t kv_slices;
  uint64_t fingerprint;     // the model, the draft and the KV type (spill_fingerprint)
  uint64_t kv_page_bytes;
  uint64_t state_bytes;
  uint32_t page_tokens;
  uint32_t ring;
  uint32_t n_ids;
  uint32_t reserved;
  uint64_t kv_bytes;
};

constexpr size_t kIoChunk = size_t(256) << 20;

bool write_all(std::FILE* f, const void* p, size_t n) {
  const char* c = static_cast<const char*>(p);
  while (n) {
    const size_t k = std::min(n, kIoChunk);
    if (std::fwrite(c, 1, k, f) != k) return false;
    c += k, n -= k;
  }
  return true;
}

bool read_all(std::FILE* f, void* p, size_t n) {
  char* c = static_cast<char*>(p);
  while (n) {
    const size_t k = std::min(n, kIoChunk);
    if (std::fread(c, 1, k, f) != k) return false;
    c += k, n -= k;
  }
  return true;
}

struct File {
  std::FILE* f = nullptr;
  File(const std::string& path, const char* mode) : f(std::fopen(path.c_str(), mode)) {}
  ~File() {
    if (f) std::fclose(f);
  }
  bool close() {  // (fclose reports a write that failed late, e.g. a full disk)
    const bool ok = std::fclose(f) == 0;
    f = nullptr;
    return ok;
  }
};

uint64_t fnv(uint64_t h, const void* p, size_t n) {
  const unsigned char* c = static_cast<const unsigned char*>(p);
  for (size_t i = 0; i < n; i++) h = (h ^ c[i]) * 1099511628211ull;
  return h;
}

// Short checkpoints (a system prompt and tools) are shared by new conversations and cheap to keep, so they are
// never subsumed by a deeper one (only by an equal one).
constexpr int kKeepShort = 16384;

// d holds the tokens of c (at least as many, the same prefix) and, when c has the draft's ring, d has it too.
bool covers(const std::vector<int32_t>& d_ids, bool d_ring, const std::vector<int32_t>& c_ids, bool c_ring) {
  return d_ids.size() >= c_ids.size() && std::equal(c_ids.begin(), c_ids.end(), d_ids.begin()) && d_ring >= c_ring;
}

}  // namespace

uint64_t Model::spill_fingerprint() const {
  if (fingerprint_set_) return fingerprint_;  // computed once: the directory scan and config.json read are not free
  uint64_t h = 1469598103934665603ull;
  h = fnv(h, opt_.model_dir.data(), opt_.model_dir.size());
  h = fnv(h, "\0", 1);
  h = fnv(h, opt_.draft_dir.data(), opt_.draft_dir.size());
  const int kv = int(opt_.kv_type);
  h = fnv(h, &kv, sizeof kv);
  h = fnv(h, &kv_page_bytes_, sizeof kv_page_bytes_);
  h = fnv(h, &ckpt_bytes_, sizeof ckpt_bytes_);
  // The weights themselves: replacing the files at the same path must not reuse KV files computed with other
  // weights. The names and sizes of the files in each directory, and the bytes of the model's config.json.
  for (const std::string& dir : {opt_.model_dir, opt_.draft_dir}) {
    if (dir.empty()) continue;
    std::error_code ec;
    std::filesystem::directory_iterator it(dir, ec), end;
    if (ec) continue;
    std::vector<std::pair<std::string, uintmax_t>> files;
    for (; !ec && it != end; it.increment(ec)) {
      std::error_code e2;
      if (!it->is_regular_file(e2) || e2) continue;
      const uintmax_t size = it->file_size(e2);
      if (e2) continue;
      files.emplace_back(it->path().filename().string(), size);
    }
    std::sort(files.begin(), files.end());
    for (const auto& [name, size] : files) {
      h = fnv(h, name.data(), name.size());
      h = fnv(h, "\0", 1);
      h = fnv(h, &size, sizeof size);
    }
    h = fnv(h, "\0", 1);
  }
  if (!opt_.model_dir.empty()) {
    std::ifstream in(std::filesystem::path(opt_.model_dir) / "config.json", std::ios::binary);
    if (in) {
      const std::string bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
      h = fnv(h, bytes.data(), bytes.size());
    }
  }
  fingerprint_ = h;
  fingerprint_set_ = true;
  return h;
}

void Model::copy_pages(const std::vector<int32_t>& pages, char* host, bool to_host) {
  copy_pages_on(pages, host, to_host, s_);
  Q27_CUDA(cudaStreamSynchronize(s_));
}

void Model::copy_pages_on(const std::vector<int32_t>& pages, char* host, bool to_host, cudaStream_t st) {
  const size_t n = pages.size();
  for (int sl = 0; sl < kKvSlices; sl++)
    for (size_t i = 0; i < n;) {
      // Pages p, p + 1 of one block are adjacent in each slice (page_ref), so a run of them is one copy.
      size_t j = i;
      while (j + 1 < n && pages[j + 1] == pages[j] + 1 && pages[j + 1] / kv_block_ == pages[i] / kv_block_) j++;
      char* dev = kv_page(pages[i], sl);
      char* h = host + (size_t(sl) * n + i) * kv_page_bytes_;
      const size_t bytes = (j - i + 1) * kv_page_bytes_;
      if (to_host)
        Q27_CUDA(cudaMemcpyAsync(h, dev, bytes, cudaMemcpyDeviceToHost, st));
      else
        Q27_CUDA(cudaMemcpyAsync(dev, h, bytes, cudaMemcpyHostToDevice, st));
      i = j + 1;
    }
}

void Model::io_begin() {
  Q27_CUDA(cudaEventRecord(io_ev_, s_));
  Q27_CUDA(cudaStreamWaitEvent(io_s_, io_ev_));
}

void Model::spill_wait(Spilled& s) {
  if (!s.inflight) return;
  Q27_CUDA(cudaEventSynchronize(s.done));
  s.inflight = false;
}

cudaEvent_t Model::spill_event() {
  if (!ev_free_.empty()) {
    cudaEvent_t e = ev_free_.back();
    ev_free_.pop_back();
    return e;
  }
  cudaEvent_t e;
  Q27_CUDA(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
  return e;
}

// (Only events whose last record has completed: re-recording one later does not affect a cudaStreamWaitEvent that
// was queued before, which captured the earlier record.)
void Model::spill_event_put(cudaEvent_t ev) {
  if (ev) ev_free_.push_back(ev);
}

void Model::spill_reap() {
  for (Retired& r : retired_) {
    Q27_CUDA(cudaEventSynchronize(r.done));  // (done already, in practice: see the header comment)
    r.kv.reset();                            // (back into the pinned pool, or delete[])
    spill_event_put(r.done);
  }
  retired_.clear();
}

KvBuf Model::spill_alloc_kv(size_t bytes) {
  if (opt_.spill_pinned_gib > 0) {
    if (char* p = pin_pool_.alloc(bytes)) {
      sstats_.pin_hits++;
      return KvBuf(p, KvDel{&pin_pool_, bytes});
    }
    sstats_.pin_misses++;  // (the pool is not grown yet, full, or fragmented, or the entry is larger than a slab)
  }
  return KvBuf(new char[bytes], KvDel{});  // (may throw bad_alloc)
}

int Model::spill_grow_pinned(int chunks) {
  if (!spill_on() || opt_.spill_pinned_gib <= 0 || pin_failed_) return 0;
  const size_t cap = size_t(opt_.spill_pinned_gib * double(size_t(1) << 30));
  const size_t slab = size_t(std::max(opt_.spill_pinned_slab_mib, 16)) << 20;
  int added = 0;
  while (added < chunks && pin_pool_.total() < cap) {
    const size_t bytes = std::min(slab, cap - pin_pool_.total()) >> 20 << 20;  // (whole MiB)
    if (bytes == 0) break;
    void* p = nullptr;
    const cudaError_t e = cudaHostAlloc(&p, bytes, cudaHostAllocPortable);
    if (e != cudaSuccess) {  // (host memory is short: not fatal, the entries use pageable memory)
      cudaGetLastError();
      pin_failed_ = true;
      log("tiered cache: could not pin %zu MiB (%s); the pinned pool stays at %zu MiB", bytes >> 20,
          cudaGetErrorString(e), pin_pool_.total() >> 20);
      break;
    }
    pin_pool_.add_slab(static_cast<char*>(p), bytes);
    added++;
  }
  return added;
}

// Keeps only the deepest checkpoint of a conversation: the checkpoints of one long prompt share most of their
// pages, and spilling each would store those pages several times. Short checkpoints (up to kKeepShort tokens)
// are only subsumed by an equal one, since a new conversation often starts from them.
bool Model::spill_subsumed(const Checkpoint& c, uint64_t ignore_id) const {
  for (const auto& o : ckpts_)
    if (&o != &c && o.ids.size() > c.ids.size() && c.pos() > kKeepShort && covers(o.ids, o.ring, c.ids, c.ring))
      return true;
  for (const auto& kv : spill_) {
    const Spilled& s = kv.second;
    if (kv.first != ignore_id && covers(s.ids, s.ring, c.ids, c.ring) &&
        (c.pos() > kKeepShort || s.ids.size() == c.ids.size()))
      return true;
  }
  return false;
}

void Model::spill_checkpoint(const Checkpoint& c) {
  if (spill_subsumed(c, 0)) {
    sstats_.subsumed++;
    return;
  }
  for (auto it = spill_.begin(); it != spill_.end();) {
    auto next = std::next(it);
    const Spilled& e = it->second;
    if (!e.busy && covers(c.ids, c.ring, e.ids, e.ring) && (e.pos() > kKeepShort || e.ids.size() == c.ids.size())) {
      spill_erase(it, true);
      sstats_.subsumed++;
    }
    it = next;
  }

  if (!spill_copy(c, false)) return;  // (dropped, as without the tiered cache)
  sstats_.spilled++;
  spill_enforce_ram();
}

uint64_t Model::spill_copy(const Checkpoint& c, bool live) {
  spill_reap();
  Spilled s;
  s.kv_bytes = c.pages.size() * size_t(kKvSlices) * kv_page_bytes_;
  s.state_bytes = ckpt_bytes_;
  try {
    s.kv = spill_alloc_kv(s.kv_bytes);
    s.state.reset(new char[s.state_bytes]);
  } catch (const std::bad_alloc&) {
    sstats_.failed++;
    return 0;
  }
  if (!spill_async()) {
    copy_pages(c.pages, s.kv.get(), true);
    Q27_CUDA(cudaStreamSynchronize(cs_));  // (the state's copy into its host slot may still be on its way)
    std::memcpy(s.state.get(), slot_mem_[c.slot], ckpt_bytes_);
  } else {
    io_begin();  // (1)
    copy_pages_on(c.pages, s.kv.get(), true, io_s_);
    s.done = spill_event();
    Q27_CUDA(cudaEventRecord(s.done, io_s_));
    s.inflight = true;
    // (2) An eviction frees these pages right after this returns: nothing queued on s_ from now on may run before the
    // copy has read them. (A live backup keeps its pages, but is waited for below instead.)
    if (!live) Q27_CUDA(cudaStreamWaitEvent(s_, s.done));
    Q27_CUDA(cudaStreamSynchronize(cs_));  // (4) the state, meanwhile, as before
    std::memcpy(s.state.get(), slot_mem_[c.slot], ckpt_bytes_);
    if (live) spill_wait(s);  // (3) an idle backup completes before it is visible to evict_checkpoint / reserve_area
  }
  s.ids = c.ids;
  s.ring = c.ring;
  s.last_use = c.last_use;
  s.live = live;
  spill_ram_ += s.ram_bytes();
  const uint64_t id = spill_next_++;
  spill_.emplace(id, std::move(s));
  return id;
}

int Model::spill_backup_one() {
  if (!spill_on()) return 0;
  // The least recently used pool checkpoint with no valid backup that spill_checkpoint would not subsume.
  Checkpoint* pick = nullptr;
  for (auto& c : ckpts_) {
    if (c.backup && spill_.find(c.backup) != spill_.end()) continue;  // (a)
    if (pick && c.last_use >= pick->last_use) continue;
    if (!spill_subsumed(c, 0)) pick = &c;  // (b): as spill_checkpoint's rules
  }
  if (!pick) return 0;
  // A backup never makes spill_enforce_ram evict anything: the copy must fit under the RAM cap as it is. (Otherwise
  // the enforce would drop another backup, which the next call picks again: an endless loop.)
  const size_t cap = size_t(opt_.spill_ram_gib * double(size_t(1) << 30));
  const size_t bytes = pick->pages.size() * size_t(kKvSlices) * kv_page_bytes_ + ckpt_bytes_;
  if (spill_ram_ + bytes > cap) return 0;
  // (c): as spill_checkpoint, the shallower non-busy entries that the copy covers go (live ones too: their
  // checkpoints are in the pool, so nothing is lost).
  for (auto it = spill_.begin(); it != spill_.end();) {
    auto next = std::next(it);
    const Spilled& e = it->second;
    if (!e.busy && covers(pick->ids, pick->ring, e.ids, e.ring) && (e.pos() > kKeepShort || e.ids.size() == pick->ids.size())) {
      spill_erase(it, true);
      sstats_.subsumed++;
    }
    it = next;
  }
  const uint64_t id = spill_copy(*pick, true);
  if (id == 0) return 0;
  pick->backup = id;
  sstats_.backed_up++;
  return 1;
}

void Model::spill_lookup(const int32_t* ids, int n, bool spec) {
  int deepest = 0;
  for (const auto& c : ckpts_)
    if (c.pos() < n && (!spec || c.ring) && c.pos() > deepest && std::equal(c.ids.begin(), c.ids.end(), ids))
      deepest = c.pos();
  uint64_t best = 0;
  for (const auto& kv : spill_) {
    const Spilled& s = kv.second;
    if (s.live) continue;  // (an idle backup: its checkpoint is in the pool, and the pool's lookup has it)
    if (s.pos() < n && (!spec || s.ring) && s.pos() > deepest && std::equal(s.ids.begin(), s.ids.end(), ids))
      best = kv.first, deepest = s.pos();
  }
  if (best && !spill_promote(best))
    log("tiered cache: a spilled checkpoint of %d tokens could not be brought back (its prompt is computed again)",
        deepest);
}

bool Model::spill_promote(uint64_t id) {
  auto it = spill_.find(id);
  if (it == spill_.end() || it->second.live) return false;  // (a live backup is never restored: its original is here)
  spill_reap();
  Spilled& s = it->second;
  s.busy = true;
  spill_wait(s);  // (3) an entry evicted moments ago may still be on its way to host memory
  const bool from_disk = !s.in_ram();
  auto give_up = [&](bool forget) {
    s.busy = false;
    if (forget) {
      spill_erase(it, true);
      sstats_.failed++;
    } else if (from_disk) {  // (back to disk only)
      spill_ram_ -= s.ram_bytes();
      s.kv.reset(), s.state.reset();
    }
    return false;
  };
  if (from_disk && !spill_load(s)) return give_up(true);
  const int npages = int(ceil_div(s.pos(), kPageTokens));
  if (size_t(npages) * kKvSlices * kv_page_bytes_ != s.kv_bytes || s.state_bytes != ckpt_bytes_) return give_up(true);

  // Feasibility first: the free pages plus the pages only checkpoints hold (counted as cache_stats does) must cover
  // the pages needed, or the eviction below would drop checkpoints and still fail. Nothing is evicted then.
  {
    int freeable = pool_.free_pages();
    std::vector<int32_t> n(pool_.pages(), 0);
    for (const auto& c : ckpts_)
      for (int32_t p : c.pages) freeable += ++n[p] == pool_.refs(p);
    if (freeable < npages) return give_up(false);
  }

  // A state slot, then the pages; a full pool spills its least recently used checkpoints (this one is busy: kept).
  int slot = -1;
  for (;;) {
    for (size_t i = 0; i < slot_used_.size() && slot < 0; i++)
      if (slot_mem_[i] && !slot_used_[i]) slot = int(i);
    if (slot >= 0 || (!grow_state_cache(1) && !evict_checkpoint())) break;
  }
  if (slot < 0) return give_up(false);
  slot_used_[slot] = true;
  std::vector<int32_t> pages;
  while (int(pages.size()) < npages) {
    int p = pool_.alloc();
    while (p < 0 && evict_checkpoint()) p = pool_.alloc();
    if (p < 0) break;
    pages.push_back(p);
  }
  if (int(pages.size()) < npages) {
    for (int32_t p : pages) pool_.unref(p);
    slot_used_[slot] = false;
    return give_up(false);
  }
  if (!spill_async()) {
    copy_pages(pages, s.kv.get(), false);
  } else {
    // The pages may have been freed by a sequence whose kernels are still queued, or by an eviction whose D2H is
    // queued on io_s_ (ordered by the stream): io_begin covers the first, the single I/O stream the second.
    io_begin();  // (1)
    copy_pages_on(pages, s.kv.get(), false, io_s_);
    if (!s.done) s.done = spill_event();
    Q27_CUDA(cudaEventRecord(s.done, io_s_));
    s.inflight = true;
    Q27_CUDA(cudaStreamWaitEvent(s_, s.done));  // (2) the resumed sequence's kernels wait on the GPU, not the CPU
  }
  Q27_CUDA(cudaStreamSynchronize(cs_));
  std::memcpy(slot_mem_[slot], s.state.get(), ckpt_bytes_);
  Checkpoint ck;
  ck.ids = std::move(s.ids);
  ck.pages = std::move(pages);  // (one reference each, from alloc)
  ck.slot = slot;
  ck.ring = s.ring;
  ck.last_use = ++use_clock_;
  ckpts_.push_back(std::move(ck));
  if (s.inflight) {  // (3) the H2D copy still reads kv: park it until the copy is done (state was copied above)
    spill_ram_ -= s.ram_bytes();
    retired_.push_back({std::move(s.kv), s.done});
    s.done = nullptr, s.inflight = false;
  }
  spill_erase(it, true);  // (the pool's copy is the one now; it is spilled again when the pool gives it up)
  sstats_.restored++;
  return true;
}

void Model::spill_enforce_ram() {
  const size_t cap = size_t(opt_.spill_ram_gib * double(size_t(1) << 30));
  while (spill_ram_ > cap) {
    // An idle backup goes first: its checkpoint is still in the pool, so dropping it loses nothing, and it is never
    // written to the SSD (it is not the only copy).
    auto victim = spill_.end();
    for (auto it = spill_.begin(); it != spill_.end(); ++it)
      if (it->second.in_ram() && !it->second.busy && it->second.live &&
          (victim == spill_.end() || it->second.last_use < victim->second.last_use))
        victim = it;
    // (A victim may be an entry whose D2H copy is still in flight: spill_erase / spill_write wait for it, (3).)
    if (victim != spill_.end()) {
      spill_erase(victim, true);
      sstats_.dropped++;
      continue;
    }
    // Only sole copies are left: written to the directory, or dropped without one.
    for (auto it = spill_.begin(); it != spill_.end(); ++it)
      if (it->second.in_ram() && !it->second.busy && !it->second.live &&
          (victim == spill_.end() || it->second.last_use < victim->second.last_use))
        victim = it;
    if (victim == spill_.end()) return;
    if (!opt_.spill_dir.empty()) {
      const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                          std::chrono::system_clock::now().time_since_epoch()).count();
      const std::string path =
          opt_.spill_dir + "/" + std::to_string(ns) + "-" + std::to_string(victim->first) + ".q27kv";
      try {
        spill_write(victim->second, path);
        continue;
      } catch (const std::exception& e) {
        log("tiered cache: %s", e.what());
      }
    }
    spill_erase(victim, true);
    sstats_.dropped++;
  }
}

void Model::spill_write(Spilled& s, const std::string& path) {
  Q27_CHECK(s.in_ram(), "tiered cache: the entry is not in memory");
  spill_wait(s);  // (3) its D2H copy must be complete before the bytes are written
  const std::string tmp = path + ".tmp";
  SpillHeader h{};
  std::memcpy(h.magic, kMagic, sizeof kMagic);
  h.version = kVersion;
  h.kv_slices = kKvSlices;
  h.fingerprint = spill_fingerprint();
  h.kv_page_bytes = kv_page_bytes_;
  h.state_bytes = s.state_bytes;
  h.page_tokens = kPageTokens;
  h.ring = s.ring ? 1 : 0;
  h.n_ids = uint32_t(s.ids.size());
  h.kv_bytes = s.kv_bytes;
  bool ok;
  {
    File f(tmp, "wb");
    ok = f.f && write_all(f.f, &h, sizeof h) && write_all(f.f, s.ids.data(), s.ids.size() * sizeof(int32_t)) &&
         write_all(f.f, s.state.get(), s.state_bytes) && write_all(f.f, s.kv.get(), s.kv_bytes);
    ok = f.f && f.close() && ok;
  }
  if (!ok || std::rename(tmp.c_str(), path.c_str()) != 0) {
    std::remove(tmp.c_str());
    throw std::runtime_error("could not write " + path);
  }
  if (!s.path.empty() && s.path != path) std::remove(s.path.c_str());
  s.path = path;
  spill_ram_ -= s.ram_bytes();
  s.kv.reset(), s.state.reset();
  sstats_.written++;
}

bool Model::spill_load(Spilled& s) {
  File f(s.path, "rb");
  SpillHeader h{};
  if (!f.f || !read_all(f.f, &h, sizeof h) || std::memcmp(h.magic, kMagic, sizeof kMagic) != 0 ||
      h.version != kVersion || h.fingerprint != spill_fingerprint() || h.n_ids != s.ids.size() ||
      h.state_bytes != ckpt_bytes_ || h.kv_bytes != s.kv_bytes)
    return false;  // (checked before anything is allocated)
  std::vector<int32_t> ids(h.n_ids);
  if (!read_all(f.f, ids.data(), ids.size() * sizeof(int32_t)) || ids != s.ids) return false;
  try {
    spill_reap();
    s.state.reset(new char[h.state_bytes]);
    s.kv = spill_alloc_kv(h.kv_bytes);  // (assignment, not reset(new ...): the deleter differs)
  } catch (const std::bad_alloc&) {
    s.kv.reset(), s.state.reset();
    return false;
  }
  s.state_bytes = h.state_bytes, s.kv_bytes = h.kv_bytes;
  if (!read_all(f.f, s.state.get(), s.state_bytes) || !read_all(f.f, s.kv.get(), s.kv_bytes)) {
    s.kv.reset(), s.state.reset();
    return false;
  }
  spill_ram_ += s.ram_bytes();
  return true;
}

void Model::spill_erase(std::map<uint64_t, Spilled>::iterator it, bool remove_file) {
  spill_wait(it->second);  // (3) the buffers are freed below: an in-flight D2H must have finished writing them
  spill_event_put(it->second.done);
  spill_ram_ -= it->second.ram_bytes();
  if (remove_file && !it->second.path.empty()) std::remove(it->second.path.c_str());
  spill_.erase(it);
}

std::string Model::spill_list() const {
  std::ostringstream out;
  for (const auto& kv : spill_) {
    const Spilled& s = kv.second;
    out << kv.first << '\t' << s.pos() << '\t' << s.ram_bytes() << '\t' << (s.path.empty() ? 0 : 1) << '\t'
        << s.last_use << '\t' << (s.live ? 1 : 0) << '\t' << s.path << '\n';
  }
  return out.str();
}

void Model::spill_export(uint64_t id, const std::string& path) {
  auto it = spill_.find(id);
  Q27_CHECK(it != spill_.end(), "tiered cache: no such entry");
  Q27_CHECK(!it->second.busy, "tiered cache: the entry is being restored");
  spill_write(it->second, path);
}

void Model::spill_drop(uint64_t id) {
  auto it = spill_.find(id);
  if (it == spill_.end() || it->second.busy) return;
  spill_erase(it, true);
  sstats_.dropped++;
}

uint64_t Model::spill_import(const std::string& path) {
  if (!spill_on()) return 0;
  for (const auto& kv : spill_)
    if (kv.second.path == path) return 0;  // (already indexed)
  File f(path, "rb");
  SpillHeader h{};
  if (!f.f || !read_all(f.f, &h, sizeof h)) return 0;
  if (std::memcmp(h.magic, kMagic, sizeof kMagic) != 0 || h.version != kVersion || h.kv_slices != uint32_t(kKvSlices) ||
      h.fingerprint != spill_fingerprint() || h.kv_page_bytes != kv_page_bytes_ || h.state_bytes != ckpt_bytes_ ||
      h.page_tokens != uint32_t(kPageTokens) || h.n_ids == 0 || h.n_ids > uint32_t(opt_.max_tokens) + 64 ||
      h.kv_bytes != size_t(ceil_div(h.n_ids, kPageTokens)) * kKvSlices * kv_page_bytes_)
    return 0;
  // The file is exactly its header, the tokens, the state and the pages (a truncated or padded file is rejected).
  std::error_code ec;
  const uintmax_t file_bytes = std::filesystem::file_size(path, ec);
  if (ec || file_bytes != sizeof h + uintmax_t(h.n_ids) * sizeof(int32_t) + uintmax_t(h.state_bytes) +
                              uintmax_t(h.kv_bytes))
    return 0;
  Spilled s;
  s.ids.resize(h.n_ids);
  if (!read_all(f.f, s.ids.data(), s.ids.size() * sizeof(int32_t))) return 0;
  s.kv_bytes = h.kv_bytes;
  s.state_bytes = h.state_bytes;
  s.ring = h.ring != 0;
  s.path = path;
  s.last_use = 0;  // (older than anything this run used)
  const uint64_t id = spill_next_++;
  spill_.emplace(id, std::move(s));
  return id;
}

void Model::spill_stats(int64_t* out) const {
  int64_t ram = 0, disk = 0;
  for (const auto& kv : spill_) {
    ram += kv.second.in_ram();
    disk += !kv.second.path.empty();
  }
  out[0] = int64_t(spill_.size()), out[1] = ram, out[2] = disk, out[3] = int64_t(spill_ram_);
  out[4] = sstats_.spilled, out[5] = sstats_.restored, out[6] = sstats_.written, out[7] = sstats_.dropped;
  out[8] = sstats_.subsumed, out[9] = sstats_.failed, out[10] = sstats_.backed_up;
  out[11] = int64_t(pin_pool_.total()), out[12] = int64_t(pin_pool_.used());
  out[13] = sstats_.pin_hits, out[14] = sstats_.pin_misses;
}

}  // namespace q27
