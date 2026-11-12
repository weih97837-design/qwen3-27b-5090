// Host-only test of the retention rules (src/retention.h): superseded checkpoints and per-agent byte attribution. No CUDA, no GPU.
//   g++ -std=c++20 -O1 -Wall -Wextra -fsanitize=address,undefined -Isrc tests/test_retention.cc -o /tmp/test_retention && /tmp/test_retention
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <random>

#include "retention.h"

using namespace q27;

static int fails = 0;
#define CHECK(c)                                                  \
  do {                                                            \
    if (!(c)) {                                                   \
      std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c);    \
      if (++fails > 20) std::exit(1);                             \
    }                                                             \
  } while (0)

static std::vector<int32_t> toks(int32_t base, size_t n, int32_t from = 0) {
  std::vector<int32_t> v(n);
  for (size_t i = 0; i < n; i++) v[i] = base * 1000003 + int32_t(i) + from;
  return v;
}
// the first `n` tokens of `a`, then `extra` different ones
static std::vector<int32_t> extend(const std::vector<int32_t>& a, size_t n, size_t extra, int32_t salt) {
  std::vector<int32_t> v(a.begin(), a.begin() + std::ptrdiff_t(n));
  for (size_t i = 0; i < extra; i++) v.push_back(salt * 7919 + int32_t(i));
  return v;
}

static std::vector<char> run(const std::vector<std::vector<int32_t>>& ids, const std::vector<uint64_t>& owner,
                             const std::vector<bool>& ring = {}, size_t n_check = size_t(-1)) {
  std::vector<std::vector<uint64_t>> bh;
  for (const auto& v : ids) bh.push_back(block_hashes(v));
  std::vector<RetNode> nodes;
  for (size_t i = 0; i < ids.size(); i++) nodes.push_back({&ids[i], &bh[i], ring.empty() ? false : bool(ring[i]), owner[i]});
  std::vector<char> flags;
  superseded_flags(nodes, n_check == size_t(-1) ? nodes.size() : n_check, flags);
  return flags;
}

int main() {
  const size_t L = kKeepShortTokens;
  const auto base = toks(1, 3 * L);
  {  // a chain: A < B < C of one conversation: A and B are superseded, C is not
    const auto A = extend(base, L + 100, 0, 0), B = extend(base, 2 * L, 0, 0), C = base;
    const auto f = run({A, B, C}, {5, 5, 5});
    CHECK(f.size() == 3 && f[0] && f[1] && !f[2]);
  }
  {  // short prefixes (<= kKeepShort) are never superseded
    const auto S = extend(base, L, 0, 0), M = extend(base, L + 1, 0, 0), T = base;
    const auto f = run({S, M, T}, {5, 5, 5});
    CHECK(!f[0] && f[1] && !f[2]);
  }
  {  // a branch point: P is extended by X and Y along different paths: kept. X and Y are leaves.
    const auto P = extend(base, L + 500, 0, 0);
    const auto X = extend(P, P.size(), 300, 1), Y = extend(P, P.size(), 300, 2);
    const auto f = run({P, X, Y}, {5, 5, 5});
    CHECK(!f[0] && !f[1] && !f[2]);
    // ... but with one extender on a chain of its own beyond: P < X < X2 and Y branches at X: X is a branch point, P is not
    const auto X2 = extend(X, X.size(), 200, 3), Y2 = extend(X, X.size(), 100, 4);
    const auto g = run({P, X, X2, Y2}, {5, 5, 5, 5});
    CHECK(!g[0] && !g[1] && !g[2] && !g[3]);
  }
  {  // a shared prefix: extended by entries of two agents -> kept; the same entries of one agent -> superseded
    const auto P = extend(base, L + 500, 0, 0), D = extend(P, P.size(), 400, 1);
    CHECK(run({P, D}, {5, 5})[0] == 1);
    CHECK(run({P, D}, {5, 6})[0] == 0);  // (P is agent 5's, its extender agent 6's: the prefix is shared)
    const auto D2 = extend(D, D.size(), 50, 2);
    CHECK(run({P, D, D2}, {5, 5, 6})[0] == 0 && run({P, D, D2}, {5, 5, 6})[1] == 0);
    CHECK(run({P, D}, {0, 0})[0] == 1);  // (unknown agents are one bucket: not "two agents")
  }
  {  // the draft's ring: a deeper checkpoint without it does not stand in for a shallow one that has it
    const auto A = extend(base, L + 100, 0, 0), B = base;
    CHECK(run({A, B}, {5, 5}, {true, false})[0] == 0);
    CHECK(run({A, B}, {5, 5}, {false, true})[0] == 1);
    CHECK(run({A, B}, {5, 5}, {true, true})[0] == 1);
  }
  {  // equal copies (a live backup and its pool checkpoint) do not supersede each other; extenders past n_check count but are not checked
    const auto A = extend(base, L + 100, 0, 0), A2 = A, B = base;
    const auto f = run({A, A2, B}, {5, 5, 5}, {}, 2);  // (B is a pool checkpoint: extender only)
    CHECK(f.size() == 2 && f[0] && f[1]);
    const auto g = run({A, A2}, {5, 5});
    CHECK(!g[0] && !g[1]);
  }
  {  // scale and agreement with a brute-force definition on random chains
    std::mt19937 rng(9);
    std::vector<std::vector<int32_t>> ids;
    std::vector<uint64_t> owner;
    const auto root = toks(2, 6 * L);
    for (int i = 0; i < 60; i++) {
      const size_t cut = L / 2 + (rng() % (6 * L - L / 2));
      auto v = extend(root, rng() % 3 ? cut : 6 * L, rng() % 4 ? 0 : 1 + rng() % 5, int32_t(rng() % 4));
      ids.push_back(std::move(v));
      owner.push_back(1 + rng() % 2);
    }
    const auto f = run(ids, owner);
    for (size_t i = 0; i < ids.size(); i++) {
      bool want = false;
      if (int(ids[i].size()) > kKeepShortTokens) {
        std::vector<size_t> ext;
        for (size_t j = 0; j < ids.size(); j++)
          if (j != i && ids[j].size() > ids[i].size() && std::equal(ids[i].begin(), ids[i].end(), ids[j].begin())) ext.push_back(j);
        if (!ext.empty()) {
          size_t d = ext[0];
          for (size_t j : ext)
            if (ids[j].size() > ids[d].size()) d = j;
          want = true;
          for (size_t j : ext) {
            want = want && std::equal(ids[j].begin(), ids[j].end(), ids[d].begin()) && owner[j] == owner[i];
          }
        }
      }
      CHECK(bool(f[i]) == want);
    }
  }

  {  // attribution: own bytes whole; a chunk shared by entries of 2 agents is split between the 2 agents (not the entries)
    int c1, c2;
    UseEntry a{1, 100, {{&c1, 1000}, {&c2, 400}}};
    UseEntry b{1, 50, {{&c1, 1000}}};  // (the same agent again: it still counts once for c1)
    UseEntry c{2, 70, {{&c1, 1000}}};
    UseEntry d{0, 5, {}};
    const auto u = agent_use({a, b, c, d});
    CHECK(u.at(1) == 100 + 50 + 500 + 400);
    CHECK(u.at(2) == 70 + 500);
    CHECK(u.at(0) == 5);
    const double total = u.at(0) + u.at(1) + u.at(2);
    CHECK(total == 100 + 50 + 70 + 5 + 1000 + 400);  // (nothing lost, nothing counted twice)
    CHECK(agent_use({}).empty());
  }
  {  // the reuse count a continued conversation carries on (Model::prefix_begin -> seq_hits_ -> the deeper checkpoint)
    CHECK(inherited_hits(3 * kKeepShortTokens, 2) == 2);          // a long match: the continuation inherits it
    CHECK(inherited_hits(kKeepShortTokens, 5) == 0);              // a system prompt every conversation starts from: it does not
    CHECK(inherited_hits(kKeepShortTokens + 1, 1) == 1 && inherited_hits(100, 9) == 0 && inherited_hits(2 * kKeepShortTokens, 0) == 0);
  }
  {  // cost at the scale of the real tier (about 800 entries of 30-100K tokens, 16 agents with long shared system prompts, chains of
    // checkpoints per conversation): superseded_flags runs on the engine thread whenever the set of entries changed
    std::mt19937 rng(77);
    std::vector<std::vector<int32_t>> ids;
    std::vector<uint64_t> owner;
    for (int a = 0; a < 16; a++) {
      const auto sys = toks(100 + a, 30000);
      for (int c = 0; c < 10; c++) {
        auto conv = extend(sys, sys.size(), 40000 + rng() % 30000, int32_t(a * 100 + c));
        const size_t n = 3 + rng() % 4;  // the checkpoints of the conversation: prefixes of its final one
        for (size_t k = 1; k <= n; k++) {
          ids.push_back(std::vector<int32_t>(conv.begin(), conv.begin() + std::ptrdiff_t(sys.size() + (conv.size() - sys.size()) * k / n)));
          owner.push_back(uint64_t(a + 1));
        }
      }
    }
    const auto t_h = std::chrono::steady_clock::now();
    std::vector<std::vector<uint64_t>> bh;  // (once per entry in the engine: cached with it)
    for (const auto& v : ids) bh.push_back(block_hashes(v));
    const double ms_h = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_h).count();
    std::vector<RetNode> nodes;
    for (size_t i = 0; i < ids.size(); i++) nodes.push_back({&ids[i], &bh[i], false, owner[i]});
    std::vector<char> f;
    const auto t0 = std::chrono::steady_clock::now();
    superseded_flags(nodes, nodes.size(), f);
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    std::printf("block_hashes of all entries (once per entry, then cached): %.1f ms\n", ms_h);
    size_t sup = 0;
    for (char c : f) sup += c;
    std::printf("superseded_flags: %zu entries (%zu superseded), %.1f ms\n", ids.size(), sup, ms);
    CHECK(ids.size() > 600 && sup > 0 && sup < ids.size());
    CHECK(ms < 5000);  // (a sanity bound for sanitizer builds; the number printed above is the one to read, on an -O2 build)
  }
  std::printf(fails ? "FAILED (%d)\n" : "ok\n", fails);
  return fails ? 1 : 0;
}
