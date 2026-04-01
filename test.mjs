#!/usr/bin/env node
/**
 * engram test script
 *
 * Usage:
 *   node test.mjs                          # standard baseline (Ollama, full history, O(n²) prefill)
 *   node test.mjs --engram                 # Engram server (persistent KV cache)
 *   node test.mjs [--ollama-host http://localhost:11434] [--ollama-model gpt-oss:20b] [--turns 10]
 *   node test.mjs --engram [--host http://localhost:8080] [--model /path/to/model.gguf] [--turns 8]
 *   node test.mjs --engram-chat [--host http://localhost:8080] [--model /path/to/model.gguf] [--turns 10]
 *
 * Requires Node 18+ (native fetch).
 */

import { parseArgs } from "node:util";
import { readFileSync } from "node:fs";

const { values: args } = parseArgs({
  options: {
    host: { type: "string", default: "http://localhost:8080" },
    model: {
      type: "string",
      default:
        "/Users/richardgustin/.ollama/models/blobs/sha256-f5074b1221da0f5a2910d33b642efa5b9eb58cfdddca1c79e16d7ad28aa2b31f",
    },
    turns: { type: "string", default: "3" },
    n_ctx: { type: "string", default: "2048" },
    gpu: { type: "string", default: "16" },
    "n-predict": { type: "string", default: "512" },
    engram: { type: "boolean", default: false },
    "engram-chat": { type: "boolean", default: false },
    "ollama-host": { type: "string", default: "http://localhost:11434" },
    "ollama-model": { type: "string", default: "mistral:7b" },
    "include-thinking": { type: "boolean", default: false },
  },
  strict: false,
});

const BASE = args.host;
const MODEL = args.model;
const N_TURNS = parseInt(args.turns, 10);
const N_CTX = parseInt(args.n_ctx, 10);
const N_GPU = parseInt(args.gpu, 10);
const SESSION_ID = `test-${Date.now()}`;

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

async function api(method, path, body) {
  const opts = { method, headers: { "Content-Type": "application/json" } };
  if (body !== undefined) opts.body = JSON.stringify(body);

  const res = await fetch(`${BASE}${path}`, opts);
  const text = await res.text();

  let json;
  try {
    json = JSON.parse(text);
  } catch {
    json = { raw: text };
  }

  if (!res.ok) {
    throw new Error(`${method} ${path} → ${res.status}: ${json.error ?? text}`);
  }
  return json;
}

const get = (path) => api("GET", path);
const post = (path, body) => api("POST", path, body);
const del = (path, body) => api("DELETE", path, body);

function bar(value, max, width = 30, fill = "█", empty = "░") {
  const n = Math.round((value / max) * width);
  return fill.repeat(n) + empty.repeat(width - n);
}

function fmt(ms) {
  if (ms == null) return "n/a";
  return ms >= 1000 ? `${(ms / 1000).toFixed(2)}s` : `${Math.round(ms)}ms`;
}

// ─────────────────────────────────────────────────────────────────────────────
// Test prompts
// ─────────────────────────────────────────────────────────────────────────────

const PROMPTS = JSON.parse(readFileSync("./prompts.json", "utf8"));

function getPrompt(i) {
  return PROMPTS[i % PROMPTS.length];
}

// ─────────────────────────────────────────────────────────────────────────────
// Conversation transcript printer
// ─────────────────────────────────────────────────────────────────────────────

const W = 72;

/**
 * Word-wrap `text` into lines of at most `width` chars, preserving newlines.
 */
function wordWrap(text, width) {
  const lines = [];
  for (const para of text.split("\n")) {
    if (para.length <= width) {
      lines.push(para);
      continue;
    }
    const words = para.split(" ");
    let current = "";
    for (const word of words) {
      if (!current) {
        current = word;
      } else if (current.length + 1 + word.length <= width) {
        current += " " + word;
      } else {
        lines.push(current);
        current = word;
      }
    }
    if (current) lines.push(current);
  }
  return lines.length ? lines : [""];
}

/**
 * Print a full conversation transcript.
 *
 * Each entry in `transcript` has:
 *   { turn, user, assistant, ms_prefill?, ms_decode?, cache_hit?, n_tokens_in_cache? }
 */
function printTranscript(transcript) {
  console.log();
  console.log("━".repeat(W));
  console.log("  CONVERSATION TRANSCRIPT");
  console.log("━".repeat(W));

  for (const t of transcript) {
    const timeParts = [
      t.ms_prefill != null ? `prefill=${fmt(t.ms_prefill)}` : null,
      t.ms_decode != null ? `decode=${fmt(t.ms_decode)}` : null,
    ]
      .filter(Boolean)
      .join("  ");

    const cachePart =
      t.cache_hit != null
        ? `  cache=${t.cache_hit ? "HIT ✓" : "MISS ✗"}  cached=${t.n_tokens_in_cache ?? "?"} tok`
        : "";

    console.log();
    console.log(`  ┌─ Turn ${t.turn}  ${timeParts}${cachePart}`);
    console.log(`  │`);

    // User
    console.log(`  │  👤 User:`);
    for (const line of wordWrap(t.user.trim(), 64)) {
      console.log(`  │     ${line}`);
    }
    console.log(`  │`);

    // Assistant
    console.log(`  │  🤖 Assistant:`);
    const reply = (t.assistant ?? "").trim() || "(no response)";
    for (const line of wordWrap(reply, 64)) {
      console.log(`  │     ${line}`);
    }
    console.log(`  └${"─".repeat(W - 4)}`);
  }

  console.log();
  console.log("━".repeat(W));
  console.log();
}

// ─────────────────────────────────────────────────────────────────────────────
// Baseline mode — hit Ollama directly, accumulate full history each turn
// ─────────────────────────────────────────────────────────────────────────────

async function runBaseline() {
  const OLLAMA_HOST = args["ollama-host"];
  const OLLAMA_MODEL = args["ollama-model"];

  console.log("╔══════════════════════════════════════════════╗");
  console.log("║       standard (baseline) — no engram        ║");
  console.log("╚══════════════════════════════════════════════╝");
  console.log();
  console.log(`  ollama host  : ${OLLAMA_HOST}`);
  console.log(`  model        : ${OLLAMA_MODEL}`);
  console.log(`  turns        : ${N_TURNS}`);
  console.log(
    `  include-thinking: ${args["include-thinking"] ? "yes (keeps <think> blocks in history)" : "no (stripped)"}`,
  );
  console.log();
  console.log(
    "  Each turn resends the FULL conversation history — prefill grows O(n²).",
  );
  console.log();

  const messages = [];
  const results = [];
  const transcript = [];
  let firstPrefill = null;

  console.log("─".repeat(W));
  console.log(
    ` ${"Turn".padEnd(5)} ${"History tok (est)".padEnd(20)} ${"Prefill".padEnd(12)} ${"Decode".padEnd(12)} Total`,
  );
  console.log("─".repeat(W));

  for (let i = 0; i < N_TURNS; i++) {
    const prompt = getPrompt(i);
    messages.push({ role: "user", content: prompt });

    const t0 = Date.now();
    const res = await fetch(`${OLLAMA_HOST}/api/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: OLLAMA_MODEL,
        messages: [...messages],
        stream: false,
        keep_alive: "0s",
        options: {
          temperature: 0.7,
          top_p: 0.9,
          top_k: 40,
          num_predict: parseInt(args["n-predict"] ?? "128"),
        },
      }),
    }).catch((e) => {
      console.error(`\n  Turn ${i + 1} FAILED (fetch): ${e.message}`);
      process.exit(1);
    });

    if (!res.ok) {
      const text = await res.text();
      console.error(`\n  Turn ${i + 1} FAILED: ${res.status} ${text}`);
      process.exit(1);
    }

    const data = await res.json();
    const wallMs = Date.now() - t0;

    const ms_load = data.load_duration != null ? data.load_duration / 1e6 : 0;
    const ms_prefill =
      data.prompt_eval_duration != null
        ? data.prompt_eval_duration / 1e6
        : null;
    const ms_decode =
      data.eval_duration != null ? data.eval_duration / 1e6 : null;
    const ms_total_prefill = ms_prefill != null ? ms_load + ms_prefill : null;
    const prompt_tokens = data.prompt_eval_count ?? "?";
    const gen_tokens = data.eval_count ?? "?";
    const rawReply = data.message?.content ?? "";
    const reply = args["include-thinking"]
      ? rawReply
      : rawReply.replace(/<think>[\s\S]*?<\/think>/g, "").trim();

    messages.push({ role: "assistant", content: reply });

    if (firstPrefill === null && ms_total_prefill !== null)
      firstPrefill = ms_total_prefill;

    const slowdown =
      firstPrefill && ms_total_prefill
        ? `${(ms_total_prefill / firstPrefill).toFixed(1)}×`
        : "—";

    results.push({
      i,
      prompt_tokens,
      gen_tokens,
      ms_load,
      ms_prefill,
      ms_total_prefill,
      ms_decode,
      wallMs,
    });
    transcript.push({
      turn: i + 1,
      user: prompt,
      assistant: reply,
      ms_prefill: ms_total_prefill,
      ms_decode,
    });

    console.log(
      ` ${String(i + 1).padEnd(5)}` +
        ` ${String(prompt_tokens).padEnd(20)}` +
        ` ${fmt(ms_total_prefill).padEnd(12)}` +
        ` ${fmt(ms_decode).padEnd(12)}` +
        ` ${fmt(wallMs)}  (${slowdown} vs turn 1)`,
    );
  }

  console.log("─".repeat(W));
  console.log();

  // ── Performance summary ───────────────────────────────────────────────────
  const maxPre = Math.max(...results.map((r) => r.ms_total_prefill ?? 0));
  const first = results[0];
  const last = results[results.length - 1];

  console.log("Performance Summary");
  console.log("───────────────────");
  console.log(
    `  Turn 1 prefill : ${fmt(first.ms_total_prefill)}  (${first.prompt_tokens} prompt tokens, load=${fmt(first.ms_load)})`,
  );
  console.log(
    `  Turn ${N_TURNS} prefill : ${fmt(last.ms_total_prefill)}  (${last.prompt_tokens} prompt tokens, load=${fmt(last.ms_load)})`,
  );
  if (first.ms_total_prefill && last.ms_total_prefill) {
    console.log(
      `  Slowdown (1→${N_TURNS}) : ${(last.ms_total_prefill / first.ms_total_prefill).toFixed(1)}×  ← this is what Engram eliminates`,
    );
  }
  console.log();
  console.log(
    "  Prefill latency per turn (load + prefill, grows with history):",
  );
  for (const r of results) {
    const pct = maxPre > 0 ? (r.ms_total_prefill ?? 0) / maxPre : 0;
    const b = bar(pct, 1, 28);
    console.log(
      `    [${r.i + 1}] ${b} ${fmt(r.ms_total_prefill)}  (${r.prompt_tokens} tok, load=${fmt(r.ms_load)})`,
    );
  }

  // ── Conversation transcript ───────────────────────────────────────────────
  printTranscript(transcript);

  console.log("✓ Baseline run complete.");
}

// ─────────────────────────────────────────────────────────────────────────────
// Engram chat mode — /session/chat with delta-aware KV reuse
// ─────────────────────────────────────────────────────────────────────────────

async function runEngramChat() {
  console.log("╔══════════════════════════════════════════════╗");
  console.log("║         engram chat — delta KV reuse         ║");
  console.log("╚══════════════════════════════════════════════╝");
  console.log();
  console.log(`  host       : ${BASE}`);
  console.log(`  session    : ${SESSION_ID}`);
  console.log(`  model      : ${MODEL || "(not specified)"}`);
  console.log(`  turns      : ${N_TURNS}`);
  console.log(`  n_ctx      : ${N_CTX}`);
  console.log(`  gpu layers : ${N_GPU}`);
  console.log();

  // ── 1. Health check ────────────────────────────────────────────────────────
  process.stdout.write("▶ GET /sessions ... ");
  const before = await get("/sessions").catch((e) => {
    console.error(`FAIL: ${e.message}`);
    process.exit(1);
  });
  console.log(
    `OK  (hot=${before.hot} warm=${before.warm} cold=${before.cold})`,
  );

  // ── 2. Create session ──────────────────────────────────────────────────────
  process.stdout.write(`▶ POST /session/create (${SESSION_ID}) ... `);
  const created = await post("/session/create", {
    session_id: SESSION_ID,
    model: MODEL,
    n_ctx: N_CTX,
    n_gpu_layers: N_GPU,
  }).catch((e) => {
    console.error(`FAIL: ${e.message}`);
    process.exit(1);
  });
  console.log(`OK  tier=${created.tier}`);
  console.log();

  // ── 3. Chat loop ───────────────────────────────────────────────────────────
  console.log("─".repeat(W));
  console.log(
    ` ${"Turn".padEnd(5)} ${"Cached tok".padEnd(12)} ${"Delta tok".padEnd(11)} ${"Prefill".padEnd(10)} ${"Decode".padEnd(10)} Hit`,
  );
  console.log("─".repeat(W));

  const messages = [];
  const results = [];
  const transcript = [];
  let firstPrefill = null;

  for (let i = 0; i < N_TURNS; i++) {
    const userMsg = getPrompt(i);
    messages.push({ role: "user", content: userMsg });

    const r = await post("/session/chat", {
      session_id: SESSION_ID,
      messages: [...messages],
      n_predict: parseInt(args["n-predict"] ?? "128"),
      temperature: 0.7,
      top_p: 0.9,
      top_k: 40,
    }).catch((e) => {
      console.error(`
  Turn ${i + 1} FAILED: ${e.message}`);
      process.exit(1);
    });

    // Accumulate assistant reply into history for next turn.
    if (r.text) messages.push({ role: "assistant", content: r.text });

    results.push(r);
    transcript.push({
      turn: i + 1,
      user: userMsg,
      assistant: r.text,
      ms_prefill: r.ms_prefill,
      ms_decode: r.ms_decode,
      cache_hit: r.cache_hit,
      n_tokens_in_cache: r.n_tokens_in_cache,
    });

    if (firstPrefill === null) firstPrefill = r.ms_prefill;
    const speedup =
      firstPrefill > 0 ? (firstPrefill / r.ms_prefill).toFixed(1) : "—";
    const hitMark = r.cache_hit ? "✓" : "✗";

    console.log(
      ` ${String(i + 1).padEnd(5)}` +
        ` ${String(r.n_tokens_in_cache).padEnd(12)}` +
        ` ${String(r.n_tokens_prompt).padEnd(11)}` +
        ` ${fmt(r.ms_prefill).padEnd(10)}` +
        ` ${fmt(r.ms_decode).padEnd(10)}` +
        ` ${hitMark}  ${speedup}×`,
    );
  }

  console.log("─".repeat(W));
  console.log();

  // ── 4. Performance summary ─────────────────────────────────────────────────
  const first = results[0];
  const last = results[results.length - 1];
  const maxPre = Math.max(...results.map((r) => r.ms_prefill));
  const totalSpeedup =
    first.ms_prefill > 0
      ? (first.ms_prefill / last.ms_prefill).toFixed(1)
      : "n/a";

  console.log("Performance Summary");
  console.log("───────────────────");
  console.log(
    `  First prefill   : ${fmt(first.ms_prefill)}  (${first.n_tokens_prompt} delta tokens, cold start)`,
  );
  console.log(
    `  Last prefill    : ${fmt(last.ms_prefill)}   (${last.n_tokens_prompt} delta tokens, cache=${last.n_tokens_in_cache})`,
  );
  console.log(`  Speedup (1→${N_TURNS})  : ${totalSpeedup}×`);
  console.log();
  console.log(
    "  Prefill latency per turn (delta tokens only — cache does the rest):",
  );
  for (let i = 0; i < results.length; i++) {
    const r = results[i];
    const pct = maxPre > 0 ? r.ms_prefill / maxPre : 0;
    const b = bar(pct, 1, 28);
    console.log(
      `    [${i + 1}] ${b} ${fmt(r.ms_prefill)}  (delta=${r.n_tokens_prompt} tok, cached=${r.n_tokens_in_cache})`,
    );
  }

  // ── 5. Conversation transcript ─────────────────────────────────────────────
  printTranscript(transcript);

  // ── 6. Session status ──────────────────────────────────────────────────────
  process.stdout.write(`▶ GET /session/status?session_id=${SESSION_ID} ... `);
  const status = await get(`/session/status?session_id=${SESSION_ID}`).catch(
    (e) => {
      console.error(`FAIL: ${e.message}`);
      process.exit(1);
    },
  );
  console.log(`OK`);
  console.log(
    `   tier=${status.tier}  tokens=${status.n_tokens_used}  created=${status.created_at}`,
  );
  console.log();

  // ── 7. Evict ───────────────────────────────────────────────────────────────
  process.stdout.write(`▶ DELETE /session/evict (${SESSION_ID}) ... `);
  const evicted = await del("/session/evict", { session_id: SESSION_ID }).catch(
    (e) => {
      console.error(`FAIL: ${e.message}`);
      process.exit(1);
    },
  );
  console.log(`OK  (${evicted.status})`);
  console.log();

  console.log("✓ Engram chat test complete.");
}

// ─────────────────────────────────────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────────────────────────────────────

if (args["engram-chat"]) {
  await runEngramChat();
  process.exit(0);
}

if (!args.engram) {
  await runBaseline();
  process.exit(0);
}

// ── Engram mode ──────────────────────────────────────────────────────────────

console.log("╔══════════════════════════════════════════════╗");
console.log("║           engram — inference test            ║");
console.log("╚══════════════════════════════════════════════╝");
console.log();
console.log(`  host       : ${BASE}`);
console.log(`  session    : ${SESSION_ID}`);
console.log(
  `  model      : ${MODEL || "(not specified — create will fail if required)"}`,
);
console.log(`  turns      : ${N_TURNS}`);
console.log(`  n_ctx      : ${N_CTX}`);
console.log(`  gpu layers : ${N_GPU}`);
console.log();

// ── 1. Health check ──────────────────────────────────────────────────────────
process.stdout.write("▶ GET /sessions ... ");
const before = await get("/sessions").catch((e) => {
  console.error(`FAIL: ${e.message}`);
  process.exit(1);
});
console.log(`OK  (hot=${before.hot} warm=${before.warm} cold=${before.cold})`);

// ── 2. Create session ────────────────────────────────────────────────────────
process.stdout.write(`▶ POST /session/create (${SESSION_ID}) ... `);
const created = await post("/session/create", {
  session_id: SESSION_ID,
  model: MODEL,
  n_ctx: N_CTX,
  n_gpu_layers: N_GPU,
}).catch((e) => {
  console.error(`FAIL: ${e.message}`);
  process.exit(1);
});
console.log(`OK  tier=${created.tier}`);
console.log();

// ── 3. Inference loop ────────────────────────────────────────────────────────
console.log("─".repeat(W));
console.log(
  ` ${"Turn".padEnd(5)} ${"Cache tokens".padEnd(14)} ${"Prompt tok".padEnd(12)} ${"Prefill".padEnd(10)} ${"Decode".padEnd(10)} Hit`,
);
console.log("─".repeat(W));

const results = [];
const transcript = [];
let firstPrefill = null;

for (let i = 0; i < N_TURNS; i++) {
  const prompt = getPrompt(i);

  const r = await post("/session/infer", {
    session_id: SESSION_ID,
    prompt,
    n_predict: parseInt(args["n-predict"] ?? "128"),
    temperature: 0.7,
    top_p: 0.9,
    top_k: 40,
  }).catch((e) => {
    console.error(`\n  Turn ${i + 1} FAILED: ${e.message}`);
    process.exit(1);
  });

  results.push(r);
  transcript.push({
    turn: i + 1,
    user: prompt,
    assistant: r.text,
    ms_prefill: r.ms_prefill,
    ms_decode: r.ms_decode,
    cache_hit: r.cache_hit,
    n_tokens_in_cache: r.n_tokens_in_cache,
  });

  if (firstPrefill === null) firstPrefill = r.ms_prefill;

  const speedup =
    firstPrefill > 0 ? (firstPrefill / r.ms_prefill).toFixed(1) : "—";
  const hitMark = r.cache_hit ? "✓" : "✗";

  console.log(
    ` ${String(i + 1).padEnd(5)}` +
      ` ${String(r.n_tokens_in_cache).padEnd(14)}` +
      ` ${String(r.n_tokens_prompt).padEnd(12)}` +
      ` ${fmt(r.ms_prefill).padEnd(10)}` +
      ` ${fmt(r.ms_decode).padEnd(10)}` +
      ` ${hitMark}  ${speedup}×`,
  );
}

console.log("─".repeat(W));
console.log();

// ── 4. Performance summary ───────────────────────────────────────────────────
const last = results[results.length - 1];
const first = results[0];
const maxPre = Math.max(...results.map((r) => r.ms_prefill));
const totalSpeedup =
  first.ms_prefill > 0
    ? (first.ms_prefill / last.ms_prefill).toFixed(1)
    : "n/a";

console.log("Performance Summary");
console.log("───────────────────");
console.log(
  `  First prefill   : ${fmt(first.ms_prefill)}  (${first.n_tokens_prompt} tokens, cold start)`,
);
console.log(
  `  Last prefill    : ${fmt(last.ms_prefill)}   (${last.n_tokens_prompt} tokens, cache=${last.n_tokens_in_cache})`,
);
console.log(`  Speedup (1→${N_TURNS})  : ${totalSpeedup}×`);
console.log();
console.log("  Prefill latency per turn:");
for (let i = 0; i < results.length; i++) {
  const r = results[i];
  const pct = maxPre > 0 ? r.ms_prefill / maxPre : 0;
  const b = bar(pct, 1, 28);
  console.log(`    [${i + 1}] ${b} ${fmt(r.ms_prefill)}`);
}

// ── 5. Conversation transcript ───────────────────────────────────────────────
printTranscript(transcript);

// ── 6. Session status ────────────────────────────────────────────────────────
process.stdout.write(`▶ GET /session/status?session_id=${SESSION_ID} ... `);
const status = await get(`/session/status?session_id=${SESSION_ID}`).catch(
  (e) => {
    console.error(`FAIL: ${e.message}`);
    process.exit(1);
  },
);
console.log(`OK`);
console.log(
  `   tier=${status.tier}  tokens=${status.n_tokens_used}  created=${status.created_at}`,
);
console.log();

// ── 7. Evict ─────────────────────────────────────────────────────────────────
process.stdout.write(`▶ DELETE /session/evict (${SESSION_ID}) ... `);
const evicted = await del("/session/evict", { session_id: SESSION_ID }).catch(
  (e) => {
    console.error(`FAIL: ${e.message}`);
    process.exit(1);
  },
);
console.log(`OK  (${evicted.status})`);
console.log();

console.log("✓ All tests passed.");
