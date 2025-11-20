import express from "express";
import fetch from "node-fetch";
import crypto from "crypto";
import { URL } from "url";

/* ===================== ENV ===================== */
const {
  PORT = 3000,
  PUBLIC_BASE_URL,
  WAVESPEED_API_KEY,
  AIRTABLE_TOKEN,
  AIRTABLE_BASE_ID,
  AIRTABLE_TABLE = "Generations",

  // WaveSpeed v3
  WAVESPEED_BASE = "https://api.wavespeed.ai",
  WAVESPEED_SUBMIT_PATH = "/api/v3/bytedance/seedream-v4/edit-sequential",
  WAVESPEED_RESULT_PATH = "/api/v3/predictions",
  WAVESPEED_AUTH_HEADER = "Authorization",
} = process.env;

if (!PUBLIC_BASE_URL || !WAVESPEED_API_KEY || !AIRTABLE_TOKEN || !AIRTABLE_BASE_ID) {
  console.error("[BOOT] Missing required env vars. Need PUBLIC_BASE_URL, WAVESPEED_API_KEY, AIRTABLE_TOKEN, AIRTABLE_BASE_ID");
  process.exit(1);
}

/* ===================== APP & CONSTANTS ===================== */
const app = express();
app.use(express.urlencoded({ extended: true }));
app.use(express.json({ limit: "10mb" }));

const MODEL_NAME = "Seedream v4 (edit-sequential)";

const SUBMIT_MAX_RETRIES = 3;
const SUBMIT_BASE_DELAY_MS = 500;
const JOB_SPACING_MS = 1200;

const POLL_INTERVAL_MS = 7000;
const POLL_TIMEOUT_MS = 20 * 60 * 1000;
const POLL_MAX_RETRIES = 3;
const POLL_BASE_DELAY_MS = 800;

const MIN_PIXELS = 921600; // 960x960 Minimum
function ensureMinPixels(w, h, min = MIN_PIXELS) {
  if (w * h >= min) return { w, h };
  const s = Math.sqrt(min / (w * h));
  return { w: Math.ceil(w * s), h: Math.ceil(h * s) };
}

// Logs
console.log(`[CONF] WAVESPEED_BASE=${WAVESPEED_BASE}`);
console.log(`[CONF] SUBMIT=${WAVESPEED_BASE}${WAVESPEED_SUBMIT_PATH}`);
console.log(`[CONF] RESULT_BASE=${WAVESPEED_BASE}${WAVESPEED_RESULT_PATH}/:id/result`);
if (!/^https:\/\/api\.wavespeed\.ai/.test(WAVESPEED_BASE)) {
  console.warn("[WARN] WAVESPEED_BASE looks wrong. Use https://api.wavespeed.ai");
}

/* ===================== HELPERS ===================== */
const sleep = (ms) => new Promise((res) => setTimeout(res, ms));
const backoff = (attempt, base = 500) => Math.round(base * Math.pow(2, attempt) * (0.75 + Math.random() * 0.5));
const nowISO = () => new Date().toISOString();
const uuid = () => crypto.randomUUID?.() || crypto.randomBytes(16).toString("hex");

const splitCSV = (str) => (str || "").split(",").map((s) => s.trim()).filter(Boolean);

async function urlToDataURL(imageUrl) {
  let res;
  try { res = await fetch(imageUrl, { timeout: 30000 }); } catch (e) { throw new Error(`Fetch failed for ${imageUrl}: ${e.message}`); }
  if (!res.ok) throw new Error(`Non-200 for ${imageUrl}: ${res.status} ${res.statusText}`);
  const buf = Buffer.from(await res.arrayBuffer());
  const ct = res.headers.get("content-type") || "image/jpeg";
  return `data:${ct};base64,${buf.toString("base64")}`;
}

/* ===================== Airtable ===================== */
const AT_BASE = `https://api.airtable.com/v0/${AIRTABLE_BASE_ID}`;
const AT_URL = `${AT_BASE}/${encodeURIComponent(AIRTABLE_TABLE)}`;
const atHeaders = { Authorization: `Bearer ${AIRTABLE_TOKEN}`, "Content-Type": "application/json" };

async function atCreate(fields) {
  const r = await fetch(AT_URL, { method: "POST", headers: atHeaders, body: JSON.stringify({ fields }) });
  if (!r.ok) throw new Error(`Airtable create failed: ${r.status} ${await r.text()}`);
  const j = await r.json(); return j.id;
}
async function atGet(id) {
  const r = await fetch(`${AT_URL}/${id}`, { headers: atHeaders });
  if (!r.ok) throw new Error(`Airtable get failed: ${r.status} ${await r.text()}`);
  return r.json();
}
async function atPatch(id, fields) {
  const r = await fetch(`${AT_URL}/${id}`, { method: "PATCH", headers: atHeaders, body: JSON.stringify({ fields }) });
  if (!r.ok) throw new Error(`Airtable patch failed: ${r.status} ${await r.text()}`);
  return r.json();
}

const mergeIdString = (s, add) => {
  const set = new Set((s || "").split(",").map(t => t.trim()).filter(Boolean));
  for (const v of add) set.add(String(v));
  return [...set].join(", ");
};
const strToSet = (s) => new Set((s || "").split(",").map(t => t.trim()).filter(Boolean));

async function markCompletedIfReady(recordId) {
  const rec = await atGet(recordId);
  const req = strToSet(rec.fields?.["Request IDs"]);
  const seen = strToSet(rec.fields?.["Seen IDs"]);
  const done = req.size > 0 && req.size === [...req].filter(x => seen.has(x)).length;
  if (done) await atPatch(recordId, { Status: "completed", "Completed At": nowISO(), "Last Update": nowISO() });
}

/* ===================== WaveSpeed API ===================== */
function authHeader() {
  return (WAVESPEED_AUTH_HEADER.toLowerCase() === "authorization")
    ? { Authorization: `Bearer ${WAVESPEED_API_KEY}` }
    : { [WAVESPEED_AUTH_HEADER]: WAVESPEED_API_KEY };
}

// requestId -> parentRecordId
const memoryRequestMap = new Map();

async function submitGeneration({ prompt, subjectDataUrl, refDataUrls, width, height }, parentRecordId) {
  // Image order for face swapping:
  // - First reference (face) appears 3 times for maximum weight
  // - Then target pose image as subject
  // - Then any additional reference images
  let images;
  if (refDataUrls && refDataUrls.length > 0) {
    // Give face reference maximum weight by including it 3 times
    images = [
      refDataUrls[0],  // Face reference #1
      refDataUrls[0],  // Face reference #2
      refDataUrls[0],  // Face reference #3
      subjectDataUrl,  // Target pose/body image
      ...refDataUrls.slice(1)  // Any additional references
    ].filter(Boolean);
  } else {
    images = [subjectDataUrl];
  }

  const payload = {
    size: `${width}*${height}`,
    max_images: 1,
    enable_base64_output: false,
    enable_sync_mode: false,
    seed: 42,
    prompt: ''Recreate image 4, using the identity from images 1, 2 and 3. Keep clothes, pose, background, body and camera quality from image 4, don't keep clothes or accesories from images 1, 2 or 3. Make her ass exactly like image 4'.',
    negative_prompt: '',
    images: images
  };

  const url = new URL(`${WAVESPEED_BASE}${WAVESPEED_SUBMIT_PATH}`);
  url.searchParams.set("webhook", `${PUBLIC_BASE_URL}/webhooks/wavespeed`);

  const res = await fetch(url.toString(), {
    method: "POST",
    headers: { "Content-Type": "application/json", ...authHeader() },
    body: JSON.stringify(payload)
  });

  if (!res.ok) throw new Error(`WaveSpeed submit failed: ${res.status} ${await res.text()}`);
  const json = await res.json();

  const requestId = json?.data?.id || json?.requestId || json?.id || json?.request_id;
  if (!requestId) throw new Error(`Missing requestId in response: ${JSON.stringify(json)}`);

  if (parentRecordId) memoryRequestMap.set(requestId, parentRecordId);
  return requestId;
}

async function getResult(requestId) {
  const r = await fetch(`${WAVESPEED_BASE}${WAVESPEED_RESULT_PATH}/${encodeURIComponent(requestId)}/result`, { headers: { ...authHeader() } });
  if (!r.ok) throw new Error(`WaveSpeed result failed: ${r.status} ${await r.text()}`);
  const json = await r.json();
  const data = json.data || json;

  const status = (data.status || json.status || json.state || "processing").toLowerCase();
  let outs = [];
  if (Array.isArray(data.output)) outs = data.output;
  else if (Array.isArray(data.outputs)) outs = data.outputs;
  else if (typeof data.output === "string") outs = [data.output];
  else if (typeof json.output === "string") outs = [json.output];

  return { status, outputs: (outs || []).map(String).filter(Boolean), raw: json };
}

async function appendOutputsToAirtable(recordId, { outputUrls = [], requestId, failed = false }) {
  const rec = await atGet(recordId);
  const f = rec.fields || {};
  const urls = (outputUrls || []).map(String).filter(Boolean);
  const prev = Array.isArray(f["Output"]) ? f["Output"] : [];

  const newOutputs = [...prev, ...urls.map((u, i) => ({ url: u, filename: `output_${requestId}_${i + 1}.png` }))];
  const newSeen = mergeIdString(f["Seen IDs"], requestId ? [requestId] : []);
  const newFailed = failed ? mergeIdString(f["Failed IDs"], requestId ? [requestId] : []) : f["Failed IDs"];

  await atPatch(recordId, {
    Output: newOutputs,
    "Output URL": urls[0] || f["Output URL"] || "",
    "Seen IDs": newSeen,
    "Failed IDs": newFailed || "",
    "Last Update": nowISO()
  });

  await markCompletedIfReady(recordId);
}

/* ===================== Poller ===================== */
const _lastStatus = new Map();

async function pollUntilDone(requestId, parentRecordId) {
  const start = Date.now();
  let attempts = 0;
  let first = true;

  while (Date.now() - start < POLL_TIMEOUT_MS) {
    try {
      const { status, outputs, raw } = await getResult(requestId);
      const prev = _lastStatus.get(requestId);
      if (first) { console.log(`[POLL INIT] ${requestId} status=${status}`, JSON.stringify(raw?.data || raw)); first = false; }
      if (prev !== status) { console.log(`[POLL] ${requestId} ${prev || "(none)"} -> ${status}`); _lastStatus.set(requestId, status); }

      if (["completed", "succeeded", "success"].includes(status)) {
        console.log(`[POLL DONE] ${requestId} outputs=${outputs.length}`);
        await appendOutputsToAirtable(parentRecordId, { outputUrls: outputs, requestId, failed: false });
        return;
      }
      if (["failed", "error"].includes(status)) {
        console.warn(`[POLL FAIL] ${requestId}`);
        await appendOutputsToAirtable(parentRecordId, { outputUrls: [], requestId, failed: true });
        return;
      }
      await sleep(POLL_INTERVAL_MS);
    } catch (err) {
      console.warn(`[POLL ERROR] ${requestId}: ${err.message || err}`);
      if (attempts < POLL_MAX_RETRIES) { await sleep(backoff(attempts++, POLL_BASE_DELAY_MS)); }
      else { await appendOutputsToAirtable(parentRecordId, { outputUrls: [], requestId, failed: true }); return; }
    }
  }
  console.warn(`[POLL TIMEOUT] ${requestId}`);
  await appendOutputsToAirtable(parentRecordId, { outputUrls: [], requestId, failed: true });
}

/* ===================== UI (Demo) ===================== */
app.get("/", (_, res) => res.send("WaveSpeed x Airtable is running. /app to start a batch, or trigger via Airtable button."));

app.get("/app", (_, res) => {
  res.setHeader("Content-Type", "text/html; charset=utf-8");
  res.end(`<!doctype html>
<html><head><meta charset="utf-8"/><title>Batch Runner</title></head>
<body style="font-family:system-ui;padding:24px;max-width:760px;margin:auto">
<h1>Seedream v4 — Face Swap</h1>
<form method="POST" action="/generate-batch">
<label>Prompt (optional - will use optimized default)</label><br/>
<textarea name="prompt" rows="3"></textarea><br/><br/>
<label>Face Image URL (source face)</label><br/>
<input name="subject_url" type="url" required/><br/><br/>
<label>Target Pose Image URL (body/pose/scene to recreate)</label><br/>
<input name="reference_urls" type="text" required/><br/><br/>
<label>Width</label><input name="width" type="number" value="1024" required/>
<label style="margin-left:12px">Height</label><input name="height" type="number" value="1344" required/>
<label style="margin-left:12px">Batch</label><input name="batch_count" type="number" value="4" min="1" max="24" required/><br/><br/>
<button type="submit">Start Batch</button>
<p><small>Webhook: ${PUBLIC_BASE_URL}/webhooks/wavespeed</small></p>
</form></body></html>`);
});

/* ===================== Core Batch ===================== */
async function startRunFromRecord(recordId, opts = {}) {
  const rec = await atGet(recordId);
  const f = rec.fields || {};

  // Subject = FACE image (what we want to transfer)
  // References = BODY/POSE images (target scenes)
  const faceUrl = f["Subject"]?.[0]?.url || f["Subject URL"] || "";
  const poseImages = Array.isArray(f["References"]) ? f["References"].map(x => x.url) : [];
  const refCsv = splitCSV(f["Reference URLs"] || "");
  const allPoseUrls = [...poseImages, ...refCsv].filter(Boolean);

  if (!faceUrl || allPoseUrls.length === 0) throw new Error("Record needs Subject (face) + References (pose/body images)");

  // Size - hardcoded to match extension
  let W = 2572, H = 3576;
  const sizeStr = String(f["Size"] || "");
  const m = sizeStr.match(/(\d+)\s*[xX*]\s*(\d+)/);
  if (m) { W = +m[1]; H = +m[2]; }

  // Convert images to data URLs
  const faceDataUrl = await urlToDataURL(faceUrl);
  const refDataUrls = [faceDataUrl]; // Face as reference (will be duplicated for weight)

  // First pose image becomes subject
  const subjectDataUrl = await urlToDataURL(allPoseUrls[0]);

  // Additional pose images as extra references (if any)
  for (let i = 1; i < allPoseUrls.length; i++) {
    try { refDataUrls.push(await urlToDataURL(allPoseUrls[i])); } catch (e) { console.warn("[REF WARN]", allPoseUrls[i], e.message); }
  }

  // Status reset
  await atPatch(recordId, { Status: "processing", "Request IDs": "", "Seen IDs": "", "Failed IDs": "", "Last Update": nowISO(), Model: MODEL_NAME, Size: `${W}x${H}` });

  // Batch-Count
  const N = Math.max(1, Math.min(24, Number(f["Batch Count"] || opts.batch || 3)));

  const requestIds = [];
  for (let i = 0; i < N; i++) {
    let rid = null, lastErr = null;
    for (let a = 0; a < SUBMIT_MAX_RETRIES && !rid; a++) {
      try {
        rid = await submitGeneration({ prompt: "", subjectDataUrl, refDataUrls, width: W, height: H }, recordId);
      } catch (err) {
        lastErr = err; await sleep(backoff(a, SUBMIT_BASE_DELAY_MS));
      }
    }
    if (!rid) { console.error("[SUBMIT FAIL]", lastErr?.message || lastErr); continue; }
    requestIds.push(rid);
    if (i < N - 1) await sleep(JOB_SPACING_MS);
  }

  await atPatch(recordId, { "Request IDs": requestIds.join(", "), "Last Update": nowISO() });

  // Poll
  requestIds.forEach(rid => pollUntilDone(rid, recordId).catch(e => console.error("[POLL ERROR]", rid, e.message)));

  return { recordId, submitted: requestIds.length, request_ids: requestIds };
}

/* POST from Airtable (Automations/Scripting) */
app.post("/airtable/run", async (req, res) => {
  try {
    const { recordId, batch } = req.body || {};
    if (!recordId) return res.status(400).json({ error: "Missing recordId" });
    const out = await startRunFromRecord(recordId, { batch });
    res.json({ ok: true, ...out });
  } catch (e) {
    console.error("[/airtable/run ERROR]", e);
    res.status(500).json({ error: e.message || String(e) });
  }
});

/* GET variant for Button field (opens a URL) */
app.get("/airtable/run/:recordId", async (req, res) => {
  try {
    const out = await startRunFromRecord(req.params.recordId, { batch: Number(req.query.batch || 0) });
    res.setHeader("Content-Type", "text/html; charset=utf-8");
    res.end(`<html><body style="font-family:system-ui;padding:24px">
      <h2>Run started</h2>
      <p>Record: ${out.recordId}</p>
      <p>Submitted: ${out.submitted}</p>
      <p>Request IDs: ${out.request_ids.join(", ")}</p>
      <p>You can close this tab.</p>
    </body></html>`);
  } catch (e) {
    res.status(500).send(String(e.message || e));
  }
});

/* Existing batch form endpoint */
app.post("/generate-batch", async (req, res) => {
  try {
    const { prompt, subject_url, reference_urls, width, height, batch_count } = req.body;
    if (!subject_url || !reference_urls || !width || !height || !batch_count) return res.status(400).json({ error: "Missing fields." });

    const refs = splitCSV(reference_urls);
    let { w: W, h: H } = ensureMinPixels(Number(width), Number(height));

    const subjectDataUrl = await urlToDataURL(subject_url);
    const refDataUrls = [];
    for (const r of refs) { try { refDataUrls.push(await urlToDataURL(r)); } catch (e) { console.warn("[REF WARN]", r, e.message); } }

    const runId = uuid();
    const parentId = await atCreate({
      Prompt: String(prompt || ""),
      Subject: [{ url: subject_url }],
      References: refs.map((u, i) => ({ url: u, filename: `ref_${i + 1}` })),
      Output: [],
      "Output URL": "",
      Model: MODEL_NAME,
      Size: `${W}x${H}`,
      "Request IDs": "",
      "Seen IDs": "",
      "Failed IDs": "",
      Status: "processing",
      "Run ID": runId,
      "Created At": nowISO(),
      "Last Update": nowISO(),
    });

    const N = Math.max(1, Math.min(100, Number(batch_count)));
    const requestIds = [];
    for (let i = 0; i < N; i++) {
      let rid = null, lastErr = null;
      for (let a = 0; a < SUBMIT_MAX_RETRIES && !rid; a++) {
        try {
          rid = await submitGeneration({ prompt: prompt || "", subjectDataUrl, refDataUrls, width: W, height: H }, parentId);
        } catch (err) {
          lastErr = err; await sleep(backoff(a, SUBMIT_BASE_DELAY_MS));
        }
      }
      if (!rid) console.error(`[SUBMIT FAIL] job ${i + 1}:`, lastErr?.message || lastErr);
      else requestIds.push(rid);
      if (i < N - 1) await sleep(JOB_SPACING_MS);
    }

    await atPatch(parentId, { "Request IDs": requestIds.join(", "), "Last Update": nowISO() });
    requestIds.forEach(rid => pollUntilDone(rid, parentId).catch(e => console.error("[POLL ERROR]", rid, e.message)));

    res.json({ ok: true, parent_record_id: parentId, run_id: runId, submitted: requestIds.length, request_ids: requestIds });
  } catch (err) {
    console.error("[/generate-batch ERROR]", err);
    res.status(500).json({ error: err.message || String(err) });
  }
});

/* Webhook from WaveSpeed */
app.post("/webhooks/wavespeed", async (req, res) => {
  try {
    const b = req.body || {};
    const requestId = b.request_id || b.id || b.requestId || b.request || req.query.request_id;
    const status = (b.status || b.state || "").toLowerCase();
    const outputs = Array.isArray(b.output) ? b.output : Array.isArray(b.outputs) ? b.outputs : typeof b.output === "string" ? [b.output] : [];

    const parentRecordId = memoryRequestMap.get(requestId);
    if (!requestId || !parentRecordId) {
      console.warn(`[WEBHOOK] Unknown parent for requestId=${requestId}. Returning 202.`);
      return res.status(202).json({ ok: false, reason: "Unknown parent; poller will handle." });
    }

    if (["completed", "succeeded", "success"].includes(status)) {
      await appendOutputsToAirtable(parentRecordId, { outputUrls: outputs, requestId, failed: false });
    } else if (["failed", "error"].includes(status)) {
      await appendOutputsToAirtable(parentRecordId, { outputUrls: [], requestId, failed: true });
    }
    res.json({ ok: true });
  } catch (e) {
    console.error("[/webhooks/wavespeed ERROR]", e);
    res.status(500).json({ error: e.message || String(e) });
  }
});

/* Debug */
app.get("/debug/prediction/:id", async (req, res) => {
  try {
    const r = await fetch(`${WAVESPEED_BASE}${WAVESPEED_RESULT_PATH}/${encodeURIComponent(req.params.id)}/result`, { headers: { ...authHeader() } });
    const text = await r.text();
    res.status(r.status).type("application/json").send(text);
  } catch (e) {
    res.status(500).json({ error: String(e.message || e) });
  }
});

/* START */
app.listen(PORT, () => {
  console.log(`[BOOT] Server running on http://localhost:${PORT}`);
  console.log(`[BOOT] Public base URL: ${PUBLIC_BASE_URL}`);
  console.log(`[BOOT] Webhook listening at: ${PUBLIC_BASE_URL}/webhooks/wavespeed`);
});
