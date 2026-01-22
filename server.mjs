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

  // WaveSpeed v3 - Updated to v4.5/edit (non-sequential)
  WAVESPEED_BASE = "https://api.wavespeed.ai",
  WAVESPEED_SUBMIT_PATH = "/api/v3/bytedance/seedream-v4.5/edit",
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

const MODEL_NAME = "Seedream v4.5 (edit)";

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

async function submitGeneration({ faceDataUrl, targetDataUrl, width, height }, parentRecordId) {
  // v4.5/edit API - simple payload with just 2 images:
  // - img1 = face source (identity to transfer)
  // - img2 = target pose/body image (scene to recreate)
  // NO duplicates - the API handles face weighting internally
  
  const images = [faceDataUrl, targetDataUrl].filter(Boolean);
  
  if (images.length < 2) {
    throw new Error("Need both face image and target image");
  }

  const payload = {
    prompt: 'Recreate img2 using the face identity from img1. Transfer ONLY the facial features and hair (color, style, texture) from img1. Copy everything else exactly from img2: body proportions, pose, angle, clothing, accessories, background, lighting, composition. If img2 shows genitals, recreate them exactly as shown. Natural amateur photography, iPhone quality, visible skin texture, realistic lighting, seamless integration.',
    images: images,
    size: `${width}*${height}`,
    enable_sync_mode: false,
    enable_base64_output: false
  };

  const url = new URL(`${WAVESPEED_BASE}${WAVESPEED_SUBMIT_PATH}`);
  url.searchParams.set("webhook", `${PUBLIC_BASE_URL}/webhooks/wavespeed`);

  console.log(`[SUBMIT] Sending to ${url.toString()}`);
  console.log(`[SUBMIT] Images: ${images.length}, Size: ${width}*${height}`);

  const res = await fetch(url.toString(), {
    method: "POST",
    headers: { "Content-Type": "application/json", ...authHeader() },
    body: JSON.stringify(payload)
  });

  if (!res.ok) {
    const errorText = await res.text();
    console.error(`[SUBMIT ERROR] ${res.status}: ${errorText}`);
    throw new Error(`WaveSpeed submit failed: ${res.status} ${errorText}`);
  }
  
  const json = await res.json();
  console.log(`[SUBMIT RESPONSE]`, JSON.stringify(json));

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
app.get("/", (_, res) => res.send("WaveSpeed x Airtable v4.5 is running. /app to start a batch, or trigger via Airtable button."));

app.get("/app", (_, res) => {
  res.setHeader("Content-Type", "text/html; charset=utf-8");
  res.end(`<!doctype html>
<html><head><meta charset="utf-8"/><title>Batch Runner</title></head>
<body style="font-family:system-ui;padding:24px;max-width:760px;margin:auto">
<h1>Seedream v4.5 — Face Swap</h1>
<form method="POST" action="/generate-batch">
<label>Face Image URL (source face - img1)</label><br/>
<input name="face_url" type="url" required style="width:100%"/><br/><br/>
<label>Target Pose Image URL (body/pose/scene - img2)</label><br/>
<input name="target_url" type="url" required style="width:100%"/><br/><br/>
<label>Width</label><input name="width" type="number" value="2572" required/>
<label style="margin-left:12px">Height</label><input name="height" type="number" value="3576" required/>
<label style="margin-left:12px">Batch</label><input name="batch_count" type="number" value="4" min="1" max="24" required/><br/><br/>
<button type="submit">Start Batch</button>
<p><small>Model: ${MODEL_NAME}</small></p>
<p><small>Webhook: ${PUBLIC_BASE_URL}/webhooks/wavespeed</small></p>
</form></body></html>`);
});

/* ===================== Core Batch ===================== */
async function startRunFromRecord(recordId, opts = {}) {
  const rec = await atGet(recordId);
  const f = rec.fields || {};

  // Subject = FACE image (what we want to transfer) - img1
  // References = TARGET/POSE images (scenes to recreate) - img2
  const faceUrl = f["Subject"]?.[0]?.url || f["Subject URL"] || "";
  const poseImages = Array.isArray(f["References"]) ? f["References"].map(x => x.url) : [];
  const refCsv = splitCSV(f["Reference URLs"] || "");
  const allTargetUrls = [...poseImages, ...refCsv].filter(Boolean);

  if (!faceUrl) throw new Error("Record needs Subject (face image)");
  if (allTargetUrls.length === 0) throw new Error("Record needs References (target pose/body images)");

  // Size - default to high resolution
  let W = 2572, H = 3576;
  const sizeStr = String(f["Size"] || "");
  const m = sizeStr.match(/(\d+)\s*[xX*]\s*(\d+)/);
  if (m) { W = +m[1]; H = +m[2]; }

  // Convert face image to data URL once (reused for all targets)
  const faceDataUrl = await urlToDataURL(faceUrl);

  // Status reset
  await atPatch(recordId, { 
    Status: "processing", 
    "Request IDs": "", 
    "Seen IDs": "", 
    "Failed IDs": "", 
    "Last Update": nowISO(), 
    Model: MODEL_NAME, 
    Size: `${W}x${H}` 
  });

  // Batch-Count per target image
  const batchPerTarget = Math.max(1, Math.min(24, Number(f["Batch Count"] || opts.batch || 4)));

  const requestIds = [];
  
  // For each target image, generate batch_count variations
  for (const targetUrl of allTargetUrls) {
    let targetDataUrl;
    try {
      targetDataUrl = await urlToDataURL(targetUrl);
    } catch (e) {
      console.warn(`[TARGET WARN] Failed to fetch ${targetUrl}: ${e.message}`);
      continue;
    }

    // Generate multiple outputs for this target
    for (let i = 0; i < batchPerTarget; i++) {
      let rid = null, lastErr = null;
      for (let a = 0; a < SUBMIT_MAX_RETRIES && !rid; a++) {
        try {
          rid = await submitGeneration({ 
            faceDataUrl, 
            targetDataUrl, 
            width: W, 
            height: H 
          }, recordId);
        } catch (err) {
          lastErr = err; 
          await sleep(backoff(a, SUBMIT_BASE_DELAY_MS));
        }
      }
      if (!rid) { 
        console.error("[SUBMIT FAIL]", lastErr?.message || lastErr); 
        continue; 
      }
      requestIds.push(rid);
      if (requestIds.length < batchPerTarget * allTargetUrls.length) {
        await sleep(JOB_SPACING_MS);
      }
    }
  }

  await atPatch(recordId, { "Request IDs": requestIds.join(", "), "Last Update": nowISO() });

  // Poll all requests
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
      <h2>✅ Run started</h2>
      <p><strong>Model:</strong> ${MODEL_NAME}</p>
      <p><strong>Record:</strong> ${out.recordId}</p>
      <p><strong>Submitted:</strong> ${out.submitted} jobs</p>
      <p><strong>Request IDs:</strong> ${out.request_ids.join(", ")}</p>
      <p>You can close this tab. Results will appear in Airtable.</p>
    </body></html>`);
  } catch (e) {
    res.status(500).send(`<html><body style="font-family:system-ui;padding:24px">
      <h2>❌ Error</h2>
      <p>${String(e.message || e)}</p>
    </body></html>`);
  }
});

/* Batch form endpoint */
app.post("/generate-batch", async (req, res) => {
  try {
    const { face_url, target_url, width, height, batch_count } = req.body;
    if (!face_url || !target_url || !width || !height || !batch_count) {
      return res.status(400).json({ error: "Missing fields. Need: face_url, target_url, width, height, batch_count" });
    }

    let { w: W, h: H } = ensureMinPixels(Number(width), Number(height));

    const faceDataUrl = await urlToDataURL(face_url);
    const targetDataUrl = await urlToDataURL(target_url);

    const runId = uuid();
    const parentId = await atCreate({
      Prompt: "Face swap using Seedream v4.5",
      Subject: [{ url: face_url }],
      References: [{ url: target_url, filename: "target_1" }],
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

    const N = Math.max(1, Math.min(24, Number(batch_count)));
    const requestIds = [];
    
    for (let i = 0; i < N; i++) {
      let rid = null, lastErr = null;
      for (let a = 0; a < SUBMIT_MAX_RETRIES && !rid; a++) {
        try {
          rid = await submitGeneration({ 
            faceDataUrl, 
            targetDataUrl, 
            width: W, 
            height: H 
          }, parentId);
        } catch (err) {
          lastErr = err; 
          await sleep(backoff(a, SUBMIT_BASE_DELAY_MS));
        }
      }
      if (!rid) console.error(`[SUBMIT FAIL] job ${i + 1}:`, lastErr?.message || lastErr);
      else requestIds.push(rid);
      if (i < N - 1) await sleep(JOB_SPACING_MS);
    }

    await atPatch(parentId, { "Request IDs": requestIds.join(", "), "Last Update": nowISO() });
    requestIds.forEach(rid => pollUntilDone(rid, parentId).catch(e => console.error("[POLL ERROR]", rid, e.message)));

    res.json({ 
      ok: true, 
      model: MODEL_NAME,
      parent_record_id: parentId, 
      run_id: runId, 
      submitted: requestIds.length, 
      request_ids: requestIds 
    });
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

    console.log(`[WEBHOOK] Received for ${requestId}: status=${status}, outputs=${outputs.length}`);

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

/* Debug endpoint */
app.get("/debug/prediction/:id", async (req, res) => {
  try {
    const r = await fetch(`${WAVESPEED_BASE}${WAVESPEED_RESULT_PATH}/${encodeURIComponent(req.params.id)}/result`, { headers: { ...authHeader() } });
    const text = await r.text();
    res.status(r.status).type("application/json").send(text);
  } catch (e) {
    res.status(500).json({ error: String(e.message || e) });
  }
});

/* Health check */
app.get("/health", (_, res) => {
  res.json({ 
    ok: true, 
    model: MODEL_NAME,
    endpoint: `${WAVESPEED_BASE}${WAVESPEED_SUBMIT_PATH}`,
    timestamp: nowISO()
  });
});

/* START */
app.listen(PORT, () => {
  console.log(`[BOOT] Server running on http://localhost:${PORT}`);
  console.log(`[BOOT] Model: ${MODEL_NAME}`);
  console.log(`[BOOT] Public base URL: ${PUBLIC_BASE_URL}`);
  console.log(`[BOOT] Webhook listening at: ${PUBLIC_BASE_URL}/webhooks/wavespeed`);
});
