"use strict";

const cards = new Map();
const feedback = document.getElementById("demo-feedback");
const batchButton = document.getElementById("send-batch");
const titles = { "iris-logistic-regression-v1": "Iris", "house-price-linear-v1": "House price", "male-female-cydral-v1": "Face image" };
let snapshot = null;
let refreshing = false;
let busy = false;
let historySignature = "";

async function jsonRequest(path, options = {}) {
    const response = await fetch(path, { ...options, cache: "no-store", signal: AbortSignal.timeout(10000),
        headers: { Accept: "application/json", ...(options.headers || {}) } });
    const data = await response.json();
    if (!response.ok) throw new Error(typeof data.detail === "string" ? data.detail : `Request failed (HTTP ${response.status}).`);
    return data;
}

function textElement(tag, text, className = "") {
    const element = document.createElement(tag);
    element.textContent = text;
    if (className) element.className = className;
    return element;
}

function formatTime(value) {
    const date = value ? new Date(value) : null;
    return date && Number.isFinite(date.getTime()) ? date.toLocaleTimeString() : "Not received yet";
}

function makeCard(worker) {
    const element = document.createElement("article");
    element.className = "panel node-card";
    const heading = textElement("div", "", "section-heading");
    heading.append(textElement("h3", worker.worker_id));
    const status = textElement("span", "", "node-status");
    heading.append(status);
    const address = textElement("a", worker.worker_url, "node-address");
    address.href = `${worker.worker_url.replace(/\/$/, "")}/health`;
    const detail = textElement("p", "", "node-detail");
    const models = textElement("p", "", "model-badges");
    const metadata = document.createElement("dl");
    metadata.className = "node-metadata";
    const fields = {};
    for (const [name, label] of [["sequence", "Heartbeat number"], ["heartbeat", "Last heartbeat"],
        ["deadline", "Heartbeat deadline"], ["count", "Completed predictions"], ["instance", "Process identity"]]) {
        const row = document.createElement("div");
        const value = document.createElement("dd");
        fields[name] = value;
        row.append(textElement("dt", label), value);
        metadata.append(row);
    }
    const button = textElement("button", "", "node-control");
    button.type = "button";
    button.addEventListener("click", () => changeWorker(worker.worker_id));
    element.append(heading, address, detail, models, metadata, button);
    document.getElementById("node-cards").append(element);
    return { element, status, detail, models, fields, button };
}

function updateButtons() {
    const nodes = snapshot?.nodes || [];
    const waiting = nodes.some(({ worker }) => worker.demo_paused && worker.available !== false);
    batchButton.disabled = busy || !snapshot || !nodes.some(({ worker }) => worker.available !== false) || waiting;
    batchButton.textContent = busy ? "Working…" : waiting ? "Waiting for failure detection…" : "Send 6 Iris requests ↗";
    for (const { worker } of nodes) {
        const card = cards.get(worker.worker_id);
        const enabled = snapshot.demo_controls_enabled && worker.demo_control_available && worker.instance_id;
        card.button.disabled = busy || !enabled;
        card.button.textContent = worker.demo_paused ? `Resume ${worker.worker_id}` : `Pause ${worker.worker_id} (demo)`;
        card.button.title = enabled ? "Control this worker's inference and heartbeat service" : "Use the local launcher to enable demonstration controls";
    }
}

function render() {
    const nodes = snapshot.nodes;
    const liveIDs = new Set(nodes.map(({ worker }) => worker.worker_id));
    for (const [id, card] of cards) if (!liveIDs.has(id)) { card.element.remove(); cards.delete(id); }
    document.getElementById("nodes-empty").hidden = nodes.length > 0;
    document.getElementById("nodes-empty").textContent = "No workers registered. Start the workers after the gateway.";
    const eligible = nodes.filter(({ worker }) => worker.available !== false).length;
    document.getElementById("node-count").textContent = `${eligible}/${nodes.length} eligible`;
    document.getElementById("demo-mode").textContent = snapshot.demo_controls_enabled ? "Local demo controls enabled" : "Observation only";
    for (const node of nodes) {
        const worker = node.worker;
        if (!cards.has(worker.worker_id)) cards.set(worker.worker_id, makeCard(worker));
        const card = cards.get(worker.worker_id);
        const waiting = worker.demo_paused && worker.available !== false;
        const unavailable = worker.available === false;
        card.status.textContent = waiting ? "Waiting for detection" : unavailable ? "Unavailable" : worker.available === null ? "Registered" : "Available";
        card.status.className = `node-status ${waiting ? "waiting" : unavailable ? "offline" : "online"}`;
        card.detail.textContent = waiting ? "Demo paused: heartbeats stopped. Waiting for the gateway's normal timeout."
            : worker.heartbeat_expired ? "Heartbeat expired. New requests skip this worker."
                : unavailable ? "A request or health check found this worker unavailable."
                    : worker.available === null ? "Registered; waiting for a heartbeat or successful request."
                        : "Eligible for compatible inference requests.";
        card.models.replaceChildren(...worker.models.map((id) => textElement("span", titles[id] || id, "model-badge")));
        card.fields.sequence.textContent = String(worker.heartbeat_sequence);
        card.fields.heartbeat.textContent = formatTime(worker.last_heartbeat_at);
        card.fields.heartbeat.title = worker.last_heartbeat_at || "No heartbeat received";
        card.fields.deadline.textContent = worker.heartbeat_expired ? "Expired" : `${Math.ceil(node.lease_remaining_seconds)}s remaining (limit ${worker.heartbeat_timeout_seconds}s)`;
        card.fields.count.textContent = String(worker.completed_predictions);
        card.fields.instance.textContent = worker.instance_id ? worker.instance_id.slice(0, 8) : "Legacy registration";
        card.fields.instance.title = worker.instance_id || "";
    }
    const signature = snapshot.recent_predictions.map((item) => item.trace_id).join(",");
    if (signature !== historySignature) {
        const rows = snapshot.recent_predictions.map((item) => {
            const row = document.createElement("tr");
            row.append(textElement("td", formatTime(item.completed_at)), textElement("td", titles[item.model] || item.model), textElement("td", item.worker_id));
            const cell = document.createElement("td");
            const link = textElement("a", item.trace_id.slice(0, 8));
            link.href = `/traces/${encodeURIComponent(item.trace_id)}`;
            link.title = `Request ${item.request_id}; process ${item.instance_id || "legacy"}`;
            cell.append(link);
            row.append(cell);
            return row;
        });
        document.getElementById("request-history").replaceChildren(...rows);
        historySignature = signature;
    }
    document.getElementById("history-empty").hidden = snapshot.recent_predictions.length > 0;
    updateButtons();
}

async function refreshNodes() {
    if (refreshing) return;
    refreshing = true;
    try {
        snapshot = await jsonRequest("/workers/status");
        if (!Array.isArray(snapshot.nodes) || !Array.isArray(snapshot.recent_predictions)) throw new Error("Unexpected worker status response");
        document.getElementById("cluster-status").textContent = "Connected";
        document.getElementById("cluster-dot").className = "dot available";
        render();
    } catch {
        snapshot = null;
        document.getElementById("cluster-status").textContent = "Unreachable — displayed data may be old";
        document.getElementById("cluster-dot").className = "dot unavailable";
        batchButton.disabled = true;
        for (const card of cards.values()) card.button.disabled = true;
    } finally {
        refreshing = false;
    }
}

async function changeWorker(workerID) {
    const worker = snapshot?.nodes.find((node) => node.worker.worker_id === workerID)?.worker;
    if (!worker || busy) return;
    const action = worker.demo_paused ? "resume" : "pause";
    busy = true;
    updateButtons();
    feedback.className = "feedback";
    feedback.textContent = `${action === "pause" ? "Pausing" : "Resuming"} ${workerID}…`;
    try {
        await jsonRequest(`/workers/${encodeURIComponent(workerID)}/demo`, { method: "POST",
            headers: { "Content-Type": "application/json" }, body: JSON.stringify({ action, instance_id: worker.instance_id }) });
        feedback.textContent = action === "pause"
            ? `${workerID} paused. Wait for its heartbeat deadline to expire, then send requests to the surviving worker. A failed health/prediction request can detect it earlier.`
            : `${workerID} resumed. Its next heartbeat will restore scheduling; the process identity stays the same.`;
    } catch (error) {
        feedback.className = "feedback error";
        feedback.textContent = error.message;
    } finally {
        busy = false;
        await refreshNodes();
        updateButtons();
    }
}

batchButton.addEventListener("click", async () => {
    if (busy) return;
    busy = true;
    updateButtons();
    feedback.className = "feedback";
    feedback.textContent = "Sending six real Iris inference requests…";
    document.getElementById("batch-summary").textContent = "";
    const list = document.getElementById("batch-results");
    list.replaceChildren();
    const counts = new Map();
    let succeeded = 0;
    for (let index = 1; index <= 6; index++) {
        try {
            const result = await jsonRequest("/predict", { method: "POST", headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ features: [5.1, 3.5, 1.4, .2] }) });
            list.append(textElement("li", `${index}. ${result.worker_id} → ${result.prediction}`));
            counts.set(result.worker_id, (counts.get(result.worker_id) || 0) + 1);
            succeeded++;
        } catch (error) {
            list.append(textElement("li", `${index}. Failed: ${error.message}`, "batch-failure"));
        }
    }
    feedback.textContent = `${succeeded}/6 requests completed. These are new requests; failed attempts were not retried.`;
    if (succeeded < 6) feedback.className = "feedback error";
    document.getElementById("batch-summary").textContent = Array.from(counts, ([id, count]) => `${id}: ${count}`).join(" · ");
    busy = false;
    await refreshNodes();
    updateButtons();
});

document.getElementById("refresh-nodes").addEventListener("click", refreshNodes);
refreshNodes();
setInterval(() => { if (!document.hidden) refreshNodes(); }, 1000);
