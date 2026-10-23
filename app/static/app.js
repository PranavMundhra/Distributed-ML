"use strict";

const form = document.getElementById("inference-form");
const inputs = Array.from(document.querySelectorAll("#iris-fields input"));
const houseInputs = Array.from(document.querySelectorAll("#house-fields input"));
const modelSelect = document.getElementById("model-select");
const imageInput = document.getElementById("face-image");
let imageFile = null;
let previewURL = null;
let busy = false;
let modelAvailability = null;
const models = {
    iris: { id: "iris-logistic-regression-v1", path: "/predict", title: "Flower measurements", unit: "centimeters", result: "CPU / Iris",
        description: "Use a sample or enter values above 0 and up to 100 cm.", note: "Sample names describe the source examples; predictions come from the model.",
        empty: "Run inference to see the predicted species and its probability.", label: "PREDICTED SPECIES",
        caveat: "Probabilities express the model's estimate, not a guarantee of accuracy." },
    house: { id: "house-price-linear-v1", path: "/predict/house", title: "House details", unit: "historical Ames data", result: "CPU / Regression",
        description: "Six features from Ames, Iowa sales. The form accepts values within each feature's training range.",
        note: "Historical US prices from 2006–2010, in USD. This model does not estimate current Indian property prices.",
        empty: "Enter house details to see an estimated sale price in USD.", label: "ESTIMATED SALE PRICE",
        caveat: "A six-feature linear regression baseline. Individual feature limits do not ensure a realistic combination; implausible estimates are possible." },
    face: { id: "male-female-cydral-v1", path: "/predict/face", title: "Face image", unit: "single cropped face", result: "CPU / Image classifier",
        description: "Upload one upright, closely cropped face, or try a prepared example.",
        note: "Pretrained Cydral model. No automatic face detection; a scene or non-face image will still receive a label.",
        empty: "Choose a face image to see the model's two appearance-label probabilities.", label: "PREDICTED APPEARANCE LABEL",
        caveat: "Binary appearance labels from a pretrained model; they do not establish gender identity or biological sex. Scores are not calibrated certainty." },
};
const runButton = document.getElementById("run-button");
const feedback = document.getElementById("feedback");
const resultPanel = document.getElementById("result-panel");
const presets = {
    setosa: [5.1, 3.5, 1.4, 0.2],
    versicolor: [6.0, 2.9, 4.5, 1.5],
    virginica: [6.5, 3.0, 5.8, 2.2],
};
let checkingHealth = false;

async function requestJSON(path, options = {}, responseInfo = null) {
    const { timeoutMs = 30000, ...fetchOptions } = options;
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);
    try {
        const response = await fetch(path, { ...fetchOptions, signal: controller.signal, cache: "no-store" });
        if (responseInfo) responseInfo.traceId = response.headers.get("X-Trace-ID");
        let data;
        try {
            data = await response.json();
        } catch {
            throw new Error(`The service returned an unreadable response (HTTP ${response.status}). Check the gateway terminal.`);
        }
        if (!response.ok) {
            let message = data.detail;
            if (Array.isArray(message)) {
                message = message.map((error) => `${error.loc.join(".")}: ${error.msg}`).join("; ");
            }
            const fallback = {
                503: "The worker is unavailable. Start the worker and try again.",
                504: "The worker took too long to respond. Check its terminal and try again.",
                502: "The worker returned an invalid response. Check the worker terminal.",
                413: "Use an image at most 5 MiB.",
                422: "Check the selected model's input values or image format.",
            };
            throw new Error(typeof message === "string" ? message : fallback[response.status] || `Request failed (HTTP ${response.status}).`);
        }
        return data;
    } finally {
        clearTimeout(timer);
    }
}

function setServiceStatus(service, available, label) {
    document.getElementById(`${service}-status`).textContent = label;
    document.getElementById(`${service}-dot`).className = `dot ${available ? "available" : "unavailable"}`;
}

async function checkHealth() {
    if (checkingHealth) return;
    checkingHealth = true;
    const refresh = document.getElementById("refresh-status");
    refresh.disabled = true;
    try {
        const health = await requestJSON("/health");
        setServiceStatus("gateway", health.ready === true, health.ready ? "Online" : "Not ready");
        setServiceStatus("worker", health.worker_available === true,
            `${health.available_workers}/${health.registered_workers} available`);
        modelAvailability = await requestJSON("/models", { timeoutMs: 5000 });
    } catch {
        setServiceStatus("gateway", false, "Unreachable");
        setServiceStatus("worker", false, "Unknown");
        modelAvailability = null;
    } finally {
        checkingHealth = false;
        refresh.disabled = false;
        showAvailability();
    }
}

function showAvailability() {
    const current = Array.isArray(modelAvailability)
        ? modelAvailability.find((item) => item.model === models[modelSelect.value].id) : null;
    document.getElementById("model-availability").textContent = current
        ? current.available_workers > 0
            ? `${current.available_workers} compatible worker(s) available for this model.`
            : "No compatible workers available. Prepare the demo models on the workers and restart them, or check their status."
        : "Model availability unknown. Check the gateway and refresh status.";
}

function updateControls() {
    modelSelect.disabled = busy;
    inputs.forEach((input) => { input.disabled = busy || modelSelect.value !== "iris"; });
    houseInputs.forEach((input) => { input.disabled = busy || modelSelect.value !== "house"; });
    imageInput.disabled = busy || modelSelect.value !== "face";
    imageInput.required = !imageFile;
    document.querySelectorAll("[data-preset], [data-face-sample], #house-sample").forEach((button) => { button.disabled = busy; });
    runButton.disabled = busy;
    runButton.textContent = busy ? "Running inference…" : "Run inference ↗";
    resultPanel.setAttribute("aria-busy", String(busy));
}

function selectModel() {
    const selected = modelSelect.value;
    const config = models[selected];
    for (const name of ["iris", "house", "face"]) document.getElementById(`${name}-fields`).hidden = name !== selected;
    document.getElementById("iris-presets").hidden = selected !== "iris";
    for (const [id, value] of [["input-title", config.title], ["input-unit", config.unit],
        ["input-description", config.description], ["input-note", config.note],
        ["result-unit", config.result], ["empty-description", config.empty],
        ["result-label", config.label], ["result-caveat", config.caveat]]) {
        document.getElementById(id).textContent = value;
    }
    clearResult();
    updateControls();
    showAvailability();
}

function clearResult(message = "") {
    document.getElementById("trace-panel").hidden = true;
    document.getElementById("clock-trace").replaceChildren();
    document.getElementById("trace-identities").textContent = "";
    document.getElementById("result").hidden = true;
    document.getElementById("empty-result").hidden = false;
    feedback.className = "feedback";
    feedback.textContent = message;
}

async function showTrace(traceId) {
    if (!traceId) return;
    const panel = document.getElementById("trace-panel");
    const summary = document.getElementById("trace-summary");
    const list = document.getElementById("clock-trace");
    panel.hidden = false;
    summary.textContent = "Loading request trace…";
    try {
        const trace = await requestJSON(`/traces/${encodeURIComponent(traceId)}`, { timeoutMs: 5000 });
        const labels = {
            request_received: "Gateway received your request",
            request_forwarded: "Gateway sent the request to a worker",
            worker_request_received: "received the request",
            worker_response_sent: "sent a response",
            worker_response_received: "Gateway received the worker response",
            response_sent: "Gateway sent the browser response",
        };
        if (trace.trace_id !== traceId || !Array.isArray(trace.events) || trace.events.length > 6 ||
            trace.events.some((item, index) => !labels[item.event] || typeof item.node_id !== "string" ||
                typeof item.instance_id !== "string" || !Number.isSafeInteger(item.timestamp) || item.timestamp < 1 ||
                (index > 0 && item.timestamp <= trace.events[index - 1].timestamp))) {
            throw new Error("Unexpected logical clock trace");
        }
        list.replaceChildren();
        const processes = new Map();
        trace.events.forEach((item, index) => {
            const row = document.createElement("li");
            const description = document.createElement("span");
            const isWorker = item.event === "worker_request_received" || item.event === "worker_response_sent";
            description.textContent = `${index + 1}. ${isWorker ? `${item.node_id} ` : ""}${labels[item.event]}`;
            const time = document.createElement("span");
            time.className = "logical-time";
            time.textContent = `L = ${item.timestamp}`;
            row.append(description, time);
            list.append(row);
            processes.set(item.instance_id, item.node_id);
        });
        summary.textContent = trace.complete
            ? `Complete message path · HTTP ${trace.status_code}`
            : `Partial trace · HTTP ${trace.status_code}. A failed request or an older worker may have no worker timestamps.`;
        document.getElementById("trace-identities").textContent = Array.from(processes,
            ([instance, node]) => `${node} process ${instance.slice(0, 8)}`).join(" · ");
    } catch {
        summary.textContent = "Trace unavailable. The gateway keeps the last 200 traces until restart. The prediction result is unaffected.";
    }
}

function showResult(data) {
    const selected = modelSelect.value;
    if (data.model !== models[selected].id) throw new Error("The worker returned a different model's result.");
    const species = selected === "face" ? ["female", "male"] : ["setosa", "versicolor", "virginica"];
    if (selected === "house") {
        if (!Number.isFinite(data.predicted_price_usd) || data.currency !== "USD" || data.market !== "Ames, Iowa, 2006-2010") {
            throw new Error("The service returned an invalid house price.");
        }
        document.getElementById("prediction").textContent = new Intl.NumberFormat("en-US", { style: "currency", currency: "USD", maximumFractionDigits: 2 }).format(data.predicted_price_usd);
    } else if (!species.includes(data.prediction) || !data.probabilities ||
        species.some((name) => !Number.isFinite(data.probabilities[name]) || data.probabilities[name] < 0 || data.probabilities[name] > 1)) {
        throw new Error("The service returned an unexpected prediction. Check the gateway terminal.");
    } else {
        document.getElementById("prediction").textContent = data.prediction;
        document.getElementById("confidence").textContent = `${(data.probabilities[data.prediction] * 100).toFixed(1)}%`;
    }
    document.getElementById("confidence-line").hidden = selected === "house";
    const probabilities = document.getElementById("probabilities");
    probabilities.hidden = selected === "house";
    probabilities.setAttribute("aria-label", selected === "face" ? "Appearance label probabilities" : "Species probabilities");
    probabilities.replaceChildren();
    for (const name of selected === "house" ? [] : species) {
        const probability = data.probabilities[name];
        const row = document.createElement("div");
        row.className = "probability";
        const label = document.createElement("div");
        label.className = "probability-label";
        const speciesLabel = document.createElement("span");
        speciesLabel.textContent = name;
        const value = document.createElement("span");
        value.textContent = `${(probability * 100).toFixed(1)}%`;
        label.append(speciesLabel, value);
        const track = document.createElement("div");
        track.className = "track";
        track.setAttribute("aria-hidden", "true");
        const fill = document.createElement("div");
        fill.className = "fill";
        fill.style.width = `${probability * 100}%`;
        track.append(fill);
        row.append(label, track);
        probabilities.append(row);
    }
    for (const [id, key] of [["model-name", "model"], ["worker-id", "worker_id"], ["request-id", "request_id"]]) {
        document.getElementById(id).textContent = data[key];
    }
    document.getElementById("empty-result").hidden = true;
    document.getElementById("result").hidden = false;
    feedback.textContent = selected === "house"
        ? "House price estimate complete. Amounts are historical USD."
        : `Inference complete: ${data.prediction}, ${(data.probabilities[data.prediction] * 100).toFixed(1)}% model confidence.`;
}

[...inputs, ...houseInputs].forEach((input) => {
    input.addEventListener("input", () => {
        input.setCustomValidity("");
        clearResult();
    });
});

document.getElementById("house-sample").addEventListener("click", () => {
    [1800, 3, 2, 2, 2000, 7].forEach((value, index) => {
        houseInputs[index].value = value;
        houseInputs[index].setCustomValidity("");
    });
    clearResult("Example house loaded. Run inference to estimate its historical sale price.");
});

function setImage(file) {
    if (previewURL) URL.revokeObjectURL(previewURL);
    previewURL = null;
    imageFile = null;
    const preview = document.getElementById("face-preview");
    preview.hidden = true;
    preview.removeAttribute("src");
    imageInput.setCustomValidity("");
    clearResult();
    if (file) {
        if (!file.size || file.size > 5 * 1024 * 1024 || !["image/png", "image/jpeg"].includes(file.type)) {
            imageInput.value = "";
            feedback.className = "feedback error";
            feedback.textContent = "Choose a nonempty PNG or JPEG image up to 5 MiB.";
        } else {
            imageFile = file;
            previewURL = URL.createObjectURL(file);
            preview.src = previewURL;
            preview.hidden = false;
        }
    }
    updateControls();
}

imageInput.addEventListener("change", () => setImage(imageInput.files[0]));
document.querySelectorAll("[data-face-sample]").forEach((button) => {
    button.addEventListener("click", async () => {
        busy = true;
        updateControls();
        try {
            const response = await fetch(`/samples/face/${button.dataset.faceSample}`, { signal: AbortSignal.timeout(5000) });
            if (!response.ok) throw new Error("Example image missing. Prepare the demo models first.");
            imageInput.value = "";
            setImage(await response.blob());
            feedback.textContent = "Example face loaded. Run inference to classify it.";
        } catch (error) {
            feedback.className = "feedback error";
            feedback.textContent = error.message;
        } finally {
            busy = false;
            updateControls();
        }
    });
});

function readImage(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(reader.result.split(",")[1]);
        reader.onerror = () => reject(new Error("Cannot read the selected image."));
        reader.readAsDataURL(file);
    });
}

document.querySelectorAll("[data-preset]").forEach((button) => {
    button.addEventListener("click", () => {
        const sample = presets[button.dataset.preset];
        inputs.forEach((input, index) => {
            input.value = sample[index];
            input.setCustomValidity("");
        });
        clearResult(`${button.textContent} sample loaded. Run inference to predict its species.`);
    });
});

form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const selected = modelSelect.value;
    const features = inputs.map((input) => input.valueAsNumber);
    const invalidIndex = selected === "iris" ? features.findIndex((value) => !Number.isFinite(value) || value <= 0 || value > 100) : -1;
    if (invalidIndex !== -1) {
        inputs[invalidIndex].setCustomValidity("Enter a finite measurement greater than zero and at most 100 centimeters.");
        inputs[invalidIndex].reportValidity();
        inputs[invalidIndex].focus();
        return;
    }
    if (!form.reportValidity()) return;
    if (selected === "face" && !imageFile) {
        feedback.textContent = "Choose a face image first.";
        return;
    }
    clearResult("Running inference…");
    busy = true;
    updateControls();
    const responseInfo = {};
    try {
        const payload = selected === "iris" ? { features } : selected === "house"
            ? Object.fromEntries(houseInputs.map((input) => [input.name, input.valueAsNumber]))
            : { image_base64: await readImage(imageFile) };
        showResult(await requestJSON(models[selected].path, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
        }, responseInfo));
    } catch (error) {
        feedback.className = "feedback error";
        feedback.textContent = error.name === "AbortError"
            ? "The request timed out. Check both service terminals and try again."
            : error instanceof TypeError
                ? "Cannot reach the gateway. Check that it is running, then try again."
                : error.message;
    } finally {
        await showTrace(responseInfo.traceId);
        busy = false;
        updateControls();
        checkHealth();
    }
});

document.getElementById("refresh-status").addEventListener("click", checkHealth);
modelSelect.addEventListener("change", selectModel);
selectModel();
checkHealth();
setInterval(() => { if (!document.hidden) checkHealth(); }, 15000);
