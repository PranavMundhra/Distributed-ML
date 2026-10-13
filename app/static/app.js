"use strict";

const form = document.getElementById("inference-form");
const inputs = Array.from(form.querySelectorAll("input"));
const runButton = document.getElementById("run-button");
const feedback = document.getElementById("feedback");
const resultPanel = document.getElementById("result-panel");
const presets = {
    setosa: [5.1, 3.5, 1.4, 0.2],
    versicolor: [6.0, 2.9, 4.5, 1.5],
    virginica: [6.5, 3.0, 5.8, 2.2],
};
let checkingHealth = false;

async function requestJSON(path, options = {}) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 30000);
    try {
        const response = await fetch(path, { ...options, signal: controller.signal, cache: "no-store" });
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
                422: "Enter four finite measurements greater than zero and at most 100 centimeters.",
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
        setServiceStatus("worker", health.worker_available === true, health.worker_available ? "Available" : "Unavailable");
    } catch {
        setServiceStatus("gateway", false, "Unreachable");
        setServiceStatus("worker", false, "Unknown");
    } finally {
        checkingHealth = false;
        refresh.disabled = false;
    }
}

function clearResult(message = "") {
    document.getElementById("result").hidden = true;
    document.getElementById("empty-result").hidden = false;
    feedback.className = "feedback";
    feedback.textContent = message;
}

function showResult(data) {
    const species = ["setosa", "versicolor", "virginica"];
    if (!species.includes(data.prediction) || !data.probabilities ||
        species.some((name) => !Number.isFinite(data.probabilities[name]) || data.probabilities[name] < 0 || data.probabilities[name] > 1)) {
        throw new Error("The service returned an unexpected prediction. Check the gateway terminal.");
    }
    document.getElementById("prediction").textContent = data.prediction;
    document.getElementById("confidence").textContent = `${(data.probabilities[data.prediction] * 100).toFixed(1)}%`;
    const probabilities = document.getElementById("probabilities");
    probabilities.replaceChildren();
    for (const name of species) {
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
    feedback.textContent = `Inference complete: ${data.prediction}, ${(data.probabilities[data.prediction] * 100).toFixed(1)}% model confidence.`;
}

inputs.forEach((input) => {
    input.addEventListener("input", () => {
        input.setCustomValidity("");
        clearResult();
    });
});

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
    const features = inputs.map((input) => input.valueAsNumber);
    const invalidIndex = features.findIndex((value) => !Number.isFinite(value) || value <= 0 || value > 100);
    if (invalidIndex !== -1) {
        inputs[invalidIndex].setCustomValidity("Enter a finite measurement greater than zero and at most 100 centimeters.");
        inputs[invalidIndex].reportValidity();
        inputs[invalidIndex].focus();
        return;
    }
    clearResult("Running inference…");
    runButton.disabled = true;
    runButton.textContent = "Running inference…";
    resultPanel.setAttribute("aria-busy", "true");
    // Keep the submitted measurements visible until this request completes.
    inputs.forEach((input) => { input.disabled = true; });
    document.querySelectorAll("[data-preset]").forEach((button) => { button.disabled = true; });
    try {
        showResult(await requestJSON("/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ features }),
        }));
    } catch (error) {
        feedback.className = "feedback error";
        feedback.textContent = error.name === "AbortError"
            ? "The request timed out. Check both service terminals and try again."
            : error instanceof TypeError
                ? "Cannot reach the gateway. Check that it is running, then try again."
                : error.message;
    } finally {
        runButton.disabled = false;
        runButton.textContent = "Run inference ↗";
        resultPanel.setAttribute("aria-busy", "false");
        inputs.forEach((input) => { input.disabled = false; });
        document.querySelectorAll("[data-preset]").forEach((button) => { button.disabled = false; });
        checkHealth();
    }
});

document.getElementById("refresh-status").addEventListener("click", checkHealth);
checkHealth();
setInterval(() => { if (!document.hidden) checkHealth(); }, 15000);
