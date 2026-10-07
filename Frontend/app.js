"use strict";

const $ = (id) => document.getElementById(id);
let frames = [];
let selectedFrame = 0;
let playback = null;
let answer = "";
let runStatus = "idle";
let connected = false;
let submitting = false;
let speaking = false;
let utterance = null;

async function request(path, options) {
  const response = await fetch(path, { cache: "no-store", ...options });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || `Request failed (${response.status}).`);
  return data;
}

function selectFrame(index) {
  selectedFrame = index;
  $("scene-image").src = frames[index].url;
  $("scene-image").alt = `Pedestrian walkway sample, frame ${index + 1} of ${frames.length}`;
  $("scene-image").hidden = false;
  $("image-placeholder").hidden = true;
  $("frame-counter").textContent = `${String(index + 1).padStart(2, "0")} / ${String(frames.length).padStart(2, "0")}`;
  $("frame-caption").textContent = index === frames.length - 1 ? "Final frame · vision input" : `Frame ${index + 1} · tracking input`;
  [...$("frame-strip").children].forEach((button, i) => button.setAttribute("aria-pressed", String(i === index)));
}

function stopPlayback() {
  clearInterval(playback);
  playback = null;
  $("play-button").textContent = "▶";
  $("play-button").setAttribute("aria-label", "Play frame sequence");
}

function renderFrames() {
  $("frame-strip").replaceChildren();
  frames.forEach((frame, index) => {
    const button = document.createElement("button");
    button.className = "frame-button";
    button.setAttribute("aria-label", `View frame ${index + 1}${index === frames.length - 1 ? ", final vision input" : ""}`);
    const img = document.createElement("img");
    img.src = frame.url;
    img.alt = "";
    const label = document.createElement("span");
    label.textContent = String(index + 1).padStart(2, "0");
    button.append(img, label);
    button.addEventListener("click", () => { stopPlayback(); selectFrame(index); });
    $("frame-strip").append(button);
  });
  $("sequence-note").textContent = `${frames.length} frames · chronological order`;
  $("play-button").disabled = frames.length < 2;
  if (frames.length) selectFrame(frames.length - 1);
  else $("image-placeholder").textContent = "No JPG sample frames found in wad_sample/images.";
}

function setOutput(title, text, isAnswer = false) {
  const paragraph = document.createElement("p");
  paragraph.textContent = text;
  paragraph.className = isAnswer ? "answer" : "";
  if (isAnswer) $("guidance-output").replaceChildren(paragraph);
  else {
    const heading = document.createElement("h3");
    heading.textContent = title;
    $("guidance-output").replaceChildren(heading, paragraph);
  }
}

function syncButtons() {
  $("generate-button").disabled = !connected || !frames.length || runStatus === "running" || submitting;
  $("speak-button").disabled = !answer || !("speechSynthesis" in window);
  $("copy-button").disabled = !answer;
}

function stopSpeech() {
  if (utterance) {
    utterance.onend = null;
    utterance.onerror = null;
  }
  if ("speechSynthesis" in window) window.speechSynthesis.cancel();
  speaking = false;
  utterance = null;
  $("speak-button").textContent = "◖)) Read aloud";
}

function readGuidance() {
  if (!answer) return;
  stopSpeech();
  $("speech-message").hidden = true;
  const unavailable = () => {
    stopSpeech();
    $("speech-message").textContent = "Automatic playback could not start. Select Read aloud to hear the guidance, or use Copy text.";
    $("speech-message").hidden = false;
  };
  if (!("speechSynthesis" in window)) {
    $("speech-message").textContent = "This browser does not support read-aloud. You can read or copy the guidance text.";
    $("speech-message").hidden = false;
    return;
  }
  try {
    utterance = new SpeechSynthesisUtterance(answer);
    utterance.lang = "en-US";
    utterance.rate = 0.95;
    utterance.onend = stopSpeech;
    utterance.onerror = (event) => {
      if (event.error === "canceled" || event.error === "interrupted") stopSpeech();
      else unavailable();
    };
    speaking = true;
    $("speak-button").textContent = "■ Stop reading";
    window.speechSynthesis.speak(utterance);
  } catch {
    unavailable();
  }
}

function showStatus(data) {
  // Speak each completed run once, including repeat runs with identical text.
  // A page refresh showing an old result should not replay it.
  const justCompleted = runStatus === "running" && data.status === "complete";
  const changed = runStatus !== data.status || answer !== data.answer;
  runStatus = data.status;
  answer = data.answer;
  $("status-badge").dataset.state = data.status;
  $("status-label").textContent = { idle: "Ready to generate", running: "Processing scene", complete: "Guidance ready", error: "Run failed" }[data.status];
  $("elapsed").textContent = data.status === "idle" ? "" : `${Math.floor(data.elapsed / 60)}m ${Math.floor(data.elapsed % 60)}s`;
  $("generate-label").textContent = data.status === "running" ? "Generating guidance…" : data.status === "idle" ? "Generate guidance" : "Generate again";
  $("run-hint").textContent = data.status === "running" ? "Loading models and processing. Guidance will be read aloud when ready." : "Guidance is read aloud automatically when ready.";
  $("error-message").hidden = !data.error;
  $("error-message").textContent = data.error;
  $("run-log").textContent = data.logs.length ? data.logs.join("\n") : data.status === "running" ? "Starting main.py…" : "No run started.";
  $("log-summary").textContent = data.status === "running" ? "Running · open for details" : data.status === "error" ? "Open to inspect the error" : data.status === "complete" ? "Run completed" : "Pipeline output will appear here";
  if (changed) {
    stopSpeech();
    $("speech-message").hidden = true;
    if (data.status === "complete") setOutput("", answer, true);
    else if (data.status === "running") setOutput("Reading the scene…", "The model is processing visual context and object motion to generate navigation guidance.");
    else if (data.status === "error") setOutput("Couldn’t generate guidance.", "Check the run log for details. After resolving the issue, you can try again.");
    else setOutput("Understand the scene.", "Generate guidance to hear about obstacles, a safe direction, and the next action to take.");
  }
  syncButtons();
  if (justCompleted) readGuidance();
}

async function poll() {
  try {
    // Reload session data on reconnect, including after the launcher restarts.
    if (!connected) {
      const session = await request("/api/session");
      frames = session.frames;
      renderFrames();
    }
    const data = await request("/api/status");
    connected = true;
    showStatus(data);
  } catch (error) {
    connected = false;
    $("status-label").textContent = "Disconnected";
    $("status-badge").dataset.state = "error";
    $("error-message").hidden = false;
    $("error-message").textContent = "Cannot reach the local launcher. Start Frontend/server.py and keep its terminal open. Reconnecting automatically…";
    syncButtons();
  } finally {
    window.setTimeout(poll, 1500);
  }
}

$("play-button").addEventListener("click", () => {
  if (playback) return stopPlayback();
  if (selectedFrame === frames.length - 1) selectFrame(0);
  $("play-button").textContent = "Ⅱ";
  $("play-button").setAttribute("aria-label", "Pause frame sequence");
  playback = window.setInterval(() => {
    if (selectedFrame >= frames.length - 1) return stopPlayback();
    selectFrame(selectedFrame + 1);
    if (selectedFrame === frames.length - 1) stopPlayback();
  }, 450);
});

$("generate-button").addEventListener("click", async () => {
  submitting = true;
  syncButtons();
  stopSpeech();
  stopPlayback();
  selectFrame(frames.length - 1);
  try {
    showStatus(await request("/api/run", { method: "POST" }));
  } catch (error) {
    $("error-message").hidden = false;
    $("error-message").textContent = error.message;
  } finally {
    submitting = false;
    syncButtons();
  }
});

$("speak-button").addEventListener("click", () => {
  if (speaking) return stopSpeech();
  readGuidance();
});

$("copy-button").addEventListener("click", async () => {
  try {
    await navigator.clipboard.writeText(answer);
    $("copy-button").textContent = "Copied!";
    $("announcement").textContent = "Guidance copied to clipboard.";
    window.setTimeout(() => { $("copy-button").textContent = "Copy text"; }, 1800);
  } catch {
    $("announcement").textContent = "Copy unavailable. Select and copy the guidance text manually.";
  }
});

$("scene-image").addEventListener("error", () => {
  $("scene-image").hidden = true;
  $("image-placeholder").hidden = false;
  $("image-placeholder").textContent = "This sample frame could not be loaded.";
});
window.addEventListener("pagehide", () => { stopPlayback(); stopSpeech(); });
if (!("speechSynthesis" in window)) $("speak-button").title = "Speech playback is not supported by this browser.";
poll();
