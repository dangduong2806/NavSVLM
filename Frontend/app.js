"use strict";

const $ = (id) => document.getElementById(id);
const PREVIEW_REPEATS = 9;
let frames = [];
let scenes = [];
let selectedScene = 0;
let latestStatus = null;
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
  $("scene-image").alt = `${scenes[selectedScene].title}, frame ${index + 1} of ${frames.length}`;
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
  $("sequence-note").textContent = `${frames.length} frames · ${PREVIEW_REPEATS} repeats`;
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
  $("sequence-note").textContent = `${frames.length} frames · ${PREVIEW_REPEATS} repeats`;
  $("play-button").disabled = frames.length < 2;
  if (frames.length) selectFrame(frames.length - 1);
  else {
    $("scene-image").hidden = true;
    $("scene-image").removeAttribute("src");
    $("image-placeholder").hidden = false;
    $("image-placeholder").textContent = "No JPG frames found for this scene.";
    $("frame-counter").textContent = "00 / 00";
    $("frame-caption").textContent = "No frames available";
  }
}

function selectScene(index) {
  stopPlayback();
  stopSpeech();
  selectedScene = index;
  const scene = scenes[index];
  frames = scene.frames;
  $("scene-count").textContent = `${index + 1} / ${scenes.length}`;
  $("scene-label").textContent = scene.title.toUpperCase();
  $("scene-folder").textContent = scene.folder.replaceAll("/", " / ");
  runStatus = "idle";
  answer = "";
  $("speech-message").hidden = true;
  setOutput("Understand the scene.", "Generate guidance to hear about obstacles, a safe direction, and the next action to take.");
  renderFrames();
  if (latestStatus) showStatus(latestStatus);
  syncButtons();
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
  const busy = latestStatus?.status === "running" || submitting;
  $("generate-button").disabled = !connected || !frames.length || busy;
  $("previous-scene").disabled = !connected || busy || selectedScene <= 0;
  $("next-scene").disabled = !connected || busy || selectedScene >= scenes.length - 1;
  $("previous-scene").title = $("next-scene").title = busy ? "Wait for generation to finish before switching scenes." : "";
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
  latestStatus = data;
  // A result belongs only to the sequence that produced it.
  if (data.scene_id !== scenes[selectedScene]?.id) {
    data = { status: "idle", answer: "", error: "", logs: [], elapsed: 0 };
  }
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
      const selectedId = scenes[selectedScene]?.id;
      const firstConnection = scenes.length === 0;
      scenes = session.scenes;
      latestStatus = await request("/api/status");
      const desiredId = firstConnection ? latestStatus.scene_id : selectedId;
      selectedScene = Math.max(0, scenes.findIndex((scene) => scene.id === desiredId));
      connected = true;
      selectScene(selectedScene);
    } else {
      const data = await request("/api/status");
      showStatus(data);
    }
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
  if (frames.length < 2) return;
  let completedLoops = 0;
  selectFrame(0);
  $("sequence-note").textContent = `${frames.length} frames · Repeat 1 / ${PREVIEW_REPEATS}`;
  $("play-button").textContent = "Ⅱ";
  $("play-button").setAttribute("aria-label", "Pause frame sequence");
  playback = window.setInterval(() => {
    if (selectedFrame === frames.length - 1) {
      completedLoops += 1;
      if (completedLoops === PREVIEW_REPEATS) return stopPlayback();
      selectFrame(0);
      $("sequence-note").textContent = `${frames.length} frames · Repeat ${completedLoops + 1} / ${PREVIEW_REPEATS}`;
    } else {
      selectFrame(selectedFrame + 1);
    }
  }, 450);
});

$("previous-scene").addEventListener("click", () => selectScene(selectedScene - 1));
$("next-scene").addEventListener("click", () => selectScene(selectedScene + 1));

$("generate-button").addEventListener("click", async () => {
  submitting = true;
  syncButtons();
  stopSpeech();
  stopPlayback();
  selectFrame(frames.length - 1);
  try {
    showStatus(await request(`/api/run?scene=${encodeURIComponent(scenes[selectedScene].id)}`, { method: "POST" }));
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
