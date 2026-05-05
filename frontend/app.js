/**
 * ProctorVision — Exam Interface Application
 * ============================================
 * Captures webcam frames and user interaction events,
 * sends to /predict API, and updates the UI in real-time.
 * Enhanced with warning system, eye tracking, and flagged incidents.
 */

// ─── Configuration ─────────────────────────────────────────────────
const API_URL = window.location.origin;
const CAPTURE_INTERVAL_MS = 3000;
const EXAM_DURATION_MINUTES = 30;

// ─── Exam Questions ────────────────────────────────────────────────
const QUESTIONS = [
    { id: 1, text: "Which data structure uses LIFO (Last In, First Out) ordering?", options: ["Queue", "Stack", "Array", "Linked List"], correct: 1 },
    { id: 2, text: "What is the time complexity of binary search on a sorted array?", options: ["O(n)", "O(log n)", "O(n²)", "O(1)"], correct: 1 },
    { id: 3, text: "Which protocol is used for secure web communication?", options: ["FTP", "HTTP", "HTTPS", "SMTP"], correct: 2 },
    { id: 4, text: "In object-oriented programming, what does 'encapsulation' refer to?", options: ["Inheriting properties from a parent class", "Bundling data and methods that operate on it", "Defining multiple methods with the same name", "Converting one data type to another"], correct: 1 },
    { id: 5, text: "What is the primary function of an operating system's kernel?", options: ["Managing user interfaces", "Compiling source code", "Managing hardware resources and system calls", "Rendering web pages"], correct: 2 },
];

// ─── State ─────────────────────────────────────────────────────────
let currentQuestion = 0;
let answers = new Array(QUESTIONS.length).fill(null);
let behaviorStats = { clicks: 0, keystrokes: 0, tabSwitches: 0, answerChanges: 0, idleTime: 0, lastActivity: Date.now() };
let webcamStream = null;
let captureTimer = null;
let examTimer = null;
let idleTimer = null;
let timeRemaining = EXAM_DURATION_MINUTES * 60;
let totalWarnings = 0;
let warningBannerTimeout = null;
let criticalModalTimeout = null;
let criticalCountdownInterval = null;
let audioCtx = null;

// ─── DOM References ────────────────────────────────────────────────
const video = document.getElementById("webcam-video");
const canvas = document.getElementById("webcam-canvas");
const webcamOverlay = document.getElementById("webcam-overlay");

const timerDisplay = document.getElementById("timer-display");
const questionArea = document.getElementById("question-area");
const questionNav = document.getElementById("question-nav");
const questionCounter = document.getElementById("question-counter");
const prevBtn = document.getElementById("prev-btn");
const nextBtn = document.getElementById("next-btn");
const submitBtn = document.getElementById("submit-exam-btn");
const detectionLog = document.getElementById("detection-log");
const statClicks = document.getElementById("stat-clicks");
const statKeystrokes = document.getElementById("stat-keystrokes");
const statTabs = document.getElementById("stat-tabs");
const statChanges = document.getElementById("stat-changes");
const statIdle = document.getElementById("stat-idle");
const statFace = document.getElementById("stat-face");

// Warning elements
const warningBanner = document.getElementById("warning-banner");
const warningBannerIcon = document.getElementById("warning-banner-icon");
const warningBannerTitle = document.getElementById("warning-banner-title");
const warningBannerMessage = document.getElementById("warning-banner-message");
const warningBannerDismiss = document.getElementById("warning-banner-dismiss");
const warningCountBadge = document.getElementById("warning-count-badge");
const warningCountEl = document.getElementById("warning-count");

// Critical modal
const criticalModal = document.getElementById("critical-modal");
const criticalModalTitle = document.getElementById("critical-modal-title");
const criticalModalMessage = document.getElementById("critical-modal-message");
const criticalCountdown = document.getElementById("critical-countdown");
const criticalModalAck = document.getElementById("critical-modal-ack");

// Eye metrics
const eyeIrisLeft = document.getElementById("eye-iris-left");
const eyeIrisRight = document.getElementById("eye-iris-right");
const eyeStatusLabel = document.getElementById("eye-status-label");
const barEyeOpen = document.getElementById("bar-eye-open");
const barGazeVel = document.getElementById("bar-gaze-vel");
const barEar = document.getElementById("bar-ear");
const metricEyeOpen = document.getElementById("metric-eye-open");
const metricGazeDir = document.getElementById("metric-gaze-dir");
const metricGazeVel = document.getElementById("metric-gaze-vel");
const metricEar = document.getElementById("metric-ear");
const gazeCompassDot = document.getElementById("gaze-compass-dot");

// Flags
const flagsList = document.getElementById("flags-list");
const flagsEmpty = document.getElementById("flags-empty");
const flagsCountBadge = document.getElementById("flags-count-badge");

// ─── Initialize ────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
    renderQuestions();
    renderQuestionNav();
    showQuestion(0);
    startExamTimer();
    startIdleTracker();
    setupEventListeners();
});

// ─── Question Rendering ───────────────────────────────────────────
function renderQuestions() {
    questionArea.innerHTML = QUESTIONS.map((q, idx) => `
        <div class="question-card" id="question-${idx}" data-question="${idx}">
            <div class="question-number">Question ${idx + 1} of ${QUESTIONS.length}</div>
            <div class="question-text">${q.text}</div>
            <div class="options-list">
                ${q.options.map((opt, oi) => `
                    <div class="option-item" data-question="${idx}" data-option="${oi}"
                         id="q${idx}-opt${oi}" onclick="selectOption(${idx}, ${oi})">
                        <div class="option-radio"></div>
                        <div class="option-label">${opt}</div>
                    </div>
                `).join("")}
            </div>
        </div>
    `).join("");
}

function renderQuestionNav() {
    questionNav.innerHTML = QUESTIONS.map((_, idx) => `
        <button class="q-nav-btn ${idx === 0 ? 'active' : ''}"
                id="nav-btn-${idx}" onclick="goToQuestion(${idx})">${idx + 1}</button>
    `).join("");
}

function showQuestion(idx) {
    currentQuestion = idx;
    document.querySelectorAll(".question-card").forEach(c => c.classList.remove("active"));
    document.getElementById(`question-${idx}`).classList.add("active");
    document.querySelectorAll(".q-nav-btn").forEach(b => b.classList.remove("active"));
    document.getElementById(`nav-btn-${idx}`).classList.add("active");
    questionCounter.textContent = `${idx + 1} / ${QUESTIONS.length}`;
    prevBtn.disabled = idx === 0;
    nextBtn.disabled = idx === QUESTIONS.length - 1;
}

function goToQuestion(idx) { showQuestion(idx); }

function selectOption(qIdx, optIdx) {
    const prev = answers[qIdx];
    answers[qIdx] = optIdx;
    if (prev !== null && prev !== optIdx) { behaviorStats.answerChanges++; updateStats(); }
    document.querySelectorAll(`[data-question="${qIdx}"]`).forEach(el => {
        if (el.classList.contains("option-item")) el.classList.remove("selected");
    });
    document.getElementById(`q${qIdx}-opt${optIdx}`).classList.add("selected");
    document.getElementById(`nav-btn-${qIdx}`).classList.add("answered");
}

// ─── Navigation ────────────────────────────────────────────────
prevBtn.addEventListener("click", () => { if (currentQuestion > 0) showQuestion(currentQuestion - 1); });
nextBtn.addEventListener("click", () => { if (currentQuestion < QUESTIONS.length - 1) showQuestion(currentQuestion + 1); });

// ─── Timer ─────────────────────────────────────────────────────
function startExamTimer() {
    updateTimerDisplay();
    examTimer = setInterval(() => {
        timeRemaining--;
        updateTimerDisplay();
        if (timeRemaining <= 0) { clearInterval(examTimer); submitExam(); }
    }, 1000);
}

function updateTimerDisplay() {
    const min = Math.floor(timeRemaining / 60);
    const sec = timeRemaining % 60;
    timerDisplay.textContent = `${min.toString().padStart(2, "0")}:${sec.toString().padStart(2, "0")}`;
    if (timeRemaining <= 300) timerDisplay.style.color = "#ef4444";
}

// ─── Idle Tracking ─────────────────────────────────────────────
function startIdleTracker() {
    idleTimer = setInterval(() => {
        const idleSec = Math.floor((Date.now() - behaviorStats.lastActivity) / 1000);
        behaviorStats.idleTime = idleSec;
        statIdle.textContent = `${idleSec}s`;
        if (idleSec > 30 && idleSec % 30 === 0) addLogEntry("Extended idle period detected", "warning");
    }, 1000);
}

// ─── Event Listeners ───────────────────────────────────────────
function setupEventListeners() {
    document.addEventListener("click", () => { behaviorStats.clicks++; behaviorStats.lastActivity = Date.now(); updateStats(); });
    document.addEventListener("keydown", () => { behaviorStats.keystrokes++; behaviorStats.lastActivity = Date.now(); updateStats(); });
    document.addEventListener("visibilitychange", () => {
        if (document.hidden) {
            behaviorStats.tabSwitches++;
            updateStats();
            addLogEntry(`Tab switch detected (count: ${behaviorStats.tabSwitches})`, "danger");
        }
    });
    submitBtn.addEventListener("click", submitExam);
    webcamOverlay.addEventListener("click", startWebcam);
    warningBannerDismiss.addEventListener("click", dismissWarningBanner);
    criticalModalAck.addEventListener("click", dismissCriticalModal);
}

function updateStats() {
    statClicks.textContent = behaviorStats.clicks;
    statKeystrokes.textContent = behaviorStats.keystrokes;
    statTabs.textContent = behaviorStats.tabSwitches;
    statChanges.textContent = behaviorStats.answerChanges;
}

// ─── Webcam ────────────────────────────────────────────────────
async function startWebcam() {
    try {
        webcamStream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480, facingMode: "user" } });
        video.srcObject = webcamStream;
        webcamOverlay.classList.add("hidden");
        addLogEntry("Webcam enabled successfully", "success");
        captureTimer = setInterval(captureAndPredict, CAPTURE_INTERVAL_MS);
    } catch (err) {
        console.error("Webcam error:", err);
        addLogEntry("Webcam access denied or unavailable", "danger");
    }
}

async function captureAndPredict() {
    if (!webcamStream || !video.videoWidth) return;
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(video, 0, 0);
    const dataUrl = canvas.toDataURL("image/jpeg", 0.7);
    const base64Data = dataUrl.split(",")[1];

    try {
        const response = await fetch(`${API_URL}/predict`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ frame: base64Data }),
        });
        if (!response.ok) { console.warn("API error:", await response.json()); return; }
        const result = await response.json();



        // Update face detection status
        let isCritical = false;
        if (result.extracted_features) {
            const facePresent = result.extracted_features.face_present;
            const gazeDir = result.extracted_features.gaze_direction;
            const eyeOpen = result.eye_metrics ? result.eye_metrics.ear_avg > 0.18 : true;

            statFace.textContent = facePresent ? "Yes" : "No";
            statFace.style.color = facePresent ? "var(--risk-normal)" : "var(--risk-high)";
            
            // Immediate visual feedback: turn webcam border red if something is wrong
            if (!facePresent || gazeDir !== "center" || !eyeOpen) {
                document.getElementById("webcam-card").style.borderColor = "var(--risk-high)";
                isCritical = true;
            } else {
                document.getElementById("webcam-card").style.borderColor = "var(--glass-border)";
            }

            if (!facePresent) addLogEntry("Face not detected in frame", "danger");
            if (gazeDir !== "center" && gazeDir !== "unknown") {
                addLogEntry(`Gaze: ${gazeDir}`, "warning");
            }
        }

        // Update eye metrics
        if (result.eye_metrics) updateEyeMetrics(result.eye_metrics, result.extracted_features);

        // Process warnings from behavior analyzer (Banner/Modal)
        if (result.warnings && result.warnings.length > 0) processWarnings(result.warnings);

        // Update warning summary
        if (result.warning_summary) {
            totalWarnings = result.warning_summary.total_warnings;
            updateWarningBadge();
        }



    } catch (err) { console.warn("Prediction request failed:", err.message); }
}

// ─── Eye Metrics ───────────────────────────────────────────────
function updateEyeMetrics(eye, features) {
    // Eye openness bar
    const openPct = Math.round((eye.eye_open_ratio || 0) * 100);
    barEyeOpen.style.width = openPct + "%";
    metricEyeOpen.textContent = openPct + "%";

    // EAR bar
    const earVal = eye.ear_avg || 0;
    const earPct = Math.min(100, Math.round(earVal / 0.35 * 100));
    barEar.style.width = earPct + "%";
    metricEar.textContent = earVal.toFixed(3);
    if (earVal < 0.18) barEar.classList.add("low");
    else barEar.classList.remove("low");

    // Gaze velocity bar
    const vel = eye.gaze_velocity || 0;
    const velPct = Math.min(100, Math.round(vel / 500 * 100));
    barGazeVel.style.width = velPct + "%";
    metricGazeVel.textContent = Math.round(vel);

    // Gaze direction
    const gazeDir = features ? features.gaze_direction || "—" : "—";
    metricGazeDir.textContent = gazeDir;

    // Gaze compass dot
    const irisRatio = eye.iris_ratio_avg || 0.5;
    const compassX = (irisRatio - 0.5) * 20;
    gazeCompassDot.style.transform = `translate(calc(-50% + ${compassX}px), -50%)`;

    // Eye widget iris positions
    const irisOffsetX = (irisRatio - 0.5) * 6;
    eyeIrisLeft.style.transform = `translate(calc(-50% + ${irisOffsetX}px), -50%)`;
    eyeIrisRight.style.transform = `translate(calc(-50% + ${irisOffsetX}px), -50%)`;

    // Eye open/closed state
    if (earVal < 0.18) {
        eyeIrisLeft.classList.add("closed");
        eyeIrisRight.classList.add("closed");
        eyeStatusLabel.textContent = "Closed";
        eyeStatusLabel.style.color = "var(--risk-high)";
    } else {
        eyeIrisLeft.classList.remove("closed");
        eyeIrisRight.classList.remove("closed");
        eyeStatusLabel.textContent = gazeDir === "center" ? "On Screen" : gazeDir;
        eyeStatusLabel.style.color = gazeDir === "center" ? "var(--risk-normal)" : "var(--risk-moderate)";
    }
}

// ─── Warning System ────────────────────────────────────────────
function processWarnings(warnings) {
    warnings.forEach(w => {
        addFlagItem(w);
        if (w.severity === "critical") {
            showCriticalModal(w.message);
            playAlertSound("critical");
        } else if (w.severity === "warning") {
            showWarningBanner(w.message, "warning");
            playAlertSound("warning");
        }
    });
}

function showWarningBanner(message, severity) {
    warningBanner.classList.remove("hidden", "critical");
    warningBanner.classList.add("show");
    if (severity === "critical") warningBanner.classList.add("critical");
    warningBannerIcon.textContent = severity === "critical" ? "🚨" : "⚠";
    warningBannerTitle.textContent = severity === "critical" ? "Critical Alert" : "Warning";
    warningBannerMessage.textContent = message;
    // Animate in
    requestAnimationFrame(() => { warningBanner.classList.add("visible"); });
    // Auto dismiss warnings after 5s
    if (warningBannerTimeout) clearTimeout(warningBannerTimeout);
    if (severity !== "critical") {
        warningBannerTimeout = setTimeout(dismissWarningBanner, 5000);
    }
}

function dismissWarningBanner() {
    warningBanner.classList.remove("visible");
    setTimeout(() => { warningBanner.classList.remove("show"); warningBanner.classList.add("hidden"); }, 400);
    if (warningBannerTimeout) { clearTimeout(warningBannerTimeout); warningBannerTimeout = null; }
}

function showCriticalModal(message) {
    criticalModalMessage.textContent = message;
    criticalModal.classList.remove("hidden");
    let countdown = 10;
    criticalCountdown.textContent = countdown;
    if (criticalCountdownInterval) clearInterval(criticalCountdownInterval);
    criticalCountdownInterval = setInterval(() => {
        countdown--;
        criticalCountdown.textContent = countdown;
        if (countdown <= 0) dismissCriticalModal();
    }, 1000);
}

function dismissCriticalModal() {
    criticalModal.classList.add("hidden");
    if (criticalCountdownInterval) { clearInterval(criticalCountdownInterval); criticalCountdownInterval = null; }
}

function updateWarningBadge() {
    if (totalWarnings > 0) {
        warningCountBadge.classList.remove("hidden");
        warningCountEl.textContent = totalWarnings;
    }
}

// ─── Flagged Incidents ─────────────────────────────────────────
function addFlagItem(warning) {
    if (flagsEmpty) flagsEmpty.style.display = "none";
    const icon = warning.severity === "critical" ? "🚨" : warning.severity === "warning" ? "⚠️" : "ℹ️";
    const item = document.createElement("div");
    item.className = `flag-item ${warning.severity}`;
    item.innerHTML = `
        <span class="flag-icon">${icon}</span>
        <div class="flag-content">
            <div class="flag-type">${warning.event_type.replace(/_/g, " ")}</div>
            <div class="flag-message">${warning.message}</div>
            <div class="flag-time">${warning.time_str || new Date().toLocaleTimeString("en-US", { hour12: false })}</div>
        </div>
    `;
    flagsList.insertBefore(item, flagsList.firstChild);
    // Update count
    const count = flagsList.querySelectorAll(".flag-item").length;
    flagsCountBadge.textContent = count;
    // Limit to 30 items
    while (flagsList.querySelectorAll(".flag-item").length > 30) flagsList.removeChild(flagsList.lastChild);
}

// ─── Audio Alerts ──────────────────────────────────────────────
function playAlertSound(type) {
    try {
        if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        const osc = audioCtx.createOscillator();
        const gain = audioCtx.createGain();
        osc.connect(gain);
        gain.connect(audioCtx.destination);
        gain.gain.value = 0.15;
        if (type === "critical") {
            osc.frequency.value = 880;
            osc.type = "square";
            gain.gain.setValueAtTime(0.15, audioCtx.currentTime);
            gain.gain.exponentialRampToValueAtTime(0.01, audioCtx.currentTime + 0.5);
            osc.start(); osc.stop(audioCtx.currentTime + 0.5);
        } else {
            osc.frequency.value = 440;
            osc.type = "sine";
            gain.gain.setValueAtTime(0.1, audioCtx.currentTime);
            gain.gain.exponentialRampToValueAtTime(0.01, audioCtx.currentTime + 0.3);
            osc.start(); osc.stop(audioCtx.currentTime + 0.3);
        }
    } catch (e) { /* audio not available */ }
}



// ─── Detection Log ─────────────────────────────────────────────
function addLogEntry(message, type = "info") {
    const entry = document.createElement("div");
    entry.className = `log-entry ${type}`;
    const now = new Date();
    entry.textContent = `[${now.toLocaleTimeString("en-US", { hour12: false })}] ${message}`;
    detectionLog.appendChild(entry);
    detectionLog.scrollTop = detectionLog.scrollHeight;
    while (detectionLog.children.length > 50) detectionLog.removeChild(detectionLog.firstChild);
}

// ─── Submit Exam ───────────────────────────────────────────────
function submitExam() {
    clearInterval(examTimer); clearInterval(captureTimer); clearInterval(idleTimer);
    if (webcamStream) webcamStream.getTracks().forEach(t => t.stop());
    let correct = 0;
    QUESTIONS.forEach((q, i) => { if (answers[i] === q.correct) correct++; });
    const answered = answers.filter(a => a !== null).length;
    const elapsed = EXAM_DURATION_MINUTES * 60 - timeRemaining;
    const elapsedMin = Math.floor(elapsed / 60);
    const elapsedSec = elapsed % 60;
    const modalStats = document.getElementById("modal-stats");
    modalStats.innerHTML = `
        <div class="stat-row"><span>Questions Answered</span><strong>${answered} / ${QUESTIONS.length}</strong></div>
        <div class="stat-row"><span>Correct Answers</span><strong>${correct} / ${QUESTIONS.length}</strong></div>
        <div class="stat-row"><span>Time Taken</span><strong>${elapsedMin}m ${elapsedSec}s</strong></div>
        <div class="stat-row"><span>Tab Switches</span><strong>${behaviorStats.tabSwitches}</strong></div>
        <div class="stat-row"><span>Total Warnings</span><strong>${totalWarnings}</strong></div>
        <div class="stat-row"><span>Total Keystrokes</span><strong>${behaviorStats.keystrokes}</strong></div>
    `;
    document.getElementById("modal-overlay").classList.remove("hidden");
    document.getElementById("modal-close-btn").addEventListener("click", () => {
        document.getElementById("modal-overlay").classList.add("hidden");
    });
}
