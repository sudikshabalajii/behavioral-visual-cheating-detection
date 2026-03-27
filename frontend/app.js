/**
 * ProctorVision — Exam Interface Application
 * ============================================
 * Captures webcam frames and user interaction events,
 * sends to /predict API, and updates the UI in real-time.
 */

// ─── Configuration ─────────────────────────────────────────────────
const API_URL = window.location.origin;
const CAPTURE_INTERVAL_MS = 3000;  // Send frame every 3 seconds
const EXAM_DURATION_MINUTES = 30;

// ─── Exam Questions ────────────────────────────────────────────────
const QUESTIONS = [
    {
        id: 1,
        text: "Which data structure uses LIFO (Last In, First Out) ordering?",
        options: ["Queue", "Stack", "Array", "Linked List"],
        correct: 1,
    },
    {
        id: 2,
        text: "What is the time complexity of binary search on a sorted array?",
        options: ["O(n)", "O(log n)", "O(n²)", "O(1)"],
        correct: 1,
    },
    {
        id: 3,
        text: "Which protocol is used for secure web communication?",
        options: ["FTP", "HTTP", "HTTPS", "SMTP"],
        correct: 2,
    },
    {
        id: 4,
        text: "In object-oriented programming, what does 'encapsulation' refer to?",
        options: [
            "Inheriting properties from a parent class",
            "Bundling data and methods that operate on it",
            "Defining multiple methods with the same name",
            "Converting one data type to another",
        ],
        correct: 1,
    },
    {
        id: 5,
        text: "What is the primary function of an operating system's kernel?",
        options: [
            "Managing user interfaces",
            "Compiling source code",
            "Managing hardware resources and system calls",
            "Rendering web pages",
        ],
        correct: 2,
    },
];

// ─── State ─────────────────────────────────────────────────────────
let currentQuestion = 0;
let answers = new Array(QUESTIONS.length).fill(null);
let behaviorStats = {
    clicks: 0,
    keystrokes: 0,
    tabSwitches: 0,
    answerChanges: 0,
    idleTime: 0,
    lastActivity: Date.now(),
};

let webcamStream = null;
let captureTimer = null;
let examTimer = null;
let idleTimer = null;
let timeRemaining = EXAM_DURATION_MINUTES * 60;

// ─── DOM References ────────────────────────────────────────────────
const video = document.getElementById("webcam-video");
const canvas = document.getElementById("webcam-canvas");
const webcamOverlay = document.getElementById("webcam-overlay");
const gaugeFill = document.getElementById("gauge-fill");
const gaugeValue = document.getElementById("gauge-value");
const gaugeLabel = document.getElementById("gauge-label");
const timerDisplay = document.getElementById("timer-display");
const questionArea = document.getElementById("question-area");
const questionNav = document.getElementById("question-nav");
const questionCounter = document.getElementById("question-counter");
const prevBtn = document.getElementById("prev-btn");
const nextBtn = document.getElementById("next-btn");
const submitBtn = document.getElementById("submit-exam-btn");
const detectionLog = document.getElementById("detection-log");

// Stat elements
const statClicks = document.getElementById("stat-clicks");
const statKeystrokes = document.getElementById("stat-keystrokes");
const statTabs = document.getElementById("stat-tabs");
const statChanges = document.getElementById("stat-changes");
const statIdle = document.getElementById("stat-idle");
const statFace = document.getElementById("stat-face");

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

function goToQuestion(idx) {
    showQuestion(idx);
}

function selectOption(qIdx, optIdx) {
    const prev = answers[qIdx];
    answers[qIdx] = optIdx;

    // Track answer changes
    if (prev !== null && prev !== optIdx) {
        behaviorStats.answerChanges++;
        updateStats();
    }

    // Update UI
    document.querySelectorAll(`[data-question="${qIdx}"]`).forEach(el => {
        if (el.classList.contains("option-item")) {
            el.classList.remove("selected");
        }
    });
    document.getElementById(`q${qIdx}-opt${optIdx}`).classList.add("selected");

    // Mark nav button as answered
    document.getElementById(`nav-btn-${qIdx}`).classList.add("answered");
}

// ─── Navigation ────────────────────────────────────────────────────
prevBtn.addEventListener("click", () => {
    if (currentQuestion > 0) showQuestion(currentQuestion - 1);
});

nextBtn.addEventListener("click", () => {
    if (currentQuestion < QUESTIONS.length - 1) showQuestion(currentQuestion + 1);
});

// ─── Timer ─────────────────────────────────────────────────────────
function startExamTimer() {
    updateTimerDisplay();
    examTimer = setInterval(() => {
        timeRemaining--;
        updateTimerDisplay();
        if (timeRemaining <= 0) {
            clearInterval(examTimer);
            submitExam();
        }
    }, 1000);
}

function updateTimerDisplay() {
    const min = Math.floor(timeRemaining / 60);
    const sec = timeRemaining % 60;
    timerDisplay.textContent = `${min.toString().padStart(2, "0")}:${sec.toString().padStart(2, "0")}`;

    if (timeRemaining <= 300) {
        timerDisplay.style.color = "#ef4444";
    }
}

// ─── Idle Tracking ─────────────────────────────────────────────────
function startIdleTracker() {
    idleTimer = setInterval(() => {
        const idleSec = Math.floor((Date.now() - behaviorStats.lastActivity) / 1000);
        behaviorStats.idleTime = idleSec;
        statIdle.textContent = `${idleSec}s`;

        if (idleSec > 30 && idleSec % 30 === 0) {
            addLogEntry("Extended idle period detected", "warning");
        }
    }, 1000);
}

// ─── Event Listeners ───────────────────────────────────────────────
function setupEventListeners() {
    // Mouse clicks
    document.addEventListener("click", () => {
        behaviorStats.clicks++;
        behaviorStats.lastActivity = Date.now();
        updateStats();
    });

    // Keystrokes
    document.addEventListener("keydown", () => {
        behaviorStats.keystrokes++;
        behaviorStats.lastActivity = Date.now();
        updateStats();
    });

    // Tab switching
    document.addEventListener("visibilitychange", () => {
        if (document.hidden) {
            behaviorStats.tabSwitches++;
            updateStats();
            addLogEntry(`Tab switch detected (count: ${behaviorStats.tabSwitches})`, "danger");
        }
    });

    // Submit exam
    submitBtn.addEventListener("click", submitExam);

    // Webcam overlay click
    webcamOverlay.addEventListener("click", startWebcam);
}

function updateStats() {
    statClicks.textContent = behaviorStats.clicks;
    statKeystrokes.textContent = behaviorStats.keystrokes;
    statTabs.textContent = behaviorStats.tabSwitches;
    statChanges.textContent = behaviorStats.answerChanges;
}

// ─── Webcam ────────────────────────────────────────────────────────
async function startWebcam() {
    try {
        webcamStream = await navigator.mediaDevices.getUserMedia({
            video: { width: 640, height: 480, facingMode: "user" },
        });
        video.srcObject = webcamStream;
        webcamOverlay.classList.add("hidden");
        addLogEntry("Webcam enabled successfully", "success");

        // Start periodic frame capture
        captureTimer = setInterval(captureAndPredict, CAPTURE_INTERVAL_MS);
    } catch (err) {
        console.error("Webcam error:", err);
        addLogEntry("Webcam access denied or unavailable", "danger");
    }
}

async function captureAndPredict() {
    if (!webcamStream || !video.videoWidth) return;

    // Draw frame to canvas
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(video, 0, 0);

    // Convert to base64 JPEG
    const dataUrl = canvas.toDataURL("image/jpeg", 0.7);
    const base64Data = dataUrl.split(",")[1];

    try {
        const response = await fetch(`${API_URL}/predict`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ frame: base64Data }),
        });

        if (!response.ok) {
            const err = await response.json();
            console.warn("API error:", err);
            return;
        }

        const result = await response.json();
        updateRiskGauge(result.risk_score, result.risk_level, result.risk_color);

        // Update face detection status
        if (result.extracted_features) {
            statFace.textContent = result.extracted_features.face_present ? "Yes" : "No";
            statFace.style.color = result.extracted_features.face_present
                ? "var(--risk-normal)" : "var(--risk-high)";

            // Log suspicious detections
            if (!result.extracted_features.face_present) {
                addLogEntry("Face not detected in frame", "danger");
            }
            if (result.extracted_features.gaze_direction !== "center" &&
                result.extracted_features.gaze_direction !== "unknown") {
                addLogEntry(`Gaze: ${result.extracted_features.gaze_direction}`, "warning");
            }
            if (result.extracted_features.head_pose !== "forward" &&
                result.extracted_features.head_pose !== "unknown") {
                addLogEntry(`Head turned: ${result.extracted_features.head_pose}`, "warning");
            }
        }

        if (result.risk_score >= 70) {
            addLogEntry(`HIGH RISK: Score ${result.risk_score}`, "danger");
        } else if (result.risk_score >= 30) {
            addLogEntry(`Moderate risk: Score ${result.risk_score}`, "warning");
        }

    } catch (err) {
        console.warn("Prediction request failed:", err.message);
    }
}

// ─── Risk Gauge ────────────────────────────────────────────────────
function updateRiskGauge(score, level, color) {
    // Update arc (total arc length ≈ 251.2)
    const arcLength = 251.2;
    const offset = arcLength - (score / 100) * arcLength;
    gaugeFill.style.strokeDashoffset = offset;
    gaugeFill.style.stroke = color;

    // Update text
    gaugeValue.textContent = score;
    gaugeValue.style.color = color;
    gaugeLabel.textContent = level;

    // Update webcam card border color based on risk
    const webcamCard = document.getElementById("webcam-card");
    webcamCard.style.borderColor = color;
}

// ─── Detection Log ─────────────────────────────────────────────────
function addLogEntry(message, type = "info") {
    const entry = document.createElement("div");
    entry.className = `log-entry ${type}`;

    const now = new Date();
    const time = now.toLocaleTimeString("en-US", { hour12: false });
    entry.textContent = `[${time}] ${message}`;

    detectionLog.appendChild(entry);
    detectionLog.scrollTop = detectionLog.scrollHeight;

    // Keep only the last 50 entries
    while (detectionLog.children.length > 50) {
        detectionLog.removeChild(detectionLog.firstChild);
    }
}

// ─── Submit Exam ───────────────────────────────────────────────────
function submitExam() {
    clearInterval(examTimer);
    clearInterval(captureTimer);
    clearInterval(idleTimer);

    // Stop webcam
    if (webcamStream) {
        webcamStream.getTracks().forEach(t => t.stop());
    }

    // Count correct answers
    let correct = 0;
    QUESTIONS.forEach((q, i) => {
        if (answers[i] === q.correct) correct++;
    });

    const answered = answers.filter(a => a !== null).length;
    const elapsed = EXAM_DURATION_MINUTES * 60 - timeRemaining;
    const elapsedMin = Math.floor(elapsed / 60);
    const elapsedSec = elapsed % 60;

    // Show modal
    const modalStats = document.getElementById("modal-stats");
    modalStats.innerHTML = `
        <div class="stat-row"><span>Questions Answered</span><strong>${answered} / ${QUESTIONS.length}</strong></div>
        <div class="stat-row"><span>Correct Answers</span><strong>${correct} / ${QUESTIONS.length}</strong></div>
        <div class="stat-row"><span>Time Taken</span><strong>${elapsedMin}m ${elapsedSec}s</strong></div>
        <div class="stat-row"><span>Tab Switches</span><strong>${behaviorStats.tabSwitches}</strong></div>
        <div class="stat-row"><span>Total Clicks</span><strong>${behaviorStats.clicks}</strong></div>
        <div class="stat-row"><span>Total Keystrokes</span><strong>${behaviorStats.keystrokes}</strong></div>
    `;

    document.getElementById("modal-overlay").classList.remove("hidden");
    document.getElementById("modal-close-btn").addEventListener("click", () => {
        document.getElementById("modal-overlay").classList.add("hidden");
    });
}
