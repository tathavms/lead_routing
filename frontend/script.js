const API_URL = window.location.hostname === "127.0.0.1"
    ? "http://127.0.0.1:8000"
    : "http://13.48.162.241/lead-routing";

const THEME_KEY = "ticket-router-theme";

function initTheme() {
    let saved = null;
    try { saved = localStorage.getItem(THEME_KEY); } catch (e) {}
    const theme = saved === "dark" ? "dark" : "light";
    document.documentElement.setAttribute("data-theme", theme);
    document.getElementById("themeToggle").setAttribute("aria-checked", theme === "dark");
}

function toggleTheme() {
    const isDark = document.documentElement.getAttribute("data-theme") === "dark";
    const next = isDark ? "light" : "dark";
    document.documentElement.setAttribute("data-theme", next);
    document.getElementById("themeToggle").setAttribute("aria-checked", next === "dark");
    try { localStorage.setItem(THEME_KEY, next); } catch (e) {}
}

function updateClearBtn() {
    const text = document.getElementById("ticketInput").value.trim();
    document.getElementById("clearBtn").classList.toggle("active", text.length > 0);
}

function clearInput() {
    const input = document.getElementById("ticketInput");
    input.value = "";
    input.focus();
    updateClearBtn();
    const resultDiv = document.getElementById("result");
    resultDiv.className = "result hidden";
}

async function submitTicket() {
    const text = document.getElementById("ticketInput").value.trim();
    const resultDiv = document.getElementById("result");
    const btn = document.getElementById("submitBtn");

    if (!text) {
        showResult("Please describe your issue before submitting.", "error");
        return;
    }

    btn.disabled = true;
    btn.textContent = "Routing...";
    resultDiv.className = "result hidden";

    try {
        const response = await fetch(`${API_URL}/predict`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ user_input: text })
        });

        if (!response.ok) {
            throw new Error("Request failed");
        }

        const data = await response.json();

        if (data.is_spam) {
            showResult("Your message was flagged as spam and cannot be submitted.", "spam");
        } else if (data.department === "unrecognised" || data.department === "unrecognized") {
            showResult(
                `We could not understand your query (${data.confidence}% confidence).<br>Please describe a specific support issue.`,
                "error"
            );
        } else {
            showResult(
                `Routed to: <strong>${data.department}</strong><br>Confidence: <strong>${data.confidence}%</strong>`,
                "success"
            );
        }
    } catch (error) {
        showResult("Something went wrong. Please try again.", "error");
    } finally {
        btn.disabled = false;
        btn.textContent = "Submit Ticket";
    }
}

function showResult(message, type) {
    const resultDiv = document.getElementById("result");
    resultDiv.innerHTML = `<span class="dot"></span><span>${message}</span>`;
    resultDiv.className = `result ${type}`;
}

document.addEventListener("DOMContentLoaded", () => {
    initTheme();
    document.getElementById("themeToggle").addEventListener("click", toggleTheme);
    document.getElementById("clearBtn").addEventListener("click", clearInput);
    document.getElementById("ticketInput").addEventListener("input", updateClearBtn);
});
