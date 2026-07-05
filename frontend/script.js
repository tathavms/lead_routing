const API_URL = window.location.hostname === "127.0.0.1"
    ? "http://127.0.0.1:8000"
    : "http://13.48.162.241/lead-routing";

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
        } else if (data.department === "unrecognized") {
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
    resultDiv.innerHTML = message;
    resultDiv.className = `result ${type}`;
}