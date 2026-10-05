const form = document.getElementById("predict-form");
const button = document.getElementById("predict-btn");
const resultBox = document.getElementById("result");
const labelEl = document.getElementById("result-label");
const probEl = document.getElementById("result-prob");
const barFill = document.getElementById("bar-fill");
const errorEl = document.getElementById("error");

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  errorEl.hidden = true;
  button.disabled = true;
  button.textContent = "Predicting...";

  // Collect every form field into a plain object
  const data = Object.fromEntries(new FormData(form).entries());

  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "Prediction failed");

    labelEl.textContent = result.prediction;
    labelEl.className = "result-label " + (result.prediction === "CHURN" ? "churn" : "safe");
    probEl.textContent = result.churn_probability + "%";
    barFill.style.width = result.churn_probability + "%";
    barFill.style.background = result.prediction === "CHURN" ? "#dc2626" : "#16a34a";
    resultBox.hidden = false;
    resultBox.scrollIntoView({ behavior: "smooth" });
  } catch (err) {
    errorEl.textContent = err.message;
    errorEl.hidden = false;
  } finally {
    button.disabled = false;
    button.textContent = "Predict";
  }
});
