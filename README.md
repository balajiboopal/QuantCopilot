## What QuantCopilot Does

---

### 1. Analyze Any Stock (LLM + Fundamentals)

Give it a ticker and your question.

QuantCopilot pulls fundamentals from Unity Catalog and feeds them into Databricks Llama-3 to generate a **structured, institutional-grade analysis**.

You get:
- A high-level view  
- Key positives  
- Key risks  
- Macro interpretation  
- A final Buy/Hold/Sell stance (aligned with your factor model)

No hallucinations — the model uses **only the data you provide**.

---

### 2. Natural-Language Stock Screener

Just type:

> “find high-quality, low-debt US stocks with strong FCF”

QuantCopilot:
1. Converts that request into SQL  
2. Runs it on your Databricks table  
3. Returns the matching stocks  
4. Explains *why* they fit the screen  

Feels like magic — except it’s grounded in real data.

---

### 3. Scenario Lab (Bull / Base / Bear)

One of the most fun parts of the project.

QuantCopilot generates:
- **Bull case**  
- **Base case**  
- **Bear case**

All **non-numeric**, grounded in real fundamentals + macro credit environment.  
No made-up price targets — just clean, qualitative scenario thinking.

Includes an **illustrative Monte Carlo simulation** for volatility.

---

### 4. Portfolio Lab

Upload any CSV of daily prices and instantly get:
- Daily returns  
- Inverse-variance (risk-parity-style) weights  
- Approx annualized return  
- Approx annualized volatility  
- Cumulative return chart  

A lightweight portfolio analytics sandbox built directly into the app.

---

### 5. Export Investment Memo (PDF)

Whatever the LLM writes can be exported as a clean, readable **PDF memo**.  
Perfect for sharing with teammates or dropping into presentations.
Accessible to all types of people: Structured for readability, the output is clear for beginners, yet detailed enough for expert analysis. 

---
