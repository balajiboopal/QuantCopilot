# quantcopilot_app.py

import ast
import io
from typing import Dict, Any, List, Tuple

import textwrap
import re

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from fpdf import FPDF

from databricks import sql
from databricks.sdk.core import Config

# ============================================
# 0. Databricks connection helpers
# ============================================

@st.cache_resource
def get_cfg():
    """
    Get a Databricks Config object.

    Priority:
    1. Local/CLI config (works on your laptop where you've run `databricks configure`)
    2. Streamlit secrets (for Streamlit Cloud deployment)
    """
    try:
        # Local dev: use ~/.databrickscfg / env vars / AAD etc.
        return Config()
    except Exception as e:
        # Streamlit Cloud: fall back to secrets
        host = st.secrets.get("DATABRICKS_HOST")
        token = st.secrets.get("DATABRICKS_TOKEN")

        if not host or not token:
            raise ValueError(
                "Could not configure Databricks. Either:\n"
                "- configure the Databricks CLI locally (for local runs), or\n"
                "- set DATABRICKS_HOST and DATABRICKS_TOKEN in Streamlit secrets (for cloud)."
            ) from e

        return Config(host=host, token=token)


@st.cache_resource
def get_connection(http_path: str):
    """
    Create and cache a Databricks SQL connection.

    Authentication:
    - Uses Databricks CLI / ~/.databrickscfg via databricks-sdk Config(), OR
    - DATABRICKS_HOST / DATABRICKS_TOKEN from st.secrets on Streamlit Cloud.
    """
    cfg = get_cfg()

    server_hostname = cfg.host.replace("https://", "").replace("http://", "")

    return sql.connect(
        server_hostname=server_hostname,
        http_path=http_path,
        credentials_provider=lambda: cfg.authenticate,
    )



def query_to_pandas(conn, sql_text: str) -> pd.DataFrame:
    """Run a SQL query and return a Pandas DataFrame."""
    with conn.cursor() as cursor:
        cursor.execute(sql_text)
        return cursor.fetchall_arrow().to_pandas()


def call_llm_api(conn, prompt: str) -> str:
    """
    Call Databricks AI function via ai_query on the warehouse connection.
    Assumes 'databricks-meta-llama-3-1-405b-instruct' is available.
    """
    # Basic escaping of single quotes for SQL string
    safe_prompt = prompt.replace("'", "''")

    sql_text = f"""
        SELECT ai_query(
            'databricks-meta-llama-3-1-405b-instruct',
            '{safe_prompt}'
        ) AS response
    """

    with conn.cursor() as cursor:
        cursor.execute(sql_text)
        row = cursor.fetchone()
        return row[0] if row else ""


# ============================================
# 1. Misc helpers
# ============================================

def _round_or_none(x, ndigits: int = 2):
    try:
        if x is None:
            return None
        return round(float(x), ndigits)
    except Exception:
        return None


def prettify_flag(flag: str) -> str:
    if not flag:
        return ""
    txt = str(flag).replace("_", " ")
    txt = txt.strip()
    return txt


# ============================================
# 2. QuantCopilotAgent – main wrapper
# ============================================

class QuantCopilotAgent:
    def __init__(self, conn, model_name: str = "databricks-meta-llama-3-1-405b-instruct"):
        self.conn = conn
        self.model_name = model_name

    # -----------------------------
    # a) Fetch context from tables
    # -----------------------------
    def fetch_company_context(self, ticker: str) -> Dict[str, Any]:
        """
        Fetches:
        - company row from workspace.default.yfin_with_anomalies
        - macro credit snapshot from workspace.default.credit_macro_latest
        and maps them into a clean dict for the LLM prompt + UI.
        """

        # 1) Company row
        company_sql = f"""
            SELECT *
            FROM workspace.default.yfin_with_anomalies
            WHERE ticker = '{ticker.upper()}'
            LIMIT 1
        """
        df = query_to_pandas(self.conn, company_sql)

        if df.empty:
            raise ValueError(f"No data found for ticker {ticker}")

        row = df.iloc[0]

        # 2) Macro snapshot (single row)
        macro_sql = """
            SELECT *
            FROM workspace.default.credit_macro_latest
            LIMIT 1
        """
        macro_df = query_to_pandas(self.conn, macro_sql)

        macro: Dict[str, Any] = {}
        if not macro_df.empty:
            m = macro_df.iloc[0]
            macro = {
                "date": str(m.get("date")),
                "spread_bps": _round_or_none(m.get("spread_bps")),
                "spread_3m_change": _round_or_none(m.get("spread_3m_change")),
                "credit_risk_bucket": m.get("credit_risk_bucket"),
            }

        # 3) Red flags – robust handling
        red_flags_list: List[str] = []

        # Prefer red_flags
        if "red_flags" in row and row["red_flags"] is not None:
            val = row["red_flags"]
            if isinstance(val, list):
                red_flags_list = [prettify_flag(f) for f in val if f]
            else:
                try:
                    parsed = ast.literal_eval(str(val))
                    if isinstance(parsed, list):
                        red_flags_list = [prettify_flag(f) for f in parsed if f]
                except Exception:
                    pass

        # Fallback: red_flags_array
        if not red_flags_list and "red_flags_array" in row and row["red_flags_array"] is not None:
            arr_val = row["red_flags_array"]
            if isinstance(arr_val, list):
                red_flags_list = [prettify_flag(f) for f in arr_val if f]
            else:
                try:
                    parsed = ast.literal_eval(str(arr_val))
                    if isinstance(parsed, list):
                        red_flags_list = [prettify_flag(f) for f in parsed if f]
                except Exception:
                    pass

        red_flags = {
            "red_flag_score": _round_or_none(row.get("red_flag_score", 0.0)),
            "red_flag_severity": row.get("red_flag_severity", None),
            "red_flag_count": int(row.get("red_flag_count", 0)) if "red_flag_count" in row else 0,
            "red_flags": red_flags_list,
        }

        # 4) Build context dict from your existing columns
        context: Dict[str, Any] = {
            # Identity
            "ticker": row["ticker"],
            "company_name": row.get("company_name"),
            "sector": row.get("sector"),
            "industry": row.get("industry"),
            "country": row.get("country"),
            "exchange": row.get("exchange"),
            "currency": row.get("currency"),
            "market_cap": _round_or_none(row.get("market_cap"), 0),

            # Scores
            "value_score": _round_or_none(row.get("value_score", 0.0)),
            "growth_score": _round_or_none(row.get("growth_score", 0.0)),
            "quality_score": _round_or_none(row.get("quality_score", 0.0)),
            "risk_score": _round_or_none(row.get("risk_score", 0.0)),
            "overall_alpha_score": _round_or_none(row.get("overall_alpha_score", 0.0)),
            "conviction_rating": row.get("conviction"),

            # Risk & volatility
            "beta": _round_or_none(row.get("beta", 0.0)),
            "volatility_30d": _round_or_none(row.get("volatility_30d", 0.0)),
            "volatility_90d": _round_or_none(row.get("volatility_90d", 0.0)),
            "volatility_52w": _round_or_none(row.get("volatility_52w", 0.0)),

            # Cash flow / leverage
            "current_price": _round_or_none(row.get("current_price", row.get("open", None))),
            "free_cash_flow": _round_or_none(row.get("free_cashflow", 0.0)),
            "free_cash_flow_margin": _round_or_none(row.get("free_cash_flow_margin", 0.0)),
            "fcf_yield": _round_or_none(row.get("fcf_yield", 0.0)),
            "debt_to_equity": _round_or_none(row.get("debt_to_equity", 0.0)),
            "current_ratio": _round_or_none(row.get("current_ratio", 0.0)),
            "cash_to_debt": _round_or_none(row.get("cash_to_debt", 0.0)),

            # Attach macro + red flags
            "macro": macro,
            "red_flags": red_flags,

            # Placeholder for peers (wire later)
            "peers": [],
        }

        return context

    # -----------------------------
    # b) Build the LLM prompt
    # -----------------------------
    def build_prompt(self, context: Dict[str, Any], user_question: str) -> str:
        macro = context.get("macro", {})
        red = context.get("red_flags", {})

        red_flags_list = red.get("red_flags", []) or []
        red_flags_str = ", ".join(red_flags_list) if red_flags_list else "None detected"

        prompt = f"""
You are **QuantCopilot**, an AI equity analyst. You must give a **realistic, numerically grounded, institution-grade** analysis.

CRITICAL RULES:
- Use ONLY the data provided below. If something is missing, say it is missing.
- Always tie your reasoning back to specific metrics (e.g., "D/E = {context.get('debt_to_equity')}", "FCF yield ≈ {context.get('fcf_yield')}%").
- Do NOT hallucinate peer statistics or future events.
- Be consistent with the provided **conviction rating** and **overall alpha score** when giving a Buy/Hold/Sell view.

FORWARD-LOOKING QUESTIONS:
- You are NOT a forecasting model. Do NOT invent precise numeric forecasts for future returns, prices, or growth rates if they are not given in the data.
- If the user asks about future growth, returns, or target prices:
  - Use the HISTORICAL metrics provided (revenue growth, earnings growth, margins, etc.) to give a QUALITATIVE view (e.g., "strong growth outlook", "decelerating", "uncertain").
  - You may discuss SCENARIOS and RISK, but NOT exact percentages or price targets.

========================
USER QUESTION
========================
{user_question}

========================
COMPANY SNAPSHOT
========================
Ticker: {context.get('ticker')}
Name: {context.get('company_name')}
Sector: {context.get('sector')}
Industry: {context.get('industry')}
Country: {context.get('country')}
Exchange / Currency: {context.get('exchange')} / {context.get('currency')}
Market cap: {context.get('market_cap')}

========================
FACTOR SCORES (0–10 scale, higher is better)
========================
Value score:      {context.get('value_score')}
Growth score:     {context.get('growth_score')}
Quality score:    {context.get('quality_score')}
Risk score:       {context.get('risk_score')}
Overall alpha:    {context.get('overall_alpha_score')}
Conviction:       {context.get('conviction_rating')}   (this should heavily influence your final stance)

Explain briefly what these scores imply about valuation, growth, quality, and risk profile.

========================
RISK, LEVERAGE & CASH FLOW
========================
Beta:                 {context.get('beta')}
30D volatility:       {context.get('volatility_30d')}
90D volatility:       {context.get('volatility_90d')}
52W volatility:       {context.get('volatility_52w')}
Debt to equity:       {context.get('debt_to_equity')}
Current ratio:        {context.get('current_ratio')}
Cash to debt:         {context.get('cash_to_debt')}
FCF yield:            {context.get('fcf_yield')}
Free cash flow margin:{context.get('free_cash_flow_margin')}

Beta interpretation rules:
- beta < 0.8 = low market sensitivity
- 0.8 <= beta <= 1.2 = roughly market-like
- beta > 1.2 = above-average market sensitivity
Use this language explicitly (low / market-like / high beta).

When discussing risk:
- High D/E and low cash-to-debt = balance sheet risk.
- High volatility = higher price risk.
- High FCF yield and strong FCF margin = potential cushion, BUT may not fully offset extreme leverage.

========================
MACRO CREDIT ENVIRONMENT
========================
BAA-AAA spread (bps): {macro.get('spread_bps')}
3M change (bps):      {macro.get('spread_3m_change')}
Credit risk bucket:   {macro.get('credit_risk_bucket')}

INTERPRETATION RULES:
- Lower spreads and a LOW credit-risk bucket generally mean easier credit conditions and lower systemic credit stress.
- A NEGATIVE 3M change in spread = spreads have narrowed recently → credit conditions have improved slightly.
- A POSITIVE 3M change in spread = spreads have widened → credit conditions have worsened.

Use this to comment on:
- Whether the current macro regime is supportive or hostile for a highly levered company like this one.
- How sensitive the company might be to a future deterioration in credit conditions.

========================
RED FLAGS
========================
Red flag score:    {red.get('red_flag_score')}
Red flag severity: {red.get('red_flag_severity')}
Red flag count:    {red.get('red_flag_count')}
Flags:             {red_flags_str}

You MUST explicitly explain:
- Why each major flag (e.g., "extreme debt") matters.
- How it interacts with leverage metrics and macro conditions.

========================
OUTPUT FORMAT (REQUIRED)
========================
Respond with clear sections:

1. **High-Level View**
- One short paragraph summarizing whether this looks attractive or not for a **long-term investor**, and why.

2. **Key Positives**
- 3–5 bullet points highlighting genuine strengths, each tied to specific metrics.

3. **Key Risks**
- 3–5 bullet points on the main risks.
- You MUST include leverage-related risks if D/E is high or "extreme debt" is flagged.
- Mention volatility and quality score if they are weak.

4. **Macro & Scenario View**
- Briefly explain how the current macro credit regime (spread level, 3M trend, LOW/MEDIUM/HIGH bucket) affects this company.
- Add a short “what if credit spreads widen again?” scenario and how that might particularly impact a company with this leverage and risk profile.

5. **Final Verdict**
- Give a clear **Buy / Hold / Sell** style conclusion.
- This verdict MUST be consistent with the provided conviction rating (`{context.get('conviction_rating')}`) and overall alpha score (`{context.get('overall_alpha_score')}`).
- Make it obvious what kind of investor (e.g., conservative income investor vs. high-risk speculator) might consider this.

Keep the answer concise but insightful. Do not restate raw data as a list; always interpret what it means for an investor.
"""
        return prompt

    # -----------------------------
    # c) LLM call wrapper
    # -----------------------------
    def call_llm(self, prompt: str) -> str:
        return call_llm_api(self.conn, prompt)

    # -----------------------------
    # d) Public entrypoints
    # -----------------------------
    def answer(self, ticker: str, user_question: str) -> str:
        context = self.fetch_company_context(ticker)
        prompt = self.build_prompt(context, user_question)
        return self.call_llm(prompt)

    def answer_with_context(self, ticker: str, user_question: str) -> Tuple[str, Dict[str, Any]]:
        """
        Returns both:
        - llm_text: the narrative
        - context: the numeric + categorical features used in the prompt
        """
        context = self.fetch_company_context(ticker)
        prompt = self.build_prompt(context, user_question)
        llm_text = self.call_llm(prompt)
        return llm_text, context

    def generate_scenarios(self, ticker: str, horizon_desc: str = "the next 12–18 months") -> str:
        """
        Safe, non-numeric scenario analysis.
        """
        context = self.fetch_company_context(ticker)

        scenario_prompt = f"""
You are QuantCopilot, an equity analyst.

You will generate **non-numeric, scenario-based** outlooks for a single stock
over {horizon_desc}. You MUST NOT invent specific numeric forecasts for prices,
returns, or growth rates.

Here is the company context:
{context}

TASK:
1. Produce three scenarios: Bull, Base, Bear.
2. For each scenario, describe:
   - Key drivers (demand, margins, leverage, macro credit conditions).
   - How the company's risk profile changes (qualitatively).
   - What kind of investor might be comfortable with that scenario.
3. Refer to:
   - Value / Growth / Quality / Risk scores,
   - Leverage, FCF, volatility,
   - Macro credit regime (spreads, bucket),
   - Red flags (e.g., extreme debt).
4. Explicit rule:
   - You may say things like "strong upside", "moderate upside", "downside risk",
     but MUST NOT say "expected +15%" or "price target 250".

Format:

**Bull scenario**
- ...

**Base scenario**
- ...

**Bear scenario**
- ...
"""
        return self.call_llm(scenario_prompt)

    # -----------------------------
    # e) NL → SQL Screener
    # -----------------------------
    def build_screener_sql_prompt(self, nl_query: str) -> str:
        prompt = f"""
You are an assistant that converts natural language stock screening requests
into a single valid SQL query.

The table is: workspace.default.yfin_with_anomalies

Important columns you can use:
- ticker, company_name, sector, industry, country
- value_score, growth_score, quality_score, risk_score, overall_alpha_score
- debt_to_equity, current_ratio, cash_to_debt
- fcf_yield, free_cash_flow_margin
- volatility_30d, volatility_90d, volatility_52w
- beta, market_cap, dividend_yield, profit_margin, operating_margin

Rules:
- ONLY output a valid SQL SELECT statement.
- Do NOT include any explanation, comments, or markdown.
- Use ORDER BY and LIMIT when appropriate.
- For "undervalued", prefer higher value_score.
- For "high growth", prefer higher growth_score.
- For "high quality", prefer higher_quality_score.
- For "low risk", prefer lower risk_score, lower volatility, lower beta, and lower debt_to_equity.
- If the user says "US" or "United States", filter with country = 'United States'.

User request:
\"\"\"{nl_query}\"\"\"
"""
        return prompt

    def generate_screener_sql(self, nl_query: str) -> str:
        prompt = self.build_screener_sql_prompt(nl_query)
        raw = self.call_llm(prompt).strip()

# Remove markdown fences or labels
        cleaned = (
            raw.replace("```sql", "")
            .replace("```", "")
            .replace("sql\n", "")
            .replace("sql ", "")
            .strip()
        )

        sql_text = cleaned

        if "SELECT" not in sql_text.upper():
            raise ValueError(f"LLM did not return a SQL query. Got:\n{sql_text[:500]}")

        return sql_text

    def screen_stocks(self, nl_query: str, max_rows: int = 10) -> Dict[str, Any]:
        """
        High-level entrypoint for the stock screener.

        1. Generate SQL from NL.
        2. Run the query on yfin_with_anomalies.
        3. Take the top N rows.
        4. Ask the LLM to summarize and explain.
        """
        generated_sql = self.generate_screener_sql(nl_query)

        results_df = query_to_pandas(self.conn, generated_sql)
        pdf = results_df.head(max_rows)

        if pdf.empty:
            explanation = f"No results found for query: {nl_query}"
            return {
                "sql": generated_sql,
                "results": pdf,
                "explanation": explanation,
            }

        sample_rows = pdf.to_dict(orient="records")

        expl_prompt = f"""
You are QuantCopilot, an AI equity analyst.

You were asked to screen stocks with the following natural language request:
\"\"\"{nl_query}\"\"\"

The following rows were returned from the database (one per stock):
{sample_rows}

TASK:
1. Briefly explain how this screen interprets the user's request.
2. Highlight 3–5 of the most interesting tickers and explain WHY they fit the screen,
   referencing specific metrics (scores, leverage, volatility, etc.)
3. Call out any potential concerns or limitations of this screen (e.g., sector concentration,
   small sample size, missing data, etc.).
4. End with a short note on what type of investor this screen is best suited for.

Keep it concise but insightful.
"""
        explanation = self.call_llm(expl_prompt)

        return {
            "sql": generated_sql,
            "results": pdf,
            "explanation": explanation,
        }


# ============================================
# 3. Quant utilities
# ============================================

def factor_expected_return(ctx: Dict[str, Any]) -> float:
    """
    Toy factor-based expected annualized return (in %), using factor scores.
    This is NOT a forecast, just a heuristic ranking measure.
    """
    v = ctx.get("value_score") or 0.0
    g = ctx.get("growth_score") or 0.0
    q = ctx.get("quality_score") or 0.0
    r = ctx.get("risk_score") or 0.0

    alpha_units = (
        0.25 * (v - 5.0) +
        0.35 * (g - 5.0) +
        0.25 * (q - 5.0) -
        0.30 * (r - 5.0)
    )

    base_market_return = 6.0  # long-run equity-type number
    expected_return = base_market_return + alpha_units
    return expected_return


def monte_carlo_paths(
    spot_price: float,
    exp_return_annual: float,
    vol_annual_pct: float,
    days: int = 252,
    n_paths: int = 200,
) -> pd.DataFrame:
    """
    Simple GBM Monte Carlo. Returns a DataFrame of shape (days+1, n_paths).
    vol_annual_pct is e.g. 30 for 30% annual vol.
    exp_return_annual is in %, e.g. 8 for 8%.
    """
    mu = exp_return_annual / 100.0
    sigma = vol_annual_pct / 100.0

    dt = 1.0 / 252.0
    steps = days

    shocks = np.random.normal(
        loc=(mu - 0.5 * sigma**2) * dt,
        scale=sigma * np.sqrt(dt),
        size=(steps, n_paths),
    )

    log_paths = np.vstack([np.zeros((1, n_paths)), shocks.cumsum(axis=0)])
    paths = spot_price * np.exp(log_paths)

    idx = pd.RangeIndex(start=0, stop=steps + 1, step=1)
    return pd.DataFrame(paths, index=idx)


def mean_variance_optimize(returns: pd.DataFrame) -> pd.Series:
    """
    Very simple heuristic optimizer: inverse-variance weights (risk-parity-ish).
    """
    cov = returns.cov()
    diag = np.diag(cov)
    inv_var = 1.0 / diag
    w = inv_var / inv_var.sum()
    return pd.Series(w, index=returns.columns, name="weight")


def radar_chart_for_factors(ctx: Dict[str, Any]):
    labels = ["Value", "Growth", "Quality", "Risk (inverted)"]
    value = ctx.get("value_score") or 0
    growth = ctx.get("growth_score") or 0
    quality = ctx.get("quality_score") or 0
    risk = ctx.get("risk_score") or 0

    # Invert risk so higher is visually "better"
    risk_inv = 10 - risk if risk is not None else 0

    values = [value, growth, quality, risk_inv]
    values.append(values[0])  # close loop
    labels_closed = labels + [labels[0]]

    fig = go.Figure(
        data=go.Scatterpolar(
            r=values,
            theta=labels_closed,
            fill="toself",
        )
    )
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 10])),
        showlegend=False,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig


def memo_to_pdf(memo_text: str) -> bytes:
    """
    Clean the LLM memo and render it as a simple PDF.
    Avoids FPDF width issues and unicode issues.
    """
    import re, textwrap
    from fpdf import FPDF

    # ---- Clean markdown / emoji / unicode ----
    cleaned = re.sub(r"\*\*(.*?)\*\*", r"\1", memo_text)   # **bold**
    cleaned = re.sub(r"\*(.*?)\*", r"\1", cleaned)         # *italic*
    cleaned = cleaned.replace("#", "")

    # Drop any characters FPDF can't handle (keep Latin-1 only)
    cleaned = cleaned.encode("latin-1", "ignore").decode("latin-1")

    # ---- Set up PDF ----
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_margins(left=10, top=10, right=10)
    pdf.add_page()

    # Use core font directly to avoid Arial deprecation warning
    pdf.set_font("Helvetica", size=11)

    # Effective page width (total width minus margins)
    effective_width = pdf.w - 2 * pdf.l_margin

    # ---- Write wrapped text ----
    for line in cleaned.split("\n"):
        wrapped_lines = textwrap.wrap(line, width=110) or [""]
        for wl in wrapped_lines:
            pdf.multi_cell(effective_width, 5, wl)

    # FPDF returns a bytearray or bytes when dest="S"
    data = pdf.output(dest="S")
    if isinstance(data, bytearray):
        return bytes(data)
    return data  # already bytes


# ============================================
# 4. Streamlit App
# ============================================

st.set_page_config(page_title="QuantCopilot", layout="wide")

st.title("QuantCopilot – LLM-Powered Quant Assistant")

# ---------------- Sidebar: Databricks connection ----------------
st.sidebar.header("Databricks SQL Settings")
http_path = st.sidebar.text_input(
    "Databricks HTTP Path",
    placeholder="/sql/1.0/warehouses/cb66ba9ae8da5d24",
)

conn = None
agent = None
conn_error = None

if http_path:
    try:
        conn = get_connection(http_path)
        agent = QuantCopilotAgent(conn)
    except Exception as e:
        conn_error = str(e)

# ---------------- Sidebar: App controls ----------------
st.sidebar.markdown("---")
st.sidebar.header("Single Stock Analysis")
ticker = st.sidebar.text_input("Ticker", value="NEM")
question = st.sidebar.text_area(
    "Question",
    value="Is this a good long-term investment, and what are the main risks?"
)

horizon = st.sidebar.text_input(
    "Scenario horizon (for Scenario Lab)",
    value="the next 12–18 months"
)

run_single = st.sidebar.button("Analyze Ticker", disabled=(agent is None))

st.sidebar.markdown("---")
st.sidebar.header("Stock Screener")

nl_query = st.sidebar.text_area(
    "Natural-language screen",
    value="find 10 high-quality, low-debt US stocks with strong free cash flow and low volatility"
)
run_screen = st.sidebar.button("Run Screener", disabled=(agent is None))

st.sidebar.markdown("---")
st.sidebar.header("Portfolio Lab")
uploaded_prices = st.sidebar.file_uploader(
    "Upload CSV of daily price history (columns are tickers, rows are dates)",
    type=["csv"]
)

# If connection failed, surface it clearly
if conn_error:
    st.error(f"Error connecting to Databricks: {conn_error}")

# ---------------- Tabs ----------------
tab1, tab2, tab3, tab4 = st.tabs(
    ["Single Stock", " Screener", " Scenario Lab", " Portfolio Lab"]
)

# -------------------------------------------------
# TAB 1: Single Stock View
# -------------------------------------------------
with tab1:
    st.subheader("Single Stock Analysis")

    if agent is None:
        st.info("Enter a valid Databricks HTTP Path in the sidebar to enable analysis.")
    elif run_single:
        try:
            with st.spinner(f"Analyzing {ticker.upper()}..."):
                llm_text, ctx = agent.answer_with_context(ticker, question)

            st.markdown(f"### LLM View on **{ticker.upper()}**")
            st.markdown(llm_text)

            st.markdown("### Factor Radar")
            fig = radar_chart_for_factors(ctx)
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("### Key Metrics")
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Value score", ctx.get("value_score"))
                st.metric("Growth score", ctx.get("growth_score"))
                st.metric("Quality score", ctx.get("quality_score"))
                st.metric("Risk score", ctx.get("risk_score"))

            with col2:
                st.metric("Debt / Equity", ctx.get("debt_to_equity"))
                st.metric("FCF Yield (%)", ctx.get("fcf_yield"))
                st.metric("FCF Margin (%)", ctx.get("free_cash_flow_margin"))
                st.metric("Current ratio", ctx.get("current_ratio"))

            with col3:
                st.metric("Beta", ctx.get("beta"))
                st.metric("Volatility 30d (%)", ctx.get("volatility_30d"))
                st.metric("Volatility 90d (%)", ctx.get("volatility_90d"))
                st.metric("Volatility 52w (%)", ctx.get("volatility_52w"))

            st.markdown("### Macro & Red Flags")
            macro = ctx.get("macro", {}) or {}
            red = ctx.get("red_flags", {}) or {}

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Macro Credit Environment**")
                st.write(f"Spread (bps): {macro.get('spread_bps')}")
                st.write(f"3M change (bps): {macro.get('spread_3m_change')}")
                st.write(f"Credit bucket: {macro.get('credit_risk_bucket')}")

            with c2:
                st.markdown("**Red Flags**")
                st.write(f"Score: {red.get('red_flag_score')}")
                st.write(f"Severity: {red.get('red_flag_severity')}")
                st.write(f"Count: {red.get('red_flag_count')}")
                st.write(f"Flags: {', '.join(red.get('red_flags', [])) or 'None'}")

            # PDF memo download
            st.markdown("###  Export Investment Memo")
            memo_bytes = memo_to_pdf(llm_text)
            st.download_button(
                label="Download memo as PDF",
                data=memo_bytes,
                file_name=f"{ticker.upper()}_investment_memo.pdf",
                mime="application/pdf",
            )

        except Exception as e:
            st.error(f"Error analyzing {ticker.upper()}: {e}")
    else:
        st.info("Use the sidebar to set Databricks HTTP Path, Ticker & Question, then click **Analyze Ticker**.")

# -------------------------------------------------
# TAB 2: Screener View
# -------------------------------------------------
with tab2:
    st.subheader("Natural-Language Stock Screener")

    if agent is None:
        st.info("Enter a valid Databricks HTTP Path in the sidebar to enable the screener.")
    elif run_screen:
        try:
            with st.spinner("Running screen..."):
                out = agent.screen_stocks(nl_query, max_rows=10)

            st.markdown("###  Generated SQL")
            st.code(out["sql"], language="sql")

            st.markdown("### Results")
            if out["results"].empty:
                st.warning("No results returned for this screen.")
            else:
                st.dataframe(out["results"])

            st.markdown("### LLM Explanation")
            st.markdown(out["explanation"])

            if not out["results"].empty:
                df_plot = out["results"].copy()
                if {"value_score", "risk_score"}.issubset(df_plot.columns):
                    st.markdown("#### Value vs Risk Scatter")
                    df_plot = df_plot.rename(
                        columns={"value_score": "Value", "risk_score": "Risk"}
                    )
                    st.scatter_chart(
                        df_plot,
                        x="Value",
                        y="Risk",
                    )
        except Exception as e:
            st.error(f"Error running screener: {e}")
    else:
        st.info("Enter a natural-language screen in the sidebar, then click **Run Screener**.")

# -------------------------------------------------
# TAB 3: Scenario Lab
# -------------------------------------------------
with tab3:
    st.subheader("Scenario Lab – Non-numeric Forward View")

    if agent is None:
        st.info("Enter a valid Databricks HTTP Path in the sidebar to enable Scenario Lab.")
    elif run_single:
        try:
            with st.spinner(f"Generating Bull/Base/Bear for {ticker.upper()}..."):
                scenarios = agent.generate_scenarios(ticker, horizon_desc=horizon)

            st.markdown(f"### Scenarios for **{ticker.upper()}** over {horizon}")
            st.markdown(scenarios)

            # Optional: Monte Carlo visualization
            st.markdown("### Monte Carlo Price Paths (Illustrative)")
            ctx = agent.fetch_company_context(ticker)
            spot = ctx.get("current_price") or 100.0
            vol = ctx.get("volatility_52w") or 30.0
            exp_ret = factor_expected_return(ctx)

            st.write(
                f"Using spot={spot}, vol_annual={vol}%, "
                f"factor-expected return≈{_round_or_none(exp_ret, 2)}%"
            )

            paths = monte_carlo_paths(
                spot_price=spot,
                exp_return_annual=exp_ret,
                vol_annual_pct=vol,
                days=126,
                n_paths=50,
            )
            st.line_chart(paths.iloc[:, :20])  # show 20 paths

            st.caption("Monte Carlo is illustrative only – not a real price forecast.")

        except Exception as e:
            st.error(f"Error generating scenarios: {e}")
    else:
        st.info("First run **Analyze Ticker** in the Single Stock tab (sidebar), then come here.")

# -------------------------------------------------
# TAB 4: Portfolio Lab
# -------------------------------------------------
with tab4:
    st.subheader("Portfolio Lab – Simple Optimization")

    if uploaded_prices is not None:
        try:
            prices = pd.read_csv(uploaded_prices, index_col=0)
            prices.index = pd.to_datetime(prices.index)

            st.markdown("### Uploaded Price Data (head)")
            st.dataframe(prices.head())

            returns = prices.pct_change().dropna(how="any")
            st.markdown("### Daily Returns (head)")
            st.dataframe(returns.head())

            weights = mean_variance_optimize(returns)
            st.markdown("### Suggested Weights (Inverse-Variance Heuristic)")
            st.dataframe(weights.to_frame("weight"))

            # Simple portfolio stats
            port_ret_daily = (returns * weights).sum(axis=1)
            mean_daily = port_ret_daily.mean()
            vol_daily = port_ret_daily.std()
            ann_ret = mean_daily * 252
            ann_vol = vol_daily * np.sqrt(252)

            st.markdown("###  Portfolio Stats (approx.)")
            st.write(f"Annualized return (approx): {ann_ret*100:.2f}%")
            st.write(f"Annualized volatility (approx): {ann_vol*100:.2f}%")

            st.markdown("###  Portfolio Return Distribution")
            st.line_chart(port_ret_daily.cumsum())

        except Exception as e:
            st.error(f"Error in Portfolio Lab: {e}")
    else:
        st.info("Upload a CSV of daily prices in the sidebar to run the Portfolio Lab.")