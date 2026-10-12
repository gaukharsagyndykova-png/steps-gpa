"""Steps and Academic Performance — Mathematical Statistics project (Streamlit app)."""
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats

st.set_page_config(
    page_title="Steps & Academic Performance",
    page_icon="👟",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_DIR = Path(__file__).parent / "data"
ALPHA = 0.05
BLUE = "#1F6FB2"
ORANGE = "#D9792B"
RED = "#B4432F"

# ---------------------------------------------------------------- style
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,700;12..96,800&display=swap');
[data-testid="stSidebar"] {background:#14231F;}
[data-testid="stSidebar"] * {color:#E7EEE9 !important;}
.hero {background:linear-gradient(135deg,#14231F 0%,#1D3A33 60%,#1F4E6E 100%);
  border-radius:20px;padding:36px 40px;margin-bottom:8px;color:#F3F5F1;}
.hero .eyebrow {font:600 12px monospace;letter-spacing:.14em;text-transform:uppercase;color:#F0A15C;}
.hero h1 {font-family:'Bricolage Grotesque',sans-serif;font-weight:800;font-size:46px;line-height:1.05;
  margin:10px 0 12px;color:#F3F5F1;}
.hero p {font-size:18px;color:#C9D6CF;max-width:60ch;margin:0;}
.sec-no {font:600 12px monospace;letter-spacing:.14em;text-transform:uppercase;color:#D9792B;margin-bottom:-8px;}
.card {border:1px solid rgba(128,128,128,.25);border-radius:14px;padding:18px 20px;height:100%;}
.card h4 {margin:0 0 6px 0;}
.verdict-yes {border:2px solid #2E8B57;border-radius:12px;padding:14px 18px;}
.verdict-no {border:2px solid #B4432F;border-radius:12px;padding:14px 18px;}
</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------- data
@st.cache_data
def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "survey_responses.csv")


df_all = load_data()
clean = df_all[df_all["Group"] == "ok"].copy()

REASONS = {
    "dup": "Duplicate submission",
    "nophone": "Does not carry the phone",
    "impossible": "Impossible step count",
    "outlier": "Outlier (IQR rule)",
}


def corr_test(x, y, method="Pearson"):
    """Return r, t, p, n for H0: rho = 0 (two-tailed)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = len(x)
    if method == "Spearman":
        r = stats.spearmanr(x, y).statistic
    else:
        r = stats.pearsonr(x, y).statistic
    t = r * np.sqrt(n - 2) / np.sqrt(1 - r**2)
    p = 2 * stats.t.sf(abs(t), n - 2)
    return float(r), float(t), float(p), n


def regression(x, y):
    res = stats.linregress(x, y)
    return res.intercept, res.slope, res.rvalue**2


def scatter_chart(data: pd.DataFrame, x_max: int = 16000):
    base = alt.Chart(data).encode(
        x=alt.X("Steps:Q", title="Average steps per day", scale=alt.Scale(domain=[0, x_max])),
        y=alt.Y("GPA:Q", title="GPA", scale=alt.Scale(domain=[1.8, 4.05])),
    )
    data = data.assign(Highlight=np.where(data["ID"] == 15, "Steps 15 000, GPA 3.95", "Student"))
    points = (
        alt.Chart(data)
        .mark_circle(size=110, opacity=0.85)
        .encode(
            x="Steps:Q",
            y="GPA:Q",
            color=alt.Color(
                "Highlight:N",
                scale=alt.Scale(domain=["Student", "Steps 15 000, GPA 3.95"], range=[BLUE, ORANGE]),
                legend=alt.Legend(title=None, orient="bottom"),
            ),
            tooltip=["ID", "Steps", "GPA", "Sleep", "Stress", "Year"],
        )
    )
    line = base.transform_regression("Steps", "GPA").mark_line(strokeDash=[8, 6], color="gray", size=2.5)
    return (points + line).properties(height=420)


def histogram(data: pd.DataFrame, col: str, step: float, title: str, color: str):
    return (
        alt.Chart(data)
        .mark_bar(color=color, cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
        .encode(
            x=alt.X(f"{col}:Q", bin=alt.Bin(step=step), title=title),
            y=alt.Y("count():Q", title="Number of students"),
            tooltip=[alt.Tooltip("count():Q", title="Students")],
        )
        .properties(height=300)
    )


def section_header(number: str, title: str, text: str = ""):
    st.markdown(f'<div class="sec-no">{number}</div>', unsafe_allow_html=True)
    st.header(title, anchor=False)
    if text:
        st.write(text)


# ---------------------------------------------------------------- sidebar
SECTIONS = [
    "Overview",
    "Data Collection",
    "Data Cleaning",
    "Frequency Distribution",
    "Descriptive Statistics",
    "Correlation & Regression",
    "Hypothesis Testing",
    "Robustness Lab",
    "Power BI Dashboard",
    "Limitations & Conclusion",
]
with st.sidebar:
    st.markdown("## Steps & academic performance")
    st.caption("A survey study of 60 students")
    page = st.radio("Sections", SECTIONS, label_visibility="collapsed")
    st.divider()
    st.caption(f"Survey: 11 October 2026  \n{len(df_all)} answers · {len(clean)} analysed  \nα = {ALPHA}")

st.markdown(
    """
<div class="hero">
  <div class="eyebrow">Mathematical statistics project</div>
  <h1>Do daily steps affect students' grades?</h1>
  <p>We asked 60 students for their average daily step count from their phone, their GPA, sleep and stress,
  and tested whether walking more goes with better grades.</p>
</div>
""",
    unsafe_allow_html=True,
)

r_main, t_main, p_main, n_main = corr_test(clean["Steps"], clean["GPA"])

# ---------------------------------------------------------------- 01 Overview
if page == "Overview":
    section_header("01 / Overview", "The research")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Answers collected", len(df_all))
    c2.metric("Answers analysed", n_main)
    c3.metric("Correlation r (steps, GPA)", f"{r_main:.3f}")
    c4.metric("p-value", f"{p_main:.3f}")

    a, b = st.columns(2)
    a.markdown(
        '<div class="card"><h4>Problem</h4>Students sit most of the day: at lectures, on the bus, '
        "at the computer. Walking is known to help concentration, sleep and mood, so it might also be "
        "linked to grades.</div>",
        unsafe_allow_html=True,
    )
    b.markdown(
        '<div class="card"><h4>Research question</h4>Is the average number of steps a student takes '
        "per day related to their GPA? We also look at sleep and stress as possible links.</div>",
        unsafe_allow_html=True,
    )
    st.subheader("Hypotheses", anchor=False)
    st.markdown(
        "- **H₀:** ρ = 0 — there is no linear relationship between daily steps and GPA.\n"
        "- **H₁:** ρ ≠ 0 — there is a linear relationship between daily steps and GPA.\n"
        f"- Significance level **α = {ALPHA}**, two-tailed test."
    )

# ---------------------------------------------------------------- 02 Data collection
elif page == "Data Collection":
    section_header(
        "02 / Data collection",
        "An anonymous survey with step counts from phones",
        "Students copied their 7-day average from Apple Health, Google Fit or Samsung Health, so the main "
        "variable is measured by the phone, not guessed. The Google Form was in English, Russian and Kazakh "
        "and stored no names or emails.",
    )
    left, right = st.columns([1.4, 1])
    with left:
        questions = pd.DataFrame(
            {
                "Variable": ["Steps", "CarriesPhone", "Sleep", "Stress", "GPA", "Year"],
                "Question": [
                    "Average steps per day, last 7 days",
                    "Do you usually carry your phone when you walk?",
                    "Hours of sleep per night",
                    "Stress level during the last week",
                    "Current GPA",
                    "Year of study",
                ],
                "Answer type": ["Number", "Yes / No", "Number", "Scale 1–10", "Number", "1 / 2 / 3 / 4"],
                "Check": ["0 – 40 000", "Required", "0 – 14", "Required", "0 – 4", "Required"],
            }
        )
        st.dataframe(questions, hide_index=True, use_container_width=True)
        st.caption(f"{len(df_all)} answers were received by 15:45 on 11 October 2026.")
    with right:
        form_img = DATA_DIR / "google_form.png"
        if form_img.exists():
            st.image(str(form_img), caption="The trilingual Google Form")

    with st.expander("Show all 60 raw answers"):
        st.dataframe(df_all.drop(columns=["Group"]), hide_index=True, use_container_width=True)

# ---------------------------------------------------------------- 03 Data cleaning
elif page == "Data Cleaning":
    q1, q3 = 5621, 10000
    section_header(
        "03 / Data cleaning",
        "Four rules take 60 answers down to 39",
        f"The rules were fixed before any correlation was computed. The outlier limit comes from the IQR rule: "
        f"Q₃ + 1.5·IQR = {q3:,} + 1.5 · {q3 - q1:,} = {q3 + 1.5 * (q3 - q1):,.0f} steps.",
    )
    steps = pd.DataFrame(
        {
            "Step": [
                "0. Answers collected",
                "1. − 2 duplicate submissions",
                "2. − 12 do not carry the phone",
                "3. − 3 impossible counts (6, 9, 2.5)",
                "4. − 4 outliers above 16 569 steps",
            ],
            "Answers left": [60, 58, 46, 43, 39],
        }
    )
    funnel = (
        alt.Chart(steps)
        .mark_bar(color=BLUE, cornerRadiusEnd=4)
        .encode(
            y=alt.Y("Step:N", sort=None, title=None, axis=alt.Axis(labelLimit=320)),
            x=alt.X("Answers left:Q", scale=alt.Scale(domain=[0, 60])),
            tooltip=["Step", "Answers left"],
        )
        .properties(height=260)
    )
    labels = funnel.mark_text(align="left", dx=6, fontWeight="bold").encode(text="Answers left:Q")
    st.altair_chart(funnel + labels, use_container_width=True)

    excluded = df_all[df_all["Group"] != "ok"].copy()
    excluded["Reason"] = excluded["Group"].map(REASONS)
    st.subheader("Excluded answers", anchor=False)
    st.dataframe(
        excluded[["ID", "Steps", "CarriesPhone", "Sleep", "Stress", "GPA", "Year", "Reason"]],
        hide_index=True,
        use_container_width=True,
    )
    st.info("Extreme but possible answers (sleep 12 h, stress 1, GPA 3.95) were kept.")

# ---------------------------------------------------------------- 04 Frequency distribution
elif page == "Frequency Distribution":
    section_header(
        "04 / Frequency distribution",
        "Most students walk 6 000 – 8 000 steps a day",
        "Histograms of the 39 analysed answers. Both distributions have one peak in the middle; "
        "steps have a short tail to the right.",
    )
    a, b = st.columns(2)
    a.altair_chart(histogram(clean, "Steps", 2000, "Steps per day (bins of 2 000)", BLUE), use_container_width=True)
    b.altair_chart(histogram(clean, "GPA", 0.25, "GPA (bins of 0.25)", ORANGE), use_container_width=True)

    bins = list(range(2000, 16001, 2000))
    freq = pd.cut(clean["Steps"], bins=bins, right=False).value_counts().sort_index()
    table = pd.DataFrame(
        {
            "Steps class": [f"{int(i.left):,} – {int(i.right) - 1:,}" for i in freq.index],
            "Frequency f": freq.values,
        }
    )
    table["Relative f"] = (table["Frequency f"] / table["Frequency f"].sum()).round(3)
    table["Cumulative f"] = table["Frequency f"].cumsum()
    st.dataframe(table, hide_index=True, use_container_width=True)

# ---------------------------------------------------------------- 05 Descriptive statistics
elif page == "Descriptive Statistics":
    section_header("05 / Summary", "Descriptive statistics", "Computed on the 39 analysed answers.")
    rows = []
    for name, col in [("Steps per day", "Steps"), ("Sleep (hours)", "Sleep"), ("Stress (1–10)", "Stress"), ("GPA", "GPA")]:
        s = clean[col]
        rows.append(
            {
                "Variable": name,
                "n": s.count(),
                "Mean": round(s.mean(), 3),
                "Median": round(s.median(), 3),
                "SD": round(s.std(ddof=1), 3),
                "Min": s.min(),
                "Max": s.max(),
            }
        )
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    g = clean["GPA"]
    n = len(g)
    t_crit = stats.t.ppf(1 - ALPHA / 2, n - 1)
    margin = t_crit * g.std(ddof=1) / np.sqrt(n)
    c1, c2, c3 = st.columns(3)
    c1.metric("Typical steps per day", f"{clean['Steps'].median():,.0f}")
    c2.metric("Mean GPA", f"{g.mean():.2f}")
    c3.metric("95% CI for mean GPA", f"[{g.mean() - margin:.2f} ; {g.mean() + margin:.2f}]")
    st.latex(r"\bar{x} \pm t_{0.025,\,n-1}\cdot\frac{s}{\sqrt{n}}")
    st.caption(f"t₀.₀₂₅,{n - 1} = {t_crit:.3f}")

# ---------------------------------------------------------------- 06 Correlation & regression
elif page == "Correlation & Regression":
    a_coef, b_coef, r2 = regression(clean["Steps"], clean["GPA"])
    section_header(
        "06 / Correlation & regression",
        "More steps go with a slightly higher GPA",
        "Each dot is one student; the dashed line is the least-squares regression line.",
    )
    left, right = st.columns([1.6, 1])
    with left:
        st.altair_chart(scatter_chart(clean), use_container_width=True)
    with right:
        st.metric("Pearson r", f"{r_main:.3f}")
        st.metric("p-value", f"{p_main:.3f}", help=f"t = {t_main:.3f}, df = {n_main - 2}")
        st.latex(rf"\widehat{{GPA}} = {a_coef:.3f} + {b_coef * 1000:.4f}\cdot\frac{{steps}}{{1000}}")
        st.write(f"R² = **{r2 * 100:.1f}%** — steps explain only this share of the differences in GPA.")
        steps_in = st.slider("Predict GPA for steps per day", 2000, 15000, 7000, 500)
        st.metric("Predicted GPA", f"{a_coef + b_coef * steps_in:.2f}")

# ---------------------------------------------------------------- 07 Hypothesis testing
elif page == "Hypothesis Testing":
    section_header(
        "07 / Hypothesis testing",
        "Only steps and GPA pass the 5% threshold",
        "For each pair: t = r·√(n − 2) / √(1 − r²), df = n − 2 = 37. Reject H₀ when p < 0.05.",
    )
    st.latex(r"t = \frac{r\sqrt{n-2}}{\sqrt{1-r^2}}")
    pairs = [("Steps – GPA", "Steps", "GPA"), ("Steps – Stress", "Steps", "Stress"), ("Stress – GPA", "Stress", "GPA"),
             ("Sleep – GPA", "Sleep", "GPA"), ("Steps – Sleep", "Steps", "Sleep")]
    out = []
    for label, x, y in pairs:
        r, t, p, n = corr_test(clean[x], clean[y])
        decision = "Reject H₀" if p < ALPHA else ("Keep H₀ (borderline)" if p < 0.1 else "Keep H₀")
        out.append({"Pair": label, "r": round(r, 3), "t": round(t, 3), "p-value": round(p, 3), "Decision": decision})
    st.dataframe(pd.DataFrame(out), hide_index=True, use_container_width=True)
    st.success(
        f"Steps and GPA: r = {r_main:.3f}, t = {t_main:.3f}, p = {p_main:.3f} < 0.05, so we reject H₀. "
        "The correlation is positive but weak."
    )

# ---------------------------------------------------------------- 08 Robustness lab
elif page == "Robustness Lab":
    section_header(
        "08 / Robustness lab",
        "Change the analysis and watch the result move",
        "A reliable result should not depend on one student or one cleaning decision. Change the options; "
        "the scatter, r and p update for steps and GPA.",
    )
    left, right = st.columns([1, 1.6])
    with left:
        keep_out = st.toggle("Keep the 4 step outliers (25 000 – 40 000)")
        keep_phone = st.toggle("Keep the 11 students who do not carry their phone")
        drop_one = st.toggle("Remove the student with 15 000 steps and GPA 3.95")
        method = st.radio("Correlation method", ["Pearson", "Spearman"], horizontal=True)

        mask = df_all["Group"].eq("ok")
        if keep_out:
            mask |= df_all["Group"].eq("outlier")
        if keep_phone:
            mask |= df_all["Group"].eq("nophone") & df_all["Steps"].between(500, 16568.5)
        data = df_all[mask]
        if drop_one:
            data = data[data["ID"] != 15]

        r, t, p, n = corr_test(data["Steps"], data["GPA"], method)
        sym = "r" if method == "Pearson" else "ρ"
        css = "verdict-yes" if p < ALPHA else "verdict-no"
        text = "Significant: reject H₀" if p < ALPHA else "Not significant: keep H₀"
        st.markdown(
            f'<div class="{css}">{method} {sym} = {r:.3f}, p = {p:.3f}<br><b style="font-size:22px">{text}</b>'
            f"<br><small>n = {n}, t = {t:.3f}, df = {n - 2}</small></div>",
            unsafe_allow_html=True,
        )
    with right:
        st.altair_chart(scatter_chart(data, 42000 if keep_out else 16000), use_container_width=True)
    st.warning(
        "With the main settings p = 0.038. Keeping the outliers, switching to Spearman's rank correlation, "
        "or removing a single student each push p above 0.05. The association is weak and fragile."
    )

# ---------------------------------------------------------------- 09 Power BI
elif page == "Power BI Dashboard":
    section_header(
        "09 / Power BI",
        "The same results, computed with DAX measures",
        "Cards show n, the means, r, t and the p-value as DAX measures; the scatter uses Power BI's trend line; "
        "the histogram uses a bin group of 2 000 steps.",
    )
    img = DATA_DIR / "powerbi_dashboard.png"
    if img.exists():
        st.image(str(img))
    st.code(
        "r Steps-GPA =\n"
        "VAR mx = AVERAGE(CleanData[Steps])\n"
        "VAR my = AVERAGE(CleanData[GPA])\n"
        "VAR sxy = SUMX(CleanData, (CleanData[Steps]-mx)*(CleanData[GPA]-my))\n"
        "VAR sxx = SUMX(CleanData, (CleanData[Steps]-mx)^2)\n"
        "VAR syy = SUMX(CleanData, (CleanData[GPA]-my)^2)\n"
        "RETURN DIVIDE(sxy, SQRT(sxx*syy))\n\n"
        "t-statistic = [r Steps-GPA] * SQRT([n]-2) / SQRT(1 - [r Steps-GPA]^2)\n"
        "p-value = T.DIST.2T(ABS([t-statistic]), [n]-2)",
        language="text",
    )

# ---------------------------------------------------------------- 10 Conclusion
else:
    section_header("10 / Limitations & conclusion", "What we can and cannot say")
    a, b = st.columns(2)
    a.markdown(
        '<div class="card"><h4>Limitations</h4>Small, non-random sample of students we could reach. '
        "GPA, sleep and stress are self-reported. Step counts are too low when the phone stays on a desk. "
        "Pearson's r measures only linear relationships. Correlation does not prove causation: motivated "
        "students may both walk more and study more.</div>",
        unsafe_allow_html=True,
    )
    b.markdown(
        '<div class="card"><h4>Other findings</h4>Students who walk more report less stress '
        "(r = −0.28, p = 0.085, borderline). Sleep is not related to steps (r = 0.10) or to GPA (r = 0.15), "
        "so sleep does not explain the link between steps and grades in this sample.</div>",
        unsafe_allow_html=True,
    )
    st.subheader("Conclusion", anchor=False)
    st.markdown(
        f"We **reject H₀ at α = 0.05**: daily steps have a weak positive correlation with GPA "
        f"(r = {r_main:.3f}, p = {p_main:.3f}, n = {n_main}). One thousand extra steps go with about "
        "0.056 more GPA points. However, the result disappears with outliers kept, with Spearman's test, "
        "or without one student, so **the evidence is weak**. A larger, random sample with study time as an "
        "extra variable is needed for a firm conclusion."
    )
