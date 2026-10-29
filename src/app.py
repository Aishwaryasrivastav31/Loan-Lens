import os
import re

import streamlit as st

from counterfactual import check_applicant  # loads the trained model and DiCE once
from guardrails import check_question, expand_query, log_event, verify_numbers

st.set_page_config(page_title="LoanLens", layout="wide")

# ============================== settings ==============================
RATE_MIN, RATE_MAX = 9.99, 24.0
LOAN_AMOUNT_SCALE = 1000  # the dataset stores LoanAmount in thousands of rupees
AMT_MIN, AMT_MAX, AMT_STEP = 25_000, 700 * LOAN_AMOUNT_SCALE, 1_000
TENURE_MAX = 30
MAX_FOIR = 0.5  # EMI may be at most 50% of total monthly income

GENDER = ["Male", "Female"]
YES_NO = ["Yes", "No"]
DEPENDENTS = ["0", "1", "2", "3+"]
EDUCATION = ["Graduate", "Not Graduate"]
PROPERTY_AREA = ["Urban", "Semiurban", "Rural"]
CREDIT = ["Good (loans repaid on time)", "Poor (missed payments or defaults)"]

CF_LABELS = {
    "ApplicantIncome": ("Monthly income", 1),
    "CoapplicantIncome": ("Co-applicant monthly income", 1),
    "LoanAmount": ("Loan amount", LOAN_AMOUNT_SCALE),
}

# ============================== RAG with guardrails ==============================
MIN_RELEVANCE = 0.35  # cosine relevance (0 to 1)
WEAK_DOMAIN_EXTRA = 0.05  # questions with no loan word must score a bit higher
SHOW_SCORES = (
    False  # True: print retrieval scores under each answer (for tuning MIN_RELEVANCE)
)
MAX_WORDS = 150
CUT_AT = [
    "\nQuestion:",
    "\nQ:",
    "\nInstruction",
    "\nUser:",
    "\n###",
    "<|end|>",
    "<|user|>",
    "<|assistant|>",
]
RAG_K = 4
CHUNK_CHARS = 1200  # chunks are up to 1000 chars (tables can be longer), do not cut answers in half
NOT_FOUND = "The bank policy documents (RBI / SBI / FAQ) do not cover this."

# These three must match src/rag/ingest.py exactly, or search silently breaks.
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
COLLECTION = "loanlens"
COLLECTION_METADATA = {"hnsw:space": "cosine"}  # relevance scores between 0 and 1
DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"  # override with GROQ_MODEL in secrets.toml

APP_DIR = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(APP_DIR, "..", "data", "rag", "chroma_db")


def get_secret(name, default=None):
    """Streamlit only reads .streamlit/secrets.toml from the folder you run `streamlit run` in."""
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    path = os.path.join(APP_DIR, ".streamlit", "secrets.toml")
    if os.path.exists(path):
        try:
            import tomllib  # Python 3.11+

            with open(path, "rb") as f:
                value = tomllib.load(f).get(name)
            if value:
                return value
        except ImportError:
            pass
    return os.getenv(name, default)


@st.cache_resource(show_spinner="Loading the policy database...")
def load_rag():
    """Opens the Chroma DB and the Groq LLM once for the whole app."""
    from langchain_chroma import Chroma
    from langchain_community.embeddings import FastEmbedEmbeddings
    from langchain_groq import ChatGroq

    if not os.path.exists(DB_DIR):
        raise FileNotFoundError(
            f"Policy database not found ({os.path.abspath(DB_DIR)}). Run: python -m src.rag.ingest"
        )
    db = Chroma(
        persist_directory=DB_DIR,
        collection_name=COLLECTION,
        embedding_function=FastEmbedEmbeddings(model_name=EMBED_MODEL),
        collection_metadata=COLLECTION_METADATA,
    )
    if db._collection.count() == 0:
        raise RuntimeError(
            "The policy database is empty. Run: python -m src.rag.ingest"
        )
    api_key = get_secret("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            'GROQ_API_KEY not found. Add  GROQ_API_KEY = "gsk_..."  to src/.streamlit/secrets.toml '
            "or set the GROQ_API_KEY environment variable."
        )
    model = get_secret("GROQ_MODEL", DEFAULT_GROQ_MODEL)
    if model.startswith("openai/gpt-oss"):
        llm = ChatGroq(
            model=model,
            temperature=0,
            max_tokens=1000,
            reasoning_effort="low",
            api_key=api_key,
        )
    else:
        llm = ChatGroq(model=model, temperature=0, max_tokens=300, api_key=api_key)
    return db, llm


def _limit_words(text: str, n: int) -> str:
    words = list(re.finditer(r"\S+", text))
    return text if len(words) <= n else text[: words[n - 1].end()]


def ask_rag(
    question: str, facts: str = "", question_number: int = 1, trusted: bool = False
):
    """Streams an answer for st.write_stream.
    Steps: input guard -> retrieval -> LLM -> number check -> sources.
    trusted=True is for questions written by the app itself (skips the input guard)."""

    # 1. Input guardrail
    weak = False
    if trusted:
        clean_q = question
    else:
        verdict = check_question(question, question_number)
        if not verdict.ok:
            log_event(
                verdict.flags[0] if verdict.flags else "blocked", question=question
            )
            yield verdict.message
            return
        clean_q = verdict.text
        weak = "weak_domain" in verdict.flags
        pii = [f for f in verdict.flags if f.startswith("pii:")]
        if pii:
            log_event("pii_removed", detail=",".join(pii), question=question)

    cache = st.session_state.setdefault("rag_cache", {})
    key = clean_q + "|" + facts
    if key in cache:
        yield cache[key]
        return

    db, llm = load_rag()

    # 2. Retrieval (short terms like "NPA" / "LTV Ratio" are expanded so they score higher)
    search_q = expand_query(clean_q)
    threshold = MIN_RELEVANCE + (WEAK_DOMAIN_EXTRA if weak else 0)
    hits = db.similarity_search_with_relevance_scores(search_q, k=RAG_K)
    chunks = [doc for doc, score in hits if score >= threshold]

    if not chunks:
        best = f"best={hits[0][1]:.2f}" if hits else "no hits"
        log_event(
            "no_context", detail=f"{best} threshold={threshold:.2f}", question=question
        )
        yield NOT_FOUND
        return

    # 3. Prompt
    context = "\n\n".join(
        f"[{os.path.basename(c.metadata['source'])}, page {c.metadata['page']}]\n"
        f"{c.page_content[:CHUNK_CHARS]}"
        for c in chunks
    )
    prompt = (
        "You are a bank loan policy assistant. Rules:\n"
        "- Answer ONLY from the policy extracts below. Do not use outside knowledge.\n"
        "- If the extracts do not contain the answer, reply exactly: NOT_IN_POLICY\n"
        "- Answer only what is asked, in at most 5 short sentences or bullet points.\n"
        "- Do not add new questions, examples, notes or disclaimers.\n\n"
        f"Policy extracts:\n{context}\n\n"
        + (
            f"Applicant facts (already calculated, they are correct):\n{facts}\n\n"
            if facts
            else ""
        )
        + f"Question: {search_q}\nAnswer:"
    )

    # 4. Streaming output (holds back the last few characters so a stop marker is never shown)
    hold = 20
    full, sent = "", 0
    for piece in llm.stream(prompt):
        full += piece.content or ""
        cut = min([full.find(m) for m in CUT_AT if m in full] + [len(full)])
        if "NOT_IN_POLICY" in full[:cut]:
            break
        if cut < len(full) or len(full[:cut].split()) > MAX_WORDS:
            break
        safe = len(full) - hold
        if safe > sent:
            yield full[sent:safe]
            sent = safe

    cut = min([full.find(m) for m in CUT_AT if m in full] + [len(full)])
    text = _limit_words(full[:cut], MAX_WORDS).rstrip()
    if "NOT_IN_POLICY" in text or not text.strip():
        log_event("not_in_policy", question=question)
        yield NOT_FOUND
        return
    if len(text) > sent:
        yield text[sent:]

    # 5. Output guardrail: every number in the answer must appear in the extracts / facts / question
    extra = ""
    bad = verify_numbers(text, context, facts, clean_q)
    if bad:
        log_event("unverified_numbers", detail=", ".join(bad), question=question)
        extra = "\n\n⚠️ Please check these figures in the source pages: " + ", ".join(
            bad
        )

    sources = sorted(
        {
            f"{os.path.basename(c.metadata['source'])} p.{c.metadata['page']}"
            for c in chunks
        }
    )
    tail = extra + "\n\nSources: " + ", ".join(sources)
    if SHOW_SCORES:
        tail += "\n\nScores: " + ", ".join(f"{s:.2f}" for _, s in hits)
    yield tail
    cache[key] = text + tail


# ============================== helpers ==============================
def inr(n) -> str:
    """Indian digit grouping: 6222485 -> 62,22,485"""
    s = str(int(round(n)))
    if len(s) <= 3:
        return s
    return re.sub(r"(\d)(?=(\d\d)+$)", r"\1,", s[:-3]) + "," + s[-3:]


def calc_emi(principal, annual_rate, years):
    r = annual_rate / 12 / 100
    n = int(years * 12)
    if r == 0:
        return principal / n
    f = (1 + r) ** n
    return principal * r * f / (f - 1)


# ============================== styling ==============================
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, .stApp, [class*="st-"] { font-family: 'Inter', sans-serif; }
.stApp { background: #EEF2F7; color: #1D2939; }
header[data-testid="stHeader"], footer, #MainMenu { display: none; }
.block-container { max-width: 1500px; padding: 1rem 1.6rem 0.6rem 1.6rem; }
div[data-testid="stVerticalBlock"] { gap: .55rem; }
[data-testid="stWidgetLabel"] p { font-size: .8rem !important; font-weight: 500; color: #475467 !important; }
.stApp p, .stApp li { color: #1D2939; }

/* top bar */
.topbar { display: flex; align-items: center; justify-content: space-between; margin-bottom: .2rem; }
.logo { font-size: 1.45rem; font-weight: 700; color: #0B2545; letter-spacing: -.02em; }
.logo span { color: #2E6BE6; }
.tagline { color: #5B6B82; font-size: .9rem; }
.stTabs [data-baseweb="tab-list"] { gap: .4rem; }
.stTabs [data-baseweb="tab"] { height: 2.3rem; padding: 0 1rem; border-radius: 8px 8px 0 0; font-weight: 600; }

/* panels */
.st-key-panel_inputs, .st-key-panel_decision, .st-key-panel_path {
    background: #FFFFFF; border: 1px solid #E3E8F0; border-radius: 16px; padding: 1.1rem 1.2rem 1.2rem 1.2rem;
    box-shadow: 0 1px 2px rgba(16,24,40,.04), 0 6px 20px rgba(16,24,40,.06);
}
.ptitle { font-size: .72rem; font-weight: 700; letter-spacing: .09em; text-transform: uppercase; color: #667085; margin: 0 0 .35rem 0; }
.sub { font-size: .72rem; font-weight: 700; letter-spacing: .09em; text-transform: uppercase; color: #667085;
       margin: .55rem 0 .1rem 0; padding-top: .55rem; border-top: 1px solid #EEF1F6; }

/* EMI hero */
.hero { background: linear-gradient(135deg, #0B2545 0%, #13315C 60%, #1D4E89 100%); border-radius: 14px;
        padding: 1.05rem 1.2rem; margin-bottom: .7rem; }
.hero .k { color: #B9C9E2; font-size: .78rem; font-weight: 500; letter-spacing: .04em; text-transform: uppercase; }
.hero .v { color: #FFFFFF; font-size: 2.05rem; font-weight: 700; letter-spacing: -.02em; line-height: 1.25; }
.hero .s { color: #D6E0EE; font-size: .82rem; }
.stats { display: grid; grid-template-columns: 1fr 1fr; gap: .6rem; margin-bottom: .9rem; }
.stats > div { background: #F7F9FC; border: 1px solid #EEF1F6; border-radius: 12px; padding: .6rem .8rem; }
.stats .k { font-size: .74rem; color: #667085; }
.stats .v { font-size: 1.02rem; font-weight: 600; color: #0B2545; }

/* decision */
.status { border-radius: 14px; padding: .95rem 1.1rem; }
.status.ok { background: #ECFDF3; border: 1px solid #ABEFC6; }
.status.no { background: #FEF3F2; border: 1px solid #FECDCA; }
.status.idle { background: #F7F9FC; border: 1px dashed #D0D5DD; text-align: center; padding: 1.6rem 1rem; }
.status .tag { display: inline-block; font-size: .68rem; font-weight: 700; letter-spacing: .09em; text-transform: uppercase;
               padding: .2rem .55rem; border-radius: 999px; margin-bottom: .45rem; }
.status.ok .tag { background: #D1FADF; color: #05603A; }
.status.no .tag { background: #FEE4E2; color: #912018; }
.status h2 { font-size: 1.25rem; font-weight: 700; margin: 0 0 .25rem 0; }
.status.ok h2 { color: #05603A; }
.status.no h2 { color: #912018; }
.status.idle h2 { color: #0B2545; font-size: 1.05rem; }
.status p { margin: 0; color: #344054 !important; font-size: .88rem; line-height: 1.45; }
.prob { margin-top: .75rem; }
.prob .row { display: flex; justify-content: space-between; font-size: .78rem; color: #475467; margin-bottom: .25rem; }
.prob .bar { height: 8px; background: #EAECF0; border-radius: 999px; overflow: hidden; }
.prob .fill { height: 100%; border-radius: 999px; }
.stale { background: #FFFAEB; border: 1px solid #FEDF89; color: #93370D; border-radius: 10px; padding: .45rem .7rem;
         font-size: .8rem; margin-top: .6rem; }
.note { color: #98A2B3; font-size: .72rem; margin-top: .6rem; }

/* path to approval */
.cf { border: 1px solid #E3E8F0; border-left: 4px solid #2E6BE6; border-radius: 10px; padding: .7rem .9rem;
      margin-bottom: .55rem; background: #FBFCFE; }
.cf .h { font-size: .7rem; font-weight: 700; letter-spacing: .09em; text-transform: uppercase; color: #2E6BE6; margin-bottom: .3rem; }
.cf .change { color: #1D2939; font-size: .9rem; margin: .1rem 0; }
.cf .meta { color: #667085; font-size: .8rem; margin-top: .35rem; }
.empty { color: #667085; font-size: .88rem; line-height: 1.5; padding: .4rem 0; }

/* buttons */
.st-key-check_btn button { background: #2E6BE6; color: #fff; border: 0; border-radius: 10px; font-weight: 600; height: 2.7rem; }
.st-key-check_btn button:hover { background: #1F55C4; color: #fff; }
.st-key-check_btn button p { color: #fff !important; }
</style>
""",
    unsafe_allow_html=True,
)

# ============================== state ==============================
st.session_state.setdefault("amt_s", 1_50_000)
st.session_state.setdefault("amt_i", 1_50_000)
st.session_state.setdefault("result", None)
st.session_state.setdefault("explain", None)
st.session_state.setdefault("chat", [])


def _slider_to_input():
    st.session_state.amt_i = st.session_state.amt_s


def _input_to_slider():
    v = int(round(st.session_state.amt_i / AMT_STEP) * AMT_STEP)
    v = min(max(v, AMT_MIN), AMT_MAX)
    st.session_state.amt_i = v
    st.session_state.amt_s = v


# ============================== top bar ==============================
st.markdown(
    '<div class="topbar"><div class="logo">Loan<span>Lens</span></div>'
    '<div class="tagline">Explainable loan eligibility, grounded in RBI and SBI policy</div></div>',
    unsafe_allow_html=True,
)
tab_check, tab_policy = st.tabs(["Eligibility check", "Policy assistant"])

PANEL_HEIGHT = 560

with tab_check:
    col_in, col_dec, col_path = st.columns([1.2, 1, 1.15], gap="medium")

    # ------------------------------ panel 1: inputs ------------------------------
    with col_in:
        with st.container(key="panel_inputs"):
            st.markdown('<p class="ptitle">Loan</p>', unsafe_allow_html=True)
            a1, a2 = st.columns([1.6, 1])
            amount = a1.slider(
                "Loan amount",
                AMT_MIN,
                AMT_MAX,
                step=AMT_STEP,
                key="amt_s",
                on_change=_slider_to_input,
                format="₹%d",
            )
            a2.number_input(
                "Amount (₹)",
                min_value=AMT_MIN,
                max_value=AMT_MAX,
                step=AMT_STEP,
                key="amt_i",
                on_change=_input_to_slider,
            )
            t1, t2 = st.columns(2)
            years = t1.slider("Tenure", 1, TENURE_MAX, TENURE_MAX, format="%d years")
            rate = t2.slider(
                "Interest rate", RATE_MIN, RATE_MAX, 9.99, step=0.01, format="%.2f%%"
            )

            st.markdown('<p class="sub">Applicant</p>', unsafe_allow_html=True)
            g1, g2, g3 = st.columns(3)
            gender = g1.selectbox("Gender", GENDER)
            married = g2.selectbox("Married", YES_NO)
            dependents = g3.selectbox("Dependents", DEPENDENTS)
            education = g1.selectbox("Education", EDUCATION)
            self_emp = g2.selectbox("Self employed", YES_NO[::-1])
            area = g3.selectbox("Property area", PROPERTY_AREA)
            i1, i2 = st.columns(2)
            income = i1.number_input(
                "Monthly income (₹)", min_value=0, value=60_000, step=1_000
            )
            co_income = i2.number_input(
                "Co-applicant income (₹)", min_value=0, value=0, step=1_000
            )
            credit = st.selectbox("Credit history", CREDIT)
            check = st.button(
                "Check eligibility", key="check_btn", use_container_width=True
            )

    applicant = {
        "Gender": gender,
        "Married": married,
        "Dependents": dependents,
        "Education": education,
        "Self_Employed": self_emp,
        "ApplicantIncome": float(income),
        "CoapplicantIncome": float(co_income),
        "LoanAmount": float(amount) / LOAN_AMOUNT_SCALE,
        "Loan_Amount_Term": float(years * 12),
        "Credit_History": 1.0 if credit.startswith("Good") else 0.0,
        "Property_Area": area,
    }
    emi = calc_emi(amount, rate, years)
    payable = emi * years * 12
    interest = payable - amount

    if check:
        with col_dec:
            with st.spinner("Assessing the application..."):
                result = check_applicant(applicant)
        total_income = income + co_income
        result.update(
            emi=emi,
            years=years,
            rate=rate,
            amount=amount,
            income=income,
            co_income=co_income,
            credit_good=credit.startswith("Good"),
            applicant=dict(applicant),
            foir=emi / total_income if total_income else float("inf"),
        )
        # affordability: even if the model approves, the EMI must fit the income
        if MAX_FOIR and result["status"] == "approved" and result["foir"] > MAX_FOIR:
            r, n = rate / 12 / 100, years * 12
            f = (1 + r) ** n
            result["status"] = "emi_too_high"
            result["max_loan"] = MAX_FOIR * total_income * (f - 1) / (r * f)
        st.session_state.result = result
        st.session_state.explain = None
    res = st.session_state.result

    # ------------------------------ panel 2: repayment + decision ------------------------------
    with col_dec:
        with st.container(key="panel_decision"):
            st.markdown('<p class="ptitle">Repayment</p>', unsafe_allow_html=True)
            st.markdown(
                f"""
<div class="hero"><div class="k">Monthly EMI</div><div class="v">₹{inr(emi)}</div>
<div class="s">₹{inr(amount)} over {years} years at {rate:.2f}% p.a.</div></div>
<div class="stats">
  <div><div class="k">Total interest</div><div class="v">₹{inr(interest)}</div></div>
  <div><div class="k">Total payable</div><div class="v">₹{inr(payable)}</div></div>
</div>""",
                unsafe_allow_html=True,
            )

            st.markdown('<p class="ptitle">Decision</p>', unsafe_allow_html=True)
            if not res:
                st.markdown(
                    '<div class="status idle"><h2>No assessment yet</h2>'
                    "<p>Enter the details and select <b>Check eligibility</b>.</p></div>",
                    unsafe_allow_html=True,
                )
            else:
                prob = res["approval_probability"] * 100
                if res["status"] == "approved":
                    cls, tag, head = "ok", "Approved", "Loan approved"
                    text = f"The estimated EMI of ₹{inr(res['emi'])} is {res['foir'] * 100:.0f}% of monthly income."
                elif res["status"] == "emi_too_high":
                    cls, tag, head = "no", "Declined", "EMI too high"
                    text = (
                        f"The EMI is {res['foir'] * 100:.0f}% of monthly income, above the {MAX_FOIR * 100:.0f}% "
                        f"limit. A loan of up to ₹{inr(res['max_loan'])} would be affordable."
                    )
                else:
                    cls, tag, head = "no", "Declined", "Loan not approved"
                    text = "See the path to approval for changes that would lead to an approval."
                colour = "#12B76A" if cls == "ok" else "#F04438"
                stale = (
                    (
                        '<div class="stale">The details have changed since this check. '
                        "Select Check eligibility again to update it.</div>"
                    )
                    if res.get("applicant") != applicant
                    else ""
                )
                st.markdown(
                    f"""
<div class="status {cls}"><span class="tag">{tag}</span><h2>{head}</h2><p>{text}</p>
<div class="prob"><div class="row"><span>Approval probability</span><b>{prob:.0f}%</b></div>
<div class="bar"><div class="fill" style="width:{prob:.0f}%; background:{colour};"></div></div></div></div>
{stale}""",
                    unsafe_allow_html=True,
                )
            st.markdown(
                '<p class="note">Model estimate based on historical loan data. Not a lending decision or '
                "financial advice.</p>",
                unsafe_allow_html=True,
            )

    # ------------------------------ panel 3: path to approval + policy context ------------------------------
    with col_path:
        with st.container(key="panel_path"):
            st.markdown(
                '<p class="ptitle">Path to approval</p>', unsafe_allow_html=True
            )
            with st.container(height=PANEL_HEIGHT, border=False):
                if not res:
                    st.markdown(
                        '<div class="empty">If an application is declined, this panel shows the smallest '
                        "realistic changes that would lead to approval, and the bank policy rules that apply.</div>",
                        unsafe_allow_html=True,
                    )
                elif res["status"] == "approved":
                    st.markdown(
                        '<div class="empty">No changes are needed. The application meets the model\'s '
                        "criteria and the EMI is within the affordability limit.</div>",
                        unsafe_allow_html=True,
                    )
                elif res["status"] == "emi_too_high":
                    st.markdown(
                        f'<div class="cf"><div class="h">Option 1</div><div class="change">Reduce the loan '
                        f"amount to <b>₹{inr(res['max_loan'])}</b> or less</div></div>"
                        f'<div class="cf"><div class="h">Option 2</div><div class="change">Choose a longer '
                        f"tenure to lower the monthly EMI</div></div>",
                        unsafe_allow_html=True,
                    )
                else:
                    cfs = res.get("counterfactuals") or []
                    if not cfs:
                        st.markdown(
                            '<div class="empty">No realistic change to income or loan amount would lead to '
                            "approval for this profile. The main factor is likely to be the credit history, "
                            "which improves over time with regular, on-time repayments.</div>",
                            unsafe_allow_html=True,
                        )
                    for i, cf in enumerate(cfs, 1):
                        lines = []
                        for feat, v in cf.items():
                            name, mult = CF_LABELS.get(feat, (feat, 1))
                            a, b = v["from"] * mult, v["to"] * mult
                            verb = "Increase" if b > a else "Reduce"
                            lines.append(
                                f'<div class="change">{verb} {name.lower()} from ₹{inr(a)} to <b>₹{inr(b)}</b></div>'
                            )
                        new_loan = (
                            cf["LoanAmount"]["to"] * LOAN_AMOUNT_SCALE
                            if "LoanAmount" in cf
                            else res["amount"]
                        )
                        new_inc = (
                            cf["ApplicantIncome"]["to"]
                            if "ApplicantIncome" in cf
                            else res["income"]
                        ) + (
                            cf["CoapplicantIncome"]["to"]
                            if "CoapplicantIncome" in cf
                            else res["co_income"]
                        )
                        new_emi = calc_emi(new_loan, res["rate"], res["years"])
                        meta = (
                            (
                                f'<div class="meta">New EMI ₹{inr(new_emi)}, {new_emi / new_inc * 100:.0f}% of monthly '
                                f"income</div>"
                            )
                            if new_inc
                            else ""
                        )
                        st.markdown(
                            f'<div class="cf"><div class="h">Option {i}</div>{"".join(lines)}{meta}</div>',
                            unsafe_allow_html=True,
                        )

                    st.markdown(
                        '<p class="sub">Policy context</p>', unsafe_allow_html=True
                    )
                    if st.session_state.explain is None:
                        prob = res["approval_probability"] * 100
                        q = (
                            "What do the loan policies say about the maximum EMI to income ratio (FOIR), loan "
                            "eligibility by income, maximum loan amount and credit history or credit score? In at "
                            "most 4 bullet points, say which of these rules matter for this applicant and what they "
                            "require. Do not list, repeat or judge the suggested options."
                        )
                        facts = (
                            f"- Monthly income: ₹{inr(res['income'])}, co-applicant income: ₹{inr(res['co_income'])}\n"
                            f"- Loan requested: ₹{inr(res['amount'])} for {res['years']} years at {res['rate']:.2f}%\n"
                            f"- EMI: ₹{inr(res['emi'])} = {res['foir'] * 100:.0f}% of total monthly income\n"
                            f"- Credit history: {'good' if res['credit_good'] else 'poor / defaults'}\n"
                            f"- Model decision: declined (approval probability {prob:.0f}%)"
                        )
                        try:
                            # trusted=True: this question is written by the app, not typed by the user
                            st.session_state.explain = st.write_stream(
                                ask_rag(q, facts, trusted=True)
                            )
                        except Exception as e:
                            st.session_state.explain = (
                                f"The policy explanation is unavailable: {e}"
                            )
                            st.write(st.session_state.explain)
                    else:
                        st.write(st.session_state.explain)

# ============================== policy assistant tab ==============================
with tab_policy:
    st.caption(
        "Answers come only from the RBI and SBI policy documents and FAQ, with the source pages cited."
    )

    question = st.chat_input("Ask a question about loan policy", key="policy_q")
    with st.container(height=PANEL_HEIGHT, border=True):
        if not st.session_state.chat and not question:
            st.markdown(
                '<div class="empty">Try: "What is the maximum loan-to-value ratio for a home loan up to '
                '₹20 lakh?", "What is EBLR?", "What is LTV?" or "What documents are needed?"</div>',
                unsafe_allow_html=True,
            )
        for role, text in st.session_state.chat:
            with st.chat_message(role):
                st.write(text)
        if question:
            st.session_state.chat.append(("user", question))
            q_number = sum(1 for role, _ in st.session_state.chat if role == "user")
            with st.chat_message("user"):
                st.write(question)
            with st.chat_message("assistant"):
                try:
                    answer = st.write_stream(
                        ask_rag(question, question_number=q_number)
                    )
                except Exception as e:
                    answer = f"The policy assistant is unavailable: {e}"
                    st.write(answer)
            st.session_state.chat.append(("assistant", answer))
