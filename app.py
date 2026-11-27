import pandas as pd
import streamlit as st
from sklearn.compose import make_column_transformer, make_column_selector
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

st.set_page_config(page_title="Student Risk Early Warning", page_icon="🎓", layout="centered")

URL = "https://raw.githubusercontent.com/arunk13/MSDA-Assignments/master/IS607Fall2015/Assignment3/student-mat.csv"

# ---------- Styling (only looks, nothing to do with the model) ----------
st.markdown(
    """
    <style>
    #MainMenu, footer {visibility: hidden;}
    .hero {
        background: linear-gradient(135deg, #1f4e79, #3a86c8);
        padding: 1.6rem 1.8rem;
        border-radius: 14px;
        margin-bottom: 1.2rem;
    }
    .hero h1 {color: white; margin: 0; font-size: 2rem;}
    .hero p  {color: #e3eefa; margin: 0.4rem 0 0 0; font-size: 1rem;}
    .result {
        padding: 1.2rem 1.4rem;
        border-radius: 12px;
        margin-top: 1rem;
        color: #222;
    }
    .result h2 {margin: 0 0 0.3rem 0;}
    .risk   {background: #fdecea; border-left: 8px solid #d93025;}
    .safe   {background: #e8f5e9; border-left: 8px solid #2e7d32;}
    .medium {background: #fff4e0; border-left: 8px solid #ef8f00;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------- Model (same as the notebook's final model) ----------
@st.cache_resource
def train_model():
    """Load the data and train the final model (tuned Random Forest, G1 features)."""
    df = pd.read_csv(URL, sep=";")
    y = (df["G3"] < 10).astype(int)
    X = df.drop(columns=["G3", "G2"])          # no G3 (target) and no G2 (too close to the final grade)

    prep = make_column_transformer(
        (StandardScaler(), make_column_selector(dtype_include="number")),
        (OneHotEncoder(handle_unknown="ignore", drop="if_binary"), make_column_selector(dtype_exclude="number")),
    )
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_leaf=1,
                                class_weight="balanced", random_state=42)
    model = make_pipeline(prep, rf).fit(X, y)

    # typical value of every column, used for the fields the teacher does not fill in
    defaults = {c: (X[c].median() if pd.api.types.is_numeric_dtype(X[c]) else X[c].mode()[0]) for c in X.columns}
    return model, defaults, list(X.columns)


model, defaults, columns = train_model()

# ---------- Sidebar ----------
with st.sidebar:
    st.header("About")
    st.write("An early-warning tool that flags students who may fail maths, "
             "so teachers can offer support *before* the final result.")
    st.write("*Model:* tuned Random Forest using the first-period grade (G1).")
    st.write("*Data:* UCI Student Performance (395 students, 2 Portuguese schools).")
    st.divider()
    st.warning("Support tool only. Never use it to grade, punish or make final decisions about a student.")

# ---------- Header ----------
st.markdown(
    """
    <div class="hero">
        <h1>🎓 Student Risk Early Warning</h1>
        <p>Enter what you know about a student after the first term and check the risk of failing maths.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------- Inputs ----------
with st.container(border=True):
    st.subheader("Student details")
    col1, col2 = st.columns(2)
    with col1:
        g1 = st.slider("First-period grade (G1, out of 20)", 0, 20, 10)
        failures = st.selectbox("Past class failures", [0, 1, 2, 3])
        absences = st.number_input("Absences", 0, 100, 4)
        age = st.slider("Age", 15, 22, 17)
    with col2:
        studytime = st.selectbox("Weekly study time (1 = <2h, 2 = 2-5h, 3 = 5-10h, 4 = >10h)", [1, 2, 3, 4], index=1)
        higher = st.radio("Wants higher education?", ["yes", "no"], horizontal=True)
        goout = st.slider("Going out with friends (1 = low, 5 = high)", 1, 5, 3)
        medu = st.slider("Mother's education (0-4)", 0, 4, 2)

# ---------- Prediction ----------
if st.button("Check risk", type="primary", use_container_width=True):
    student = dict(defaults)       # start from typical values, then overwrite with the inputs
    student.update({"G1": g1, "failures": failures, "absences": absences, "age": age,
                    "studytime": studytime, "higher": higher, "goout": goout, "Medu": medu})
    prob = model.predict_proba(pd.DataFrame([student])[columns])[0, 1]

    if prob >= 0.5:
        css, title, msg = "risk", "⚠️ At risk", "Please look at this student and consider extra support (tutoring, a talk with parents)."
    elif prob >= 0.3:
        css, title, msg = "medium", "🟠 Keep an eye on this student", "Below the alert line, but worth watching in the next term."
    else:
        css, title, msg = "safe", "✅ Not at risk right now", "Keep monitoring as normal."

    st.markdown(
        f'<div class="result {css}"><h2>{title}</h2>'
        f'<b>Risk score: {prob:.0%}</b><br>{msg}</div>',
        unsafe_allow_html=True,
    )
    st.progress(float(prob))
    st.caption("Fields not shown here use typical values from the dataset. The alert line is 50%.")

with st.expander("How to read this result"):
    st.write(
        "- The risk score is the model's estimated chance that the student finishes below 10 out of 20.\n"
        "- On our test data the model caught about 85% of the students who really failed, "
        "but about 3 in 10 of its alerts were false alarms.\n"
        "- Treat every alert as \"please look at this student\", never as a verdict."
    )