import pandas as pd
import streamlit as st
from sklearn.compose import make_column_transformer, make_column_selector
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

st.set_page_config(page_title="Student Risk Early Warning", page_icon="🎓")

URL = "https://raw.githubusercontent.com/arunk13/MSDA-Assignments/master/IS607Fall2015/Assignment3/student-mat.csv"


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

st.title("🎓 Student Risk Early Warning")
st.write("Fill in what you know about a student after the **first term**. "
         "The tool shows whether the student may be at risk of failing maths.")
st.caption("Support tool only. Never use it to grade, punish or make final decisions about a student.")

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

if st.button("Check risk", type="primary"):
    student = dict(defaults)       # start from typical values, then overwrite with the inputs
    student.update({"G1": g1, "failures": failures, "absences": absences, "age": age,
                    "studytime": studytime, "higher": higher, "goout": goout, "Medu": medu})
    prob = model.predict_proba(pd.DataFrame([student])[columns])[0, 1]

    st.metric("Risk score", f"{prob:.0%}")
    st.progress(float(prob))
    if prob >= 0.5:
        st.error("⚠️ AT RISK: please look at this student and consider extra support.")
    else:
        st.success("✅ Not at risk right now. Keep monitoring as normal.")
    st.caption("Fields not shown here use typical values from the dataset.")
