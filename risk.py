import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

import shap


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Student At-Risk Prediction System",
    page_icon="🎓",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.main {
    background-color: #f8fafc;
}
[data-testid="stMetric"] {
    background-color: #262730;
    border-radius: 12px;
    padding: 15px;
    border: 1px solid #444444;
}

[data-testid="stMetricLabel"] {
    color: #ffffff !important;
}

[data-testid="stMetricValue"] {
    color: #ffffff !important;
}

h1 {
    font-weight: 700;
}

h2 {
    font-weight: 650;
}

h3 {
    font-weight: 600;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    data = pd.read_csv("student_data.csv")

    return data


df = load_data()


# ============================================================
# TARGET CREATION
# ============================================================

df["At_Risk"] = (df["Final_Score"] < 50).astype(int)


# ============================================================
# FEATURES
# ============================================================

features = [
    "Gender",
    "Age",
    "Department",
    "Attendance (%)",
    "Midterm_Score",
    "Assignments_Avg",
    "Quizzes_Avg",
    "Participation_Score",
    "Projects_Score",
    "test_preparation_course",
    "math_score",
    "reading_score",
    "writing_score",
    "science_score"
]

X = df[features]
y = df["At_Risk"]


categorical_features = [
    "Gender",
    "Department"
]

numerical_features = [
    "Age",
    "Attendance (%)",
    "Midterm_Score",
    "Assignments_Avg",
    "Quizzes_Avg",
    "Participation_Score",
    "Projects_Score",
    "test_preparation_course",
    "math_score",
    "reading_score",
    "writing_score",
    "science_score"
]


# ============================================================
# PREPROCESSING
# ============================================================

preprocessor = ColumnTransformer(

    transformers=[

        (
            "num",
            SimpleImputer(strategy="median"),
            numerical_features
        ),

        (
            "cat",
            Pipeline([
                (
                    "imputer",
                    SimpleImputer(strategy="most_frequent")
                ),

                (
                    "encoder",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        sparse_output=False
                    )
                )
            ]),
            categorical_features
        )
    ]
)


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ============================================================
# GRADIENT BOOSTING MODEL
# ============================================================

model = GradientBoostingClassifier(

    n_estimators=300,
    learning_rate=0.03,
    max_depth=3,
    min_samples_leaf=5,
    subsample=0.9,
    random_state=42

)


pipeline = Pipeline([

    ("preprocessor", preprocessor),

    ("model", model)

])


# ============================================================
# TRAIN MODEL
# ============================================================

@st.cache_resource
def train_model(X_train, y_train):

    trained_model = Pipeline([

        ("preprocessor", preprocessor),

        (
            "model",
            GradientBoostingClassifier(
                n_estimators=300,
                learning_rate=0.03,
                max_depth=3,
                min_samples_leaf=5,
                subsample=0.9,
                random_state=42
            )
        )

    ])

    sample_weights = np.where(
        y_train == 1,
        4.0,
        1.0
    )

    trained_model.fit(
        X_train,
        y_train,
        model__sample_weight=sample_weights
    )

    return trained_model


pipeline = train_model(
    X_train,
    y_train
)


# ============================================================
# MODEL PREDICTIONS
# ============================================================

y_pred = pipeline.predict(X_test)

y_probability = pipeline.predict_proba(X_test)[:, 1]


accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    y_probability
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🎓 Student Risk System")

page = st.sidebar.radio(

    "Navigation",

    [
        "📊 Overview",
        "🤖 Model Performance",
        "🔍 Student Risk Prediction"
    ]

)


# ============================================================
# HEADER
# ============================================================

st.title("🎓 Explainable Student At-Risk Prediction System")

st.markdown(
    """
    An AI-powered system for identifying students who may require
    early academic intervention.

    The system combines **machine learning + explainable AI (SHAP)**
    to provide both a risk prediction and the factors contributing
    to that prediction.
    """
)


# ============================================================
# OVERVIEW
# ============================================================

if page == "📊 Overview":

    st.header("📊 Student Performance Overview")

    total_students = len(df)

    at_risk_students = df["At_Risk"].sum()

    at_risk_percentage = (
        at_risk_students /
        total_students
    ) * 100

    average_attendance = df["Attendance (%)"].mean()

    average_final_score = df["Final_Score"].mean()


    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)


    col1.metric(
        "Total Students",
        f"{total_students:,}"
    )


    col2.metric(
        "At-Risk Students",
        f"{at_risk_students:,}"
    )


    col3.metric(
        "At-Risk Rate",
        f"{at_risk_percentage:.1f}%"
    )


    col4.metric(
        "Average Attendance",
        f"{average_attendance:.1f}%"
    )


    st.divider()


    # --------------------------------------------------------
    # AT-RISK DISTRIBUTION
    # --------------------------------------------------------

    col1, col2 = st.columns(2)


    with col1:

        st.subheader("At-Risk Distribution")

        risk_counts = pd.DataFrame({

            "Status": [
                "Not At Risk",
                "At Risk"
            ],

            "Students": [
                (df["At_Risk"] == 0).sum(),
                (df["At_Risk"] == 1).sum()
            ]

        })


        fig = px.pie(

            risk_counts,

            names="Status",

            values="Students",

            hole=0.45,

            title="Student Risk Distribution"

        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


    # --------------------------------------------------------
    # GRADE DISTRIBUTION
    # --------------------------------------------------------

    with col2:

        st.subheader("Grade Distribution")

        grade_counts = (
            df["Grade"]
            .value_counts()
            .reset_index()
        )

        grade_counts.columns = [
            "Grade",
            "Students"
        ]


        fig = px.bar(

            grade_counts,

            x="Grade",

            y="Students",

            title="Grade Distribution"

        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


    # --------------------------------------------------------
    # ATTENDANCE
    # --------------------------------------------------------

    st.subheader("Attendance Distribution")

    fig = px.histogram(

        df,

        x="Attendance (%)",

        nbins=30,

        title="Student Attendance"

    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


    # --------------------------------------------------------
    # FINAL SCORE
    # --------------------------------------------------------

    st.subheader("Final Score Distribution")

    fig = px.histogram(

        df,

        x="Final_Score",

        nbins=30,

        title="Final Score Distribution"

    )

    fig.add_vline(
        x=50,
        line_dash="dash",
        annotation_text="At-Risk Threshold"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


    # --------------------------------------------------------
    # DATA QUALITY
    # --------------------------------------------------------

    st.subheader("🔎 Data Quality")

    q1, q2, q3 = st.columns(3)


    q1.metric(
        "Records",
        f"{len(df):,}"
    )


    q2.metric(
        "Features",
        "21"
    )


    q3.metric(
        "Missing Values",
        f"{df.isna().sum().sum():,}"
    )


    st.info(
        """
        The current dataset contains academic performance and
        attendance variables. It does not contain actual LMS
        activity or longitudinal attendance records, so those
        variables are not fabricated in this dashboard.
        """
    )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

elif page == "🤖 Model Performance":

    st.header("🤖 Model Performance")

    st.markdown(
        """
        The system uses **Gradient Boosting** as the deployed
        prediction model. It is suitable for capturing nonlinear
        relationships between academic and attendance variables.
        """
    )


    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    col1, col2, col3, col4, col5 = st.columns(5)


    col1.metric(
        "Accuracy",
        f"{accuracy:.3f}"
    )

    col2.metric(
        "Precision",
        f"{precision:.3f}"
    )

    col3.metric(
        "Recall",
        f"{recall:.3f}"
    )

    col4.metric(
        "F1 Score",
        f"{f1:.3f}"
    )

    col5.metric(
        "ROC-AUC",
        f"{roc_auc:.3f}"
    )


    st.divider()


    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    st.subheader("Confusion Matrix")

    cm = confusion_matrix(
        y_test,
        y_pred
    )


    cm_df = pd.DataFrame(

        cm,

        index=[
            "Actual: Not At Risk",
            "Actual: At Risk"
        ],

        columns=[
            "Predicted: Not At Risk",
            "Predicted: At Risk"
        ]

    )


    st.dataframe(
        cm_df,
        use_container_width=True
    )


    st.markdown(
        """
        **Why recall matters:**  
        A false negative means a student who is actually at risk
        was not flagged. In an intervention system, missing such
        students can be more serious than generating additional
        false alarms.
        """
    )


    st.divider()


    # --------------------------------------------------------
    # MODEL COMPARISON
    # --------------------------------------------------------

    st.subheader("Model Comparison")

    comparison = pd.DataFrame({

        "Model": [
            "Logistic Regression",
            "Random Forest",
            "Gradient Boosting",
            "Neural Network"
        ],

        "Accuracy": [
            0.5225,
            0.8250,
            0.8225,
            0.8300
        ],

        "Precision": [
            0.2077,
            0.0000,
            0.0000,
            0.6667
        ],

        "Recall": [
            0.6143,
            0.0000,
            0.0000,
            0.0571
        ],

        "F1": [
            0.3105,
            0.0000,
            0.0000,
            0.1053
        ],

        "ROC-AUC": [
            0.5693,
            0.4871,
            0.4949,
            0.5778
        ]

    })


    st.dataframe(
        comparison,
        use_container_width=True
    )


    st.info(
        """
        Gradient Boosting is used as the deployed model for the
        explainable prediction workflow.
        """
    )


# ============================================================
# STUDENT RISK PREDICTION
# ============================================================

elif page == "🔍 Student Risk Prediction":

    st.header("🔍 Student Risk Prediction")

    st.markdown(
        """
        Enter a student's academic information to generate a
        personalized risk prediction.
        """
    )


    # --------------------------------------------------------
    # INPUTS
    # --------------------------------------------------------

    col1, col2 = st.columns(2)


    with col1:

        gender = st.selectbox(

            "Gender",

            sorted(
                df["Gender"]
                .dropna()
                .unique()
                .tolist()
            )

        )


        department = st.selectbox(

            "Department",

            sorted(
                df["Department"]
                .dropna()
                .unique()
                .tolist()
            )

        )


        age = st.number_input(

            "Age",

            min_value=18,

            max_value=30,

            value=20

        )


        attendance = st.slider(

            "Attendance (%)",

            0.0,

            100.0,

            75.0

        )


        midterm = st.slider(

            "Midterm Score",

            0.0,

            100.0,

            70.0

        )


        assignments = st.slider(

            "Assignments Average",

            0.0,

            100.0,

            75.0

        )


        quizzes = st.slider(

            "Quizzes Average",

            0.0,

            100.0,

            75.0

        )


    with col2:

        participation = st.slider(

            "Participation Score",

            0.0,

            10.0,

            5.0

        )


        projects = st.slider(

            "Projects Score",

            0.0,

            100.0,

            75.0

        )


        test_prep = st.selectbox(

            "Test Preparation Course",

            sorted(
                df["test_preparation_course"]
                .dropna()
                .unique()
                .tolist()
            )

        )


        math_score = st.slider(

            "Math Score",

            0.0,

            100.0,

            60.0

        )


        reading_score = st.slider(

            "Reading Score",

            0.0,

            100.0,

            70.0

        )


        writing_score = st.slider(

            "Writing Score",

            0.0,

            100.0,

            70.0

        )


        science_score = st.slider(

            "Science Score",

            0.0,

            100.0,

            65.0

        )


    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    student = pd.DataFrame({

        "Gender": [gender],

        "Age": [age],

        "Department": [department],

        "Attendance (%)": [attendance],

        "Midterm_Score": [midterm],

        "Assignments_Avg": [assignments],

        "Quizzes_Avg": [quizzes],

        "Participation_Score": [participation],

        "Projects_Score": [projects],

        "test_preparation_course": [test_prep],

        "math_score": [math_score],

        "reading_score": [reading_score],

        "writing_score": [writing_score],

        "science_score": [science_score]

    })


    if st.button(
        "🚨 Predict Student Risk",
        type="primary",
        use_container_width=True
    ):

        probability = pipeline.predict_proba(
            student
        )[0][1]


        prediction = int(
            probability >= 0.5
        )


        st.divider()


        # ----------------------------------------------------
        # RESULT
        # ----------------------------------------------------

        if prediction == 1:

            st.error(
                f"⚠️ Student At Risk — Risk Probability: "
                f"{probability * 100:.1f}%"
            )

        else:

            st.success(
                f"✅ Student Not Currently At Risk — Risk Probability: "
                f"{probability * 100:.1f}%"
            )


        # ----------------------------------------------------
        # RISK GAUGE
        # ----------------------------------------------------

        fig = px.bar(

            x=["Risk Probability"],

            y=[probability * 100],

            range_y=[0, 100],

            labels={
                "y": "Probability (%)",
                "x": ""
            },

            title="Predicted Risk Probability"

        )


        st.plotly_chart(
            fig,
            use_container_width=True
        )


        # ----------------------------------------------------
        # SHAP EXPLANATION
        # ----------------------------------------------------

        st.subheader("🧠 Why did the model make this prediction?")


        transformed_student = pipeline.named_steps[
            "preprocessor"
        ].transform(student)


        trained_model = pipeline.named_steps[
            "model"
        ]


        explainer = shap.TreeExplainer(
            trained_model
        )


        shap_values = explainer.shap_values(
            transformed_student
        )


        if isinstance(shap_values, list):

            values = shap_values[1][0]

        else:

            values = shap_values[0]


        feature_names = (
            pipeline
            .named_steps["preprocessor"]
            .get_feature_names_out()
        )


        shap_df = pd.DataFrame({

            "Feature": feature_names,

            "SHAP Value": values

        })


        shap_df["Absolute"] = (
            shap_df["SHAP Value"]
            .abs()
        )


        shap_df = (
            shap_df
            .sort_values(
                "Absolute",
                ascending=False
            )
            .head(10)
        )


        # ----------------------------------------------------
        # SHAP CHART
        # ----------------------------------------------------

        fig = px.bar(

            shap_df.sort_values(
                "SHAP Value"
            ),

            x="SHAP Value",

            y="Feature",

            orientation="h",

            title="Top Factors Influencing Prediction"

        )


        fig.add_vline(
            x=0,
            line_dash="dash"
        )


        st.plotly_chart(
            fig,
            use_container_width=True
        )


        # ----------------------------------------------------
        # HUMAN READABLE EXPLANATION
        # ----------------------------------------------------

        positive = (
            shap_df[
                shap_df["SHAP Value"] > 0
            ]
            .sort_values(
                "SHAP Value",
                ascending=False
            )
        )


        negative = (
            shap_df[
                shap_df["SHAP Value"] < 0
            ]
            .sort_values(
                "SHAP Value"
            )
        )


        col1, col2 = st.columns(2)


        with col1:

            st.subheader(
                "🔴 Factors Increasing Risk"
            )


            if len(positive) == 0:

                st.write(
                    "No major risk-increasing factors identified."
                )

            else:

                for _, row in positive.head(5).iterrows():

                    st.write(
                        f"• {row['Feature']}"
                    )


        with col2:

            st.subheader(
                "🟢 Factors Reducing Risk"
            )


            if len(negative) == 0:

                st.write(
                    "No major protective factors identified."
                )

            else:

                for _, row in negative.head(5).iterrows():

                    st.write(
                        f"• {row['Feature']}"
                    )


        # ----------------------------------------------------
        # INTERVENTION
        # ----------------------------------------------------

        st.subheader("💡 Recommended Intervention")


        recommendations = []


        if attendance < 70:

            recommendations.append(
                "📅 Attendance is low. Consider attendance monitoring and academic counseling."
            )


        if assignments < 60:

            recommendations.append(
                "📝 Assignment performance is low. Consider assignment support and deadline planning."
            )


        if midterm < 60:

            recommendations.append(
                "📚 Midterm performance is low. Recommend targeted academic support."
            )


        if quizzes < 60:

            recommendations.append(
                "✏️ Quiz performance is low. Recommend additional revision and practice."
            )


        if participation < 4:

            recommendations.append(
                "🙋 Participation is low. Encourage classroom engagement."
            )


        if projects < 60:

            recommendations.append(
                "🧪 Project performance is low. Consider mentoring or project guidance."
            )


        if not recommendations:

            recommendations.append(
                "✅ No major intervention signal detected. Continue monitoring academic progress."
            )


        for recommendation in recommendations:

            st.info(
                recommendation
            )


        st.caption(
            """
            SHAP explains how features contributed to the model's
            prediction. These explanations describe model behavior
            and should not be interpreted as proof of causal effects.
            """
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()


