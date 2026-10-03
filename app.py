import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Career Recommendation System",
    page_icon="🎯",
    layout="wide"
)

# ---------------- CAREER DATA ----------------
career_data = {
    "Career": [
        "Data Analyst",
        "Data Scientist",
        "AI / ML Engineer",
        "Python Developer",
        "Web Developer",
        "Cloud Engineer",
        "Business Analyst"
    ],

    "Skills": [
        "Python SQL Excel Power BI Statistics",
        "Python Statistics Machine Learning SQL Pandas",
        "Python Machine Learning Artificial Intelligence Statistics",
        "Python Java OOP APIs",
        "HTML CSS JavaScript Web Development",
        "AWS Linux Networking Cloud Computing",
        "Excel SQL Power BI Communication Business Analysis"
    ],

    "Interests": [
        "Data Analytics Data Science",
        "Data Science Artificial Intelligence",
        "Artificial Intelligence Machine Learning",
        "Software Development Programming",
        "Web Development Software Development",
        "Cloud Computing DevOps",
        "Data Analytics Business"
    ]
}

df = pd.DataFrame(career_data)

# ---------------- HEADER ----------------
st.title("🎯 AI-Based Career Recommendation System")

st.write(
    "An intelligent career recommendation platform that analyzes "
    "student skills and interests to identify suitable career paths."
)

st.divider()

# ---------------- STUDENT PROFILE ----------------
st.header("👤 Student Profile")

name = st.text_input("Enter your name")

col1, col2 = st.columns(2)

with col1:
    skills = st.multiselect(
        "💻 Select your skills",
        [
            "Python",
            "Java",
            "SQL",
            "Excel",
            "Power BI",
            "Machine Learning",
            "Artificial Intelligence",
            "Statistics",
            "Pandas",
            "HTML",
            "CSS",
            "JavaScript",
            "AWS",
            "Linux",
            "Networking",
            "Communication",
            "APIs",
            "OOP"
        ]
    )

with col2:
    interests = st.multiselect(
        "❤️ Select your interests",
        [
            "Data Analytics",
            "Data Science",
            "Artificial Intelligence",
            "Machine Learning",
            "Web Development",
            "Software Development",
            "Cloud Computing",
            "DevOps",
            "Programming",
            "Business"
        ]
    )

st.divider()

# ---------------- RECOMMENDATION ----------------
if st.button("🚀 Recommend My Career", use_container_width=True):

    if not name:
        st.warning("Please enter your name.")

    elif not skills and not interests:
        st.warning("Please select at least one skill or interest.")

    else:

        # Combine user information
        user_profile = " ".join(skills + interests)

        # Combine career information
        career_profiles = (
            df["Skills"] + " " + df["Interests"]
        ).tolist()

        # TF-IDF + Cosine Similarity
        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform(
            career_profiles + [user_profile]
        )

        similarity = cosine_similarity(
            vectors[-1],
            vectors[:-1]
        )[0]

        # Convert similarity to percentage
        match_percentage = similarity * 100

        # Create result dataframe
        results = pd.DataFrame({
            "Career": df["Career"],
            "Match": match_percentage
        })

        results = results.sort_values(
            by="Match",
            ascending=False
        ).head(3)

        # ---------------- RESULTS ----------------
        st.success(
            f"Hello {name}! Your personalized career analysis is ready."
        )

        st.header("🎯 Recommended Career Paths")

        for _, row in results.iterrows():

            career = row["Career"]
            score = min(round(row["Match"], 1), 99.0)

            st.subheader(f"💼 {career}")

            st.progress(
                int(score)
            )

            st.write(f"**Career Match: {score}%**")

        st.divider()

        # ---------------- TOP CAREER ----------------
        top_career = results.iloc[0]["Career"]

        st.header("🏆 Top Career Recommendation")

        st.info(
            f"Based on your selected skills and interests, "
            f"**{top_career}** has the highest similarity with your profile."
        )

        # ---------------- SKILL GAP ----------------
        st.header("🔍 Skill Gap Analysis")

        career_row = df[df["Career"] == top_career].iloc[0]

        required_skills = career_row["Skills"].split()

        normalized_user_skills = [skill.lower() for skill in skills]

        missing_skills = [
            skill
            for skill in required_skills
            if skill.lower() not in normalized_user_skills
        ]

        if missing_skills:

            st.write(
                "Skills you can develop for this career:"
            )

            for skill in missing_skills:
                st.write(f"📌 {skill}")

        else:

            st.success(
                "You have selected the major skills associated "
                "with this career profile!"
            )

        # ---------------- LEARNING ROADMAP ----------------
        st.header("📚 Personalized Learning Roadmap")

        roadmap = {
            "Python": "Strengthen Python programming and problem-solving.",
            "SQL": "Learn SQL queries, joins, grouping and databases.",
            "Power BI": "Create interactive dashboards and reports.",
            "Statistics": "Study probability, statistics and hypothesis testing.",
            "Machine": "Learn machine learning algorithms and model evaluation.",
            "Learning": "Practice machine learning with real datasets.",
            "Artificial": "Learn AI fundamentals and intelligent systems.",
            "Intelligence": "Explore AI concepts and practical applications.",
            "AWS": "Learn AWS fundamentals and cloud services.",
            "Linux": "Practice Linux commands and system administration.",
            "Networking": "Learn networking fundamentals and protocols.",
            "HTML": "Learn semantic HTML and webpage structure.",
            "CSS": "Practice responsive web design.",
            "JavaScript": "Learn JavaScript fundamentals and DOM manipulation.",
            "Excel": "Practice formulas, pivot tables and data analysis.",
            "Pandas": "Learn data manipulation and analysis using Pandas."
        }

        shown = 0

        for skill in missing_skills:

            if skill in roadmap:

                shown += 1

                st.write(
                    f"**{shown}. {skill}** — {roadmap[skill]}"
                )

            if shown >= 5:
                break

        if shown == 0:

            st.write(
                "Continue building projects and strengthening "
                "your existing skills."
            )

        st.divider()

        # ---------------- DISCLAIMER ----------------
        st.caption(
            "Note: This system provides career recommendations "
            "based on the selected profile. It is intended as a "
            "decision-support tool and not as a definitive career assessment."
        )

# ---------------- FOOTER ----------------
st.divider()

st.caption(
    "AI-Based Career Recommendation System | "
    "Python • Scikit-learn • Pandas • Streamlit"
)