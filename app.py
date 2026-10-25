from generator import generate_questions
from utils import read_file
from metrics import check_validity, relevance_score
import streamlit as st
import io
from reportlab.platypus import SimpleDocTemplate, Spacer, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="AI Question Paper Generator",
    page_icon="📄",
    layout="centered",
)

# ─────────────────────────────────────────────
# Session state initialisation
# ─────────────────────────────────────────────
for key in ("source_text", "questions_str", "gen_time", "q_list",
            "validity", "relevance"):
    if key not in st.session_state:
        st.session_state[key] = None

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def process(file, marks):
    text = read_file(file)
    questions_str, gen_time = generate_questions(text, marks)
    return text, questions_str, gen_time


def create_pdf(questions):
    buffer = io.BytesIO()
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    story = []
    story.append(Paragraph("University Of Engineering And Management", styles["Title"]))
    story.append(Paragraph("Question Paper", styles["Title"]))
    story.append(Spacer(1, 20))
    for i, question in enumerate(questions):
        story.append(Paragraph(f"Q{i+1}. {question}", styles["Normal"]))
        story.append(Spacer(1, 10))
    doc.build(story)
    buffer.seek(0)
    return buffer


# ─────────────────────────────────────────────
# UI — Input
# ─────────────────────────────────────────────
st.title("📄 AI Question Paper Generator")
st.caption("Powered by flan-t5-base · Built with Streamlit")

uploaded_file = st.file_uploader("Upload Your PDF", type=["pdf"])
marks = st.radio("Select Mark Category", (2, 3, 5), horizontal=True,
                 help="2 = short, 3 = medium, 5 = descriptive")

if st.button("⚡ Generate Questions", use_container_width=True):
    if uploaded_file is not None:
        with st.spinner("Generating questions…"):
            source_text, questions_str, gen_time = process(uploaded_file, marks)

        q_list = [q for q in questions_str.split("\n") if q.strip()]

        # ── Persist everything in session state ──────────────────────────
        st.session_state.source_text  = source_text
        st.session_state.questions_str = questions_str
        st.session_state.gen_time     = gen_time
        st.session_state.q_list       = q_list
        st.session_state.validity     = check_validity(q_list, marks)
        st.session_state.relevance    = None   # reset so old result isn't shown
    else:
        st.warning("⚠️ Please upload a PDF file first.")


# ─────────────────────────────────────────────
# UI — Results (shown whenever session state has data)
# ─────────────────────────────────────────────
if st.session_state.q_list:
    q_list   = st.session_state.q_list
    validity = st.session_state.validity

    # ── Questions ────────────────────────────────────────────────────────
    st.subheader("Generated Questions")
    for i, q in enumerate(q_list, 1):
        st.markdown(f"**Q{i}.** {q}")

    pdf_buffer = create_pdf(q_list)
    st.download_button(
        label="📥 Download Question Paper (PDF)",
        data=pdf_buffer,
        file_name="question_paper.pdf",
        mime="application/pdf",
        use_container_width=True,
    )

    st.divider()

    # ════════════════════════════════════════════════════════════════════
    # METRICS DASHBOARD
    # ════════════════════════════════════════════════════════════════════
    st.subheader("📊 Quality Metrics")

    # ── Metric 1 & 2 ────────────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)
    col1.metric("⏱ Generation Time", f"{st.session_state.gen_time} s",
                help="Wall-clock time for the flan-t5-base inference call")
    col2.metric("✅ Well-Formed",
                f"{validity['well_formed']['percent']}%",
                help="% starting with Define/Explain/… or ending with '?'")
    col3.metric("📏 Length-Compliant",
                f"{validity['right_length']['percent']}%",
                help=f"% within expected word-count range for {marks}-mark questions")

    col4, col5 = st.columns(2)
    col4.metric("🔁 Unique Questions",
                f"{validity['unique']['percent']}%",
                help="% with no exact duplicates")
    col5.metric("🔢 Total Questions", validity["total_questions"])



    st.divider()

    # ── Metric 3: Source Relevance ───────────────────────────────────────
    st.subheader("🔗 Source Relevance Analysis")
    st.info(
        "Embeds questions + source PDF with **all-MiniLM-L6-v2** and "
        "measures cosine similarity. Higher = more on-topic."
    )

    if st.button("▶ Run Relevance Analysis", use_container_width=True):
        with st.spinner("Embedding questions and source text… (~10 s)"):
            st.session_state.relevance = relevance_score(
                q_list, st.session_state.source_text, threshold=0.5
            )

    # Show relevance results if they exist in session state
    if st.session_state.relevance is not None:
        rel = st.session_state.relevance
        r1, r2 = st.columns(2)
        r1.metric(
            "Avg Cosine Similarity",
            rel["avg_cosine_similarity"],
            help="Mean best-match similarity vs. source chunks (0–1)",
        )
        r2.metric(
            f"Above {rel['threshold']} Threshold",
            f"{rel['above_threshold']['percent']}%",
            help="% of questions clearly grounded in the source PDF",
        )

        st.markdown("**Per-question similarity scores:**")
        score_data = {f"Q{i+1}": s for i, s in enumerate(rel["per_question_scores"])}
        st.bar_chart(score_data)
