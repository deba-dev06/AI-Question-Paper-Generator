from generator import generate_questions
from utils import read_file
import streamlit as st
import io
from reportlab.platypus import SimpleDocTemplate,Spacer,Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4


def process(file, marks):
    text = read_file(file)
    questions = generate_questions(text, marks)
    return questions

def create_pdf(questions):
    buffer = io.BytesIO()
    styles=getSampleStyleSheet()
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
st.title("AI Tutor")
uploaded_file=st.file_uploader("Upload Your PDF", type=["pdf"])
marks=st.radio("Select Marks", (2, 3, 5))
if st.button("Generate Questions"):
    if uploaded_file is not None:
        questions = process(uploaded_file, marks)
        st.subheader("Generated QP:")
        st.text(questions)
        pdf_buffer = create_pdf([a for a in questions.split("\n") if a != ""])
        st.download_button(
            label="Download Question Paper",
            data=pdf_buffer,
            file_name="question_paper.pdf",
            mime="application/pdf"
        )
    else:
        st.warning("Please upload a PDF file.")