import re
from collections import Counter
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfpage import PDFPage
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from io import StringIO
import spacy
import fitz
from spacy.matcher import Matcher


def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        pdf_document = fitz.open(pdf_path)
        num_pages = pdf_document.page_count

        for page_num in range(num_pages):
            page = pdf_document.load_page(page_num)
            text += page.get_text()
    except Exception as e:
        print("Error extracting text from PDF:", e)
    
    return text


def extract_contact_number_from_resume(text):
    contact_number = None

    # Use regex pattern to find potential contact numbers
    patterns = [
        r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",  # XXX-XXX-XXXX or (XXX) XXX-XXXX
        r"\b\d{10}\b",  # 10-digit numbers
        r"\b\d{3}[-.\s]?\d{4}[-.\s]?\d{3}\b",  # XXX-XXXX-XXX or XXX.XXXX.XXX
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            contact_number = match.group()
            break

    return contact_number

def extract_email_from_resume(text):
    email = None

    # Use regex pattern to find a potential email address
    pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
    match = re.search(pattern, text)
    if match:
        email = match.group()

    return email

def extract_skills_from_resume(text, skills_list):
    skills = []

    for skill in skills_list:
        pattern = r"\b{}\b".format(re.escape(skill))
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            skills.append(skill)

    return skills
def get_skill_frequency(text, skills_list):
    skill_counts = Counter()

    for skill in skills_list:
        pattern = r"\b{}\b".format(re.escape(skill))
        matches = re.findall(pattern, text, re.IGNORECASE)
        skill_counts[skill] = len(matches)

    return skill_counts

def extract_education_from_resume(text):
    education = []

    # Use regex pattern to find education information
    pattern = r"(?i)(?:Bsc|\bB\.\w+|\bM\.\w+|\bPh\.D\.\w+|\bBachelor(?:'s)?|\bMaster(?:'s)?|\bPh\.D)\s(?:\w+\s)*\w+"
    matches = re.findall(pattern, text)
    for match in matches:
        education.append(match.strip())

    return education

def extract_name(resume_text):
    nlp = spacy.load('en_core_web_sm')
    matcher = Matcher(nlp.vocab)

    # Define name patterns
    patterns = [
        [{'POS': 'PROPN'}, {'POS': 'PROPN'}],  # First name and Last name
        [{'POS': 'PROPN'}, {'POS': 'PROPN'}, {'POS': 'PROPN'}],  # First name, Middle name, and Last name
        [{'POS': 'PROPN'}, {'POS': 'PROPN'}, {'POS': 'PROPN'}, {'POS': 'PROPN'}]  # First name, Middle name, Middle name, and Last name
        # Add more patterns as needed
    ]

    for pattern in patterns:
        matcher.add('NAME', patterns=[pattern])

    doc = nlp(resume_text)
    matches = matcher(doc)

    for match_id, start, end in matches:
        span = doc[start:end]
        return span.text

    return None