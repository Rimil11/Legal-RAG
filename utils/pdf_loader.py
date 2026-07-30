import fitz
import os
from langchain_core.documents import Document


REMOVE_KEYWORDS = [
    "TABLE OF CONTENTS",
    "CONTENTS",
    "ARRANGEMENT OF SECTIONS",
    "LIST OF ABBREVIATIONS"
]


def clean_page(text: str):

    lines = []

    for line in text.splitlines():

        line = line.strip()

        if not line:
            continue

        # remove page numbers
        if line.isdigit():
            continue

        lines.append(line)

    return "\n".join(lines)


def load_pdf(pdf_path):

    pdf = fitz.open(pdf_path)

    documents = []

    for page_number, page in enumerate(pdf):

        text = page.get_text()

        if not text.strip():
            print("Blank page")
            continue

        upper = text.upper()

        if any(keyword in upper for keyword in REMOVE_KEYWORDS):
            print(f"Skipping page {page_number+1}")
            continue

        print("Page loaded")
        text = clean_page(text)

        documents.append(

            Document(

                page_content=text,

                metadata={

                    "source": os.path.basename(pdf_path),

                    "page": page_number + 1,

                },

            )

        )

    return documents