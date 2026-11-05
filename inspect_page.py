
import pymupdf

pdf_path = "documents/How-To-Make-Money.pdf"

with pymupdf.open(pdf_path) as doc:
    for page_number in [9, 19, 29]:
        if page_number < len(doc):
            page = doc[page_number]

            print(f"\n{'=' * 20} PAGE {page_number + 1} {'=' * 20}")
            print(page.get_text()[:2500])
