
import pymupdf

pdf_path = "documents/How-To-Make-Money.pdf"

doc = pymupdf.open(pdf_path)

print("Total pages:", len(doc))

for page_number in range(min(3, len(doc))):
    page = doc[page_number]
    text = page.get_text()

    print(f"\n--- Page {page_number + 1} ---")
    print(text[:1500])

doc.close()
