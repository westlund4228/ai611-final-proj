from pathlib import Path
from pdf2image import convert_from_path

pdfs = Path(".").glob("*.pdf")

for pdf in pdfs:
    pages = convert_from_path(pdf)
    page = pages[0]
    out_path = Path(f"{pdf.stem}.png")
    page.save(out_path, "PNG")
    print(f"Converted {pdf.name}")
