"""
Batch-converts all .docx files in a folder to PDF, in filename order.
Requires Microsoft Word installed (Windows/Mac).
"""

import os
from pathlib import Path
from docx2pdf import convert

# Folder containing your downloaded/extracted .docx files
SOURCE_DIR = Path("./specs_docx")   # put all your .docx files here
OUTPUT_DIR = Path("./specs_pdf")    # converted PDFs land here

OUTPUT_DIR.mkdir(exist_ok=True)

# Get all docx files, sorted so multi-part specs (like 24.501) stay in order
docx_files = sorted(SOURCE_DIR.glob("*.docx"))

if not docx_files:
    print(f"No .docx files found in {SOURCE_DIR.resolve()}")
else:
    print(f"Found {len(docx_files)} files. Converting...")
    for f in docx_files:
        print(f"  -> {f.name}")

    # docx2pdf converts a whole folder in one call
    convert(str(SOURCE_DIR), str(OUTPUT_DIR))

    print(f"\nDone. PDFs saved to {OUTPUT_DIR.resolve()}")