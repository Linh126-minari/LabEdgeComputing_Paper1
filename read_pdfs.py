import os
from pypdf import PdfReader

dir_path = "d:/Lab Edge Computing/LabEdgeComputing_Paper1/notebooks"
pdfs = [
    "Joint_Optimization_of_Security_Strength_and_Resource_Allocation_for_Computation_Offloading_in_Vehicular_Edge_Computing.pdf",
    "1-s2.0-S1570870524001562-main.pdf"
]

for pdf in pdfs:
    print(f"=== {pdf} ===")
    try:
        reader = PdfReader(os.path.join(dir_path, pdf))
        # Extract first 2 pages
        text = ""
        for i in range(min(2, len(reader.pages))):
            text += reader.pages[i].extract_text() + "\n"
        print(text[:2500]) # Print first 2500 chars to avoid overwhelming output
    except Exception as e:
        print("Error:", e)
    print("\n" + "="*50 + "\n")
