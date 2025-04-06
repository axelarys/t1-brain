import os
import fitz  # PyMuPDF
import docx
import openpyxl

def run_action(file_path: str, summary: bool = False, **kwargs) -> str:
    """
    Read and optionally summarize the contents of a file.
    
    Supports: PDF, DOCX/DOC, XLSX/XLS files.

    Parameters:
    - file_path (str): Absolute path to the file.
    - summary (bool): Return first 2000 chars if True.
    
    Returns:
    - str: Extracted or summarized text.
    """
    if not os.path.exists(file_path):
        return f"❌ File not found: {file_path}"

    ext = file_path.lower().split(".")[-1]
    text = ""

    try:
        if ext == "pdf":
            text = extract_pdf(file_path)
        elif ext in ["doc", "docx"]:
            text = extract_docx(file_path)
        elif ext in ["xls", "xlsx"]:
            text = extract_excel(file_path)
        else:
            return f"❌ Unsupported file type: .{ext}"

        if not text.strip():
            return "⚠️ No readable text found in file."

        return text[:2000] + "..." if summary and len(text) > 2000 else text

    except Exception as e:
        return f"❌ Error reading file: {str(e)}"


def extract_pdf(path):
    """Extract text from a PDF using PyMuPDF."""
    text = ""
    with fitz.open(path) as doc:
        for page in doc:
            text += page.get_text()
    return text


def extract_docx(path):
    """Extract text from a Word document."""
    doc = docx.Document(path)
    return "\n".join([para.text for para in doc.paragraphs])


def extract_excel(path):
    """Extract text from all cells in all sheets of an Excel file."""
    wb = openpyxl.load_workbook(path)
    all_text = ""
    for sheet in wb.worksheets:
        for row in sheet.iter_rows(values_only=True):
            row_text = " | ".join([str(cell) if cell is not None else "" for cell in row])
            all_text += row_text + "\n"
    return all_text
