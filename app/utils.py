import markdown
import fitz  # PyMuPDF

def format_content(content):
    """Convert markdown to HTML"""
    html_content = markdown.markdown(content)
    return html_content

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file"""
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text("text") + "\n"
        return text.strip()
    except Exception as e:
        raise Exception(f"Error extracting text from PDF: {str(e)}")