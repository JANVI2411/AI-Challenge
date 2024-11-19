import re
from PyPDF2 import PdfReader
from .config import Config

class DocumentProcessor:
    def __init__(self, chunk_size=Config.CHUNK_SIZE, overlap=Config.CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def load_pdf(self, file_path):
        reader = PdfReader(file_path)
        text = " ".join(page.extract_text() for page in reader.pages)
        return text

    def chunk_text(self,text):
        start = 0
        chunk_list = []
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunk = text[start:end]
            chunk_list.append(chunk)
            start += self.chunk_size - self.overlap
        return chunk_list

    def process_pdf(self, file_path):
        text = self.load_pdf(file_path)
        return self.chunk_text(text)
