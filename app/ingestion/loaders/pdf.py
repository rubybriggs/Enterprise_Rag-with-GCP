import io
import time
import gc
import logfire
from pypdf import PdfReader, PdfWriter
from google.cloud import documentai
from google.api_core import exceptions as gcp_exceptions
from app.config import settings

client = documentai.DocumentProcessorServiceClient()
MAX_PAGES_PER_REQUEST = 15

def parse_pdf(file_path: str):
    """
    Parses PDF using Google Cloud Document AI.
    Automatically splits large PDFs into 15-page chunks to bypass synchronous API limits.
    Memory-optimized to prevent OOM crashes on massive documents.
    """
    with logfire.span("📄 Document AI Parsing", filename=file_path):
        try:
            reader = PdfReader(file_path)
            total_pages = len(reader.pages)
            logfire.info(f"Total pages: {total_pages}")

            name = client.processor_path(
                settings.PROJECT_ID, 
                settings.GCP_DOC_AI_LOCATION, 
                settings.GCP_DOC_AI_PROCESSOR_ID
            )

            # If small enough, process entirely
            if total_pages <= MAX_PAGES_PER_REQUEST:
                with open(file_path, "rb") as f:
                    image_content = f.read()
                full_text = process_document_chunk(image_content, name)
                
                if not full_text.strip():
                    logfire.warning(f"⚠️ Document AI returned empty text for {file_path}")
                else:
                    logfire.info(f"✅ Document AI successfully parsed {len(full_text)} characters")
                return full_text

            # --- HANDLING MASSIVE PDFs ---
            logfire.info(f"PDF exceeds {MAX_PAGES_PER_REQUEST} pages. Splitting into chunks...")
            text_parts = [] # Use a list to prevent Python string memory fragmentation
            
            for i in range(0, total_pages, MAX_PAGES_PER_REQUEST):
                writer = PdfWriter()
                chunk_end = min(i + MAX_PAGES_PER_REQUEST, total_pages)
                
                for page_num in range(i, chunk_end):
                    writer.add_page(reader.pages[page_num])
                
                # Write chunk to bytes
                with io.BytesIO() as bytes_stream:
                    writer.write(bytes_stream)
                    chunk_bytes = bytes_stream.getvalue()
                
                # Delete the writer object immediately to free up RAM
                del writer 

                with logfire.span(f"Processing pages {i+1} to {chunk_end}"):
                    try:
                        chunk_text = process_document_chunk(chunk_bytes, name)
                    except gcp_exceptions.ResourceExhausted:
                        # If we hit a Google API rate limit, pause and retry once
                        logfire.warning("⏳ Hit Google API Quota limit. Sleeping for 10 seconds before retrying...")
                        time.sleep(10)
                        chunk_text = process_document_chunk(chunk_bytes, name)

                    text_parts.append(chunk_text)

                # Explicitly clear massive variables from memory and force garbage collection
                del chunk_bytes
                gc.collect()

                # Sleep slightly to avoid hammering the Google API and triggering limits
                time.sleep(1.5)

            # Join all parts at the very end
            full_text = "\n".join(text_parts)

            if not full_text.strip():
                logfire.warning(f"⚠️ Document AI returned empty text for {file_path}")
            else:
                logfire.info(f"✅ Document AI successfully parsed {len(full_text)} characters")

            return full_text

        except Exception as e:
            logfire.error(f"❌ Document AI Parse Failed: {e}")
            logfire.info("💡 Ensure the Processor ID is correct and the API is enabled.")
            raise e


def process_document_chunk(image_content: bytes, name: str) -> str:
    """Helper function to send a specific byte chunk to Document AI"""
    raw_document = documentai.RawDocument(
        content=image_content, 
        mime_type="application/pdf"
    )

    request = documentai.ProcessRequest(
        name=name, 
        raw_document=raw_document
    )

    result = client.process_document(request=request)
    return result.document.text
    