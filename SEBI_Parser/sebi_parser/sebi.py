import requests
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
import re
import json
from collections import defaultdict
import fitz  
from .DocumentExtractor import DocumentExtractor
from transformers.pipelines import pipeline
from multiprocessing import Pool
from datetime import datetime
import os
import time
from urllib.parse import urljoin, unquote
import pytesseract
from PIL import Image 
import io 





# =======================
# CONFIG
# =======================

HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64)"
}
NUM_WORKERS = 2
TIMEOUT_PER_ITEM = 300
_global_summarizer = None


# =======================
# SEBI RSS PARSER
# =======================

class SEBIRSSParser:

    def __init__(self):
        self.rss_url = "https://www.sebi.gov.in/sebirss.xml"

        self.relevant_keywords = [
            "mutual fund", "mf", "ipo", "equity",
            "derivative", "future", "option",
            "stock broker", "trading member",
            "clearing", "settlement", "margin",
            "kyc", "pms", "aif", "custodian",
            "capital market", "securities", "auction", "regulations", "adjudication", "sebi", "stock options"
        ]

        self.irrelevant_keywords = [
            "tender", "recruitment", "vacancy",
            "court", "penalty", "recovery", "notice"
        ]

        self.extractor = DocumentExtractor()



        self.summarizer = None
        self.use_transformer = False

    # def get_summarizer(self):
    #     if self.summarizer is not None:
    #         return self.summarizer

    #     try:
    #         print("Loading transformer summarizer...")

    #         self.summarizer = pipeline(
    #             "summarization",
    #             model="facebook/bart-large-cnn",
    #             device=-1,
    #             framework="pt"
    #         )

    #         self.use_transformer = True
    #         return self.summarizer

    #     except Exception as e:
    #         print("Transformer load failed:")
    #         print(e)
    #         self.use_transformer = False
    #         return None
    

    def get_summarizer(self):
        """Load summarizer once per worker process"""
        global _global_summarizer
        
        if _global_summarizer is not None:
            return _global_summarizer

        try:
            from transformers import pipeline
            print(f"[PID {os.getpid()}] Loading transformer summarizer...")
            
            _global_summarizer = pipeline(
                "summarization",
                model="facebook/bart-large-cnn",
                device=-1,
                framework="pt"
            )
            
            if _global_summarizer.model.config.max_length is None:
                _global_summarizer.model.config.max_length = 1024
            
            print(f"[PID {os.getpid()}] ✅ Summarizer loaded")
            return _global_summarizer
            
        except Exception as e:
            print(f"[PID {os.getpid()}] ❌ Transformer load failed: {e}")
            return None

    # =======================
    # RSS
    # =======================

    def fetch_rss(self):
        """Fetch RSS XML from SEBI website"""
        r = requests.get(self.rss_url, headers=HEADERS, timeout=30)
        r.raise_for_status()
        return r.content

    def parse_xml(self, xml):
        """Parse XML and extract items"""
        root = ET.fromstring(xml) # Parse XML content to get root element (element tree object)
        items = []

        for item in root.findall(".//item"):
            items.append({
                "title": item.findtext("title", ""),
                "description": item.findtext("description", ""),
                "link": item.findtext("link", ""),
                "pub_date": item.findtext("pubDate", "")
            })

        return items

    # =======================
    # RELEVANCE
    # =======================

    def calculate_score(self, title, desc):
        """Calculate relevance score based on keywords"""
        text = f"{title} {desc}".lower()
        score = 0

        for k in self.relevant_keywords:
            if k in text:
                score += 2

        for k in self.irrelevant_keywords:
            if k in text:
                score -= 3

        return score

    # =======================
    # PDF LINK EXTRACTION
    # =======================


    def extract_pdf_links(self,html_url):
        if html_url.lower().endswith(".pdf"):
            return [html_url]
        """Extract PDF links from HTML page"""
        r = requests.get(html_url, headers=HEADERS, timeout=120)
        r.raise_for_status()

        soup = BeautifulSoup(r.content, "html.parser")
        pdf_links = set()

        # ---------- Anchor tags ----------
        for a in soup.find_all("a",href=True):
            href = a["href"].strip()
            if not href.lower().endswith(".pdf"):
                continue

            if href.startswith("https") and href.endswith(".pdf"):
                pdf_links.add(href)  

        for iframe in soup.find_all("iframe", src=True):
            src = iframe["src"].strip()

            # Case 1: PDF wrapped inside file=
            match = re.search(r'file=(https?://[^&]+\.pdf)', src, re.IGNORECASE)
            if match:
                pdf_links.add(unquote(match.group(1)))
                continue

            # Case 2: iframe directly points to PDF
            if src.lower().endswith(".pdf"):
                pdf_links.add(src)
            

        return list(pdf_links)    

  
    # =======================
    # PDF TEXT EXTRACTION
    # =======================

    def extract_text_from_pdf(self, pdf_url):
        """Extract text from PDF using PyMuPDF"""
        r = requests.get(pdf_url, headers=HEADERS, timeout=60)
        r.raise_for_status()

        doc = fitz.open(stream=r.content, filetype="pdf")
        text = ""

        for page in doc:
            # text += page.get_text()
            text = page.get_text().strip()

            # If no text → OCR
            if not text:
                pix = page.get_pixmap(dpi=300)
                img = Image.open(io.BytesIO(pix.tobytes("png")))
                text = pytesseract.image_to_string(img)

        text += " " + text    

        text = re.sub(r'\s+', ' ', text).strip()
        return text

    # =======================
    # CHUNKING
    # =======================

    def chunk_text(self, text, tokenizer, max_tokens=900):
        """
        Split text into token-safe chunks for summarization.
        Increased default to 900 tokens for better context preservation.
        """
        tokens = tokenizer(
            text,
            return_tensors="pt",
            truncation=False,
            add_special_tokens=False  
        )["input_ids"][0]

        chunks = []
        
        for i in range(0, len(tokens), max_tokens):
            chunk_tokens = tokens[i:i + max_tokens]
            chunk_text = tokenizer.decode(
                chunk_tokens,
                skip_special_tokens=True
            )

            # Only yield chunks with meaningful content
            if len(chunk_text.strip()) > 50:
                chunks.append(chunk_text)
        
        return chunks


    def summarize_short(self, text):
        if not text or len(text) < 300:
            return text

        summarizer = self.get_summarizer()
        if not summarizer:
            return text[:200] + "..."

        tokenizer = summarizer.tokenizer
        print(f" tokenizer: {len(tokenizer)}")

        # ---------- PASS 1: chunk summaries ----------
        chunks = self.chunk_text(text, tokenizer, max_tokens=900)
        if not chunks:
            return text[:200] + "..."

        partial_summaries = []

        for idx, chunk in enumerate(chunks):
            word_count = len(chunk.split())
            if word_count < 80:
                continue

            max_len = min(180, int(word_count * 0.4))
            min_len = min(80, int(word_count * 0.25))

            try:
                out = summarizer(
                    chunk,
                    max_length=max_len,
                    min_length=min_len,
                    do_sample=False,
                    truncation=True
                )
                partial_summaries.append(out[0]["summary_text"])
            except Exception as e:
                print(f"Chunk {idx + 1} summary failed: {e}")
                partial_summaries.append(chunk[:300] + "...")

        if not partial_summaries:
            return text[:200] + "..."

        # ---------- PASS 2: GROUP chunk summaries ----------
        GROUP_SIZE = 4
        grouped_texts = []

        for i in range(0, len(partial_summaries), GROUP_SIZE):
            group = partial_summaries[i:i + GROUP_SIZE]
            grouped_texts.append(" ".join(group))

        # ---------- PASS 3: section-level summaries ----------
        section_summaries = []

        for idx, group_text in enumerate(grouped_texts):
            group_words = len(group_text.split())
            if group_words < 100:
                section_summaries.append(group_text)
                continue

            max_len = min(200, int(group_words * 0.4))
            min_len = min(100, int(group_words * 0.25))

            try:
                out = summarizer(
                    group_text,
                    max_length=max_len,
                    min_length=min_len,
                    do_sample=False,
                    truncation=True
                )
                section_summaries.append(out[0]["summary_text"])
            except Exception as e:
                print(f"Group {idx + 1} summary failed: {e}")
                section_summaries.append(group_text[:300] + "...")

        # ---------- FINAL OUTPUT (NO PASS 4) ----------
        return " ".join(section_summaries)

    # =======================
    # SUMMARIZATION - LONG 
    # =======================

    
    def summarize_long(self, text):

        if not text or len(text) < 300:
            return text

        summarizer = self.get_summarizer()
        if not summarizer:
            return text[:300] + "..."

        tokenizer = summarizer.tokenizer

        # -----------------------------
        # Step 1: Token-safe chunking
        # -----------------------------
        chunks = self.chunk_text(text, tokenizer, max_tokens=900)
        if not chunks:
            return text[:300] + "..."

        summaries = []

        # -----------------------------
        # Step 2: Summarize each chunk
        # -----------------------------
        for chunk in chunks:
            word_count = len(chunk.split())

            # Skip meaningless chunks
            if word_count < 60:
                continue

            # Light compression (retain details)
            max_len = min(400, int(word_count * 0.6))
            min_len = min(150, int(word_count * 0.4))

            if min_len >= max_len:
                min_len = max(40, max_len - 40)

            try:
                result = summarizer(
                    chunk,
                    max_length=max_len,
                    min_length=min_len,
                    do_sample=False,
                    truncation=True
                )
                summaries.append(result[0]["summary_text"])

            except Exception:
                # Safe fallback: include partial original content
                print(f"long summaries chunks failed {len(chunk)}")
                summaries.append(chunk[:800] + "...")

        if not summaries:
            print("summaries not found")
            return text[:2000] + "..."

        # -----------------------------
        # Step 3: Combine summaries ONLY
        # -----------------------------
        final_summary = "\n\n".join(summaries)

        return final_summary


    def pdf_to_xml(self, pdf_url, title):
        # Fetch PDF
        r = requests.get(pdf_url, headers=HEADERS, timeout=60)
        r.raise_for_status()
        
        # Open PDF and extract all text
        doc = fitz.open(stream=r.content, filetype="pdf")
        full_text = ""
        
        for page in doc:
            full_text += page.get_text()
        
        doc.close()
        
        # Create XML
        root = ET.Element("document")
        
        # Metadata
        meta = ET.SubElement(root, "metadata")
        ET.SubElement(meta, "title").text = title
        ET.SubElement(meta, "source").text = pdf_url
        
        # Content
        content = ET.SubElement(root, "content")
        
        # Split into paragraphs (by double newline or single newline)
        paragraphs = [p.strip() for p in full_text.split("\n") if p.strip()]
        
        for para_text in paragraphs:
            para = ET.SubElement(content, "paragraph")
            para.text = para_text
        
        return ET.tostring(root, encoding="unicode", method="xml")



    # =======================
    # MAIN PROCESS
    # =======================

    # def process_items(self, items):
    #     """Process all RSS items and generate results"""
    #     results = defaultdict(dict)

    #     for idx, item in enumerate(items, 1):
    #         score = self.calculate_score(item["title"], item["description"])
            
    #         if score < 2:
    #             continue

    #         print(f"Processing item {idx}: {item['title'][:60]}... (score: {score})")

    #         pdf_links = self.extract_pdf_links(item["link"])
            
    #         # Initialize variables with defaults
    #         full_text = ""
    #         extracted_data = {}
    #         xml_content = ""
    #         long_summary = "No PDF available"
    #         short_summary = "No PDF available"

    #         if pdf_links:
    #             try:
    #                 print(f"  Extracting from PDF: {pdf_links[0]}")
    #                 text = self.extract_text_from_pdf(pdf_links[0])
    #                 textlen = len(text)
    #                 print(f"  Extracted {textlen} characters")
    #                 if textlen > 150000:
    #                     continue
    #                 full_text = text
    #                 extracted_data = self.extractor.extract_document_fields(full_text)
    #                 xml_content = self.pdf_to_xml(pdf_links[0], item["title"])
    #                 t = time.time()
    #                 short_summary = self.summarize_short(text)
    #                 t_s = time.time() - t
    #                 print(f"  Short summary generated in {t_s:.2f} seconds")
    #                 print(f"  Short summary generated: {len(short_summary)} characters")
    #                 t_la = time.time()
    #                 long_summary = self.summarize_long(text)
    #                 t_l = time.time() - t_la
    #                 print(f"  Long summary generated in {t_l:.2f} seconds")
    #                 print(f"  Long summary generated: {len(long_summary)} characters")
    #             except Exception as e:
    #                 print(f"  Error processing PDF: {e}")
    #                 long_summary = f"Error processing PDF: {str(e)}"
    #                 short_summary = f"Error processing PDF: {str(e)}"
    #         else:
    #             print(f"  No PDF found")

    #         results[item["pub_date"]][f"item_{idx}"] = {
    #             "score": score,
    #             "title": item["title"],
    #             "publish_date": item["pub_date"],
    #             "link": item["link"],
    #             "pdf_links": pdf_links,
    #             "long_summary": long_summary,
    #             "short_summary": short_summary,
    #             "extracted_metadata": extracted_data,
    #             "full_text": full_text,
    #             "xml_content": xml_content
    #         }

    #     return dict(results)




    def process_single_item(self, item):
        """Process single RSS item"""
        parser = self
        pid = os.getpid()

        # Initialize defaults
        full_text = ""
        extracted_data = {}
        xml_content = ""
        long_summary = "No PDF available"
        short_summary = "No PDF available"
        pdf_url = None
        pdfs = []

        try:
            print(f"[PID {pid}] Processing: {item['title'][:60]}...")

            score = parser.calculate_score(item["title"], item["description"])
            print(f"[PID {pid}] Score: {score}")

            if score < 2:
                print(f"[PID {pid}] ❌ Low score, skipping")
                return None

            print(f"[PID {pid}] Extracting PDF links...")
            pdfs = parser.extract_pdf_links(item["link"])

            if not pdfs:
                print(f"[PID {pid}] ⚠️ No PDF found")
                return {
                    "score": score,
                    "title": item["title"],
                    "publish_date": item["pub_date"],
                    "link": item["link"],
                    "pdf_url": None,
                    "pdf_links": [],
                    "long_summary": "No PDF found",
                    "short_summary": "No PDF found",
                    "extracted_metadata": {},
                    "full_text": "",
                    "xml_content": ""
                }

            pdf_url = pdfs[0]
            print(f"[PID {pid}] Extracting text from PDF...")

            text = parser.extract_text_from_pdf(pdf_url)
            text_len = len(text)
            print(f"[PID {pid}] Extracted {text_len} characters")

            if text_len == 0:
                print(f"[PID {pid}] ⚠️ Empty PDF, skipping")
                return {
                    "score": score,
                    "title": item["title"],
                    "publish_date": item["pub_date"],
                    "link": item["link"],
                    "pdf_url": pdf_url,
                    "pdf_links": pdfs,
                    "long_summary": "Empty PDF - no text extracted",
                    "short_summary": "Empty PDF - no text extracted",
                    "extracted_metadata": {},
                    "full_text": "",
                    "xml_content": ""
                }

            if text_len > 100000:
                print(f"[PID {pid}] ⚠️ PDF too large, skipping summarization")
                return None

            full_text = text

            print(f"[PID {pid}] Extracting metadata...")
            extracted_data = parser.extractor.extract_document_fields(full_text)

            print(f"[PID {pid}] Generating short summary...")
            short_summary = parser.summarize_short(text)
            print(f"[PID {pid}] Short summary: {len(short_summary)} chars")

            print(f"[PID {pid}] Generating long summary...")
            long_summary = parser.summarize_long(text)
            print(f"[PID {pid}] Long summary: {len(long_summary)} chars")

            print(f"[PID {pid}] Converting to XML...")
            xml_content = parser.pdf_to_xml(pdf_url, item["title"], full_text)

            print(f"[PID {pid}] ✅ Success")

            return {
                "score": score,
                "title": item["title"],
                "publish_date": item["pub_date"],
                "link": item["link"],
                "pdf_url": pdf_url,
                "pdf_links": pdfs,
                "long_summary": long_summary,
                "short_summary": short_summary,
                "extracted_metadata": extracted_data,
                "full_text": full_text,
                "xml_content": xml_content
            }

        except Exception as e:
            print(f"[PID {pid}] ❌ Error: {str(e)}")
            return {
                "score": score if 'score' in locals() else 0,
                "title": item["title"],
                "publish_date": item["pub_date"],
                "link": item["link"],
                "pdf_url": pdf_url,
                "pdf_links": pdfs,
                "long_summary": f"Error: {str(e)}",
                "short_summary": f"Error: {str(e)}",
                "extracted_metadata": extracted_data,
                "full_text": full_text,
                "xml_content": xml_content
            }


    
# def run_parallel(items):
#     """Run parallel processing with timeout"""
#     results = []

#     with Pool(NUM_WORKERS) as pool:
#         # Submit all jobs
#         job_data = []
#         for item in items:
#             job = pool.apply_async(process_single_item, (item,))
#             job_data.append((item, job))

#         # Collect results in order
#         for item, job in job_data:
#             try:
#                 res = job.get(timeout=TIMEOUT_PER_ITEM)
#                 if res:
#                     results.append(res)
#                 else:
#                     print(f"⚠️ Skipped: {item['title'][:40]}")
#             except Exception as e:
#                 print(f"❌ Timeout/Error for {item['title'][:40]}: {str(e)}")
#                 results.append({
#                     "score": 0,
#                     "title": item["title"],
#                     "publish_date": item["pub_date"],
#                     "link": item["link"],
#                     "pdf_url": None,
#                     "pdf_links": [],
#                     "long_summary": f"Timeout: {str(e)}",
#                     "short_summary": f"Timeout: {str(e)}",
#                     "extracted_metadata": {},
#                     "full_text": "",
#                     "xml_content": "",
#                     "error": f"Timeout: {str(e)}"
#                 })

#     return results



    def run_parallel_imap(self, items):
        results = []

        with Pool(NUM_WORKERS) as pool:
            for res in pool.imap_unordered(self.process_single_item, items):
                if res is None:
                    results.append({
                        "error": "Skipped due to low score or filtering"
                    })
                else:
                    results.append(res)

        return results
    
    

    def run(self):
        """Main execution method"""
        print("="*70)
        print("SEBI RSS Parser - Starting")
        print("="*70)
        
        print("\n[1/3] Fetching RSS feed...")
        xml = self.fetch_rss()
        print("RSS feed fetched")
        
        print("\n[2/3] Parsing XML...")
        items = self.parse_xml(xml)
        print(f"Found {len(items)} items")
        
        # print("\n[3/3] Processing items...")
        # results = self.process_items(items)

        print(f"\nProcessing {len(items)} items using {NUM_WORKERS} workers")
        print(f"Timeout per item: {TIMEOUT_PER_ITEM} seconds\n")

        results = self.run_parallel_imap(items)

        
        
        return results


# =======================
# ENTRY POINT
# =======================

def parse_sebi_pdf():
    parser = SEBIRSSParser()
    return parser.run()

