import requests
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
import re
import json
from collections import defaultdict
import fitz  # PyMuPDF
from .DocumentExtractor import DocumentExtractor


# =======================
# CONFIG
# =======================

HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64)"
}

USE_TRANSFORMER = False
try:
    from transformers import pipeline
    summarizer = pipeline(
        "summarization",
        model="facebook/bart-large-cnn",
        device=-1    # Use CPU
    )
    USE_TRANSFORMER = True
except Exception:
    print("Transformers not available, using fallback summary")


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
            "capital market", "securities"
        ]

        self.irrelevant_keywords = [
            "tender", "recruitment", "vacancy",
            "court", "penalty", "recovery", "notice"
        ]

        self.extractor = DocumentExtractor()

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

    def extract_pdf_links(self, html_url):
        """Extract PDF links from HTML page"""
        r = requests.get(html_url, headers=HEADERS, timeout=30)
        r.raise_for_status()
        soup = BeautifulSoup(r.content, "html.parser")

        pdf_links = set()

        # Anchor tags
        for a in soup.find_all("a", href=True):
            href = a["href"].strip()
            if not href.lower().endswith(".pdf"):
                continue

            if href.startswith("http"):
                pdf_links.add(href)
            elif href.startswith("/"):
                pdf_links.add("https://www.sebi.gov.in" + href)
            else:
                base = html_url.rsplit("/", 1)[0]
                pdf_links.add(base + "/" + href)

        # Iframes with ?file= parameter
        for iframe in soup.find_all("iframe", src=True):
            src = iframe["src"]
            match = re.search(r'file=(https?://[^&]+\.pdf)', src)
            if match:
                from urllib.parse import unquote
                pdf_links.add(unquote(match.group(1)))
            elif src.lower().endswith(".pdf"):
                if src.startswith("http"):
                    pdf_links.add(src)
                elif src.startswith("/"):
                    pdf_links.add("https://www.sebi.gov.in" + src)

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
            text += page.get_text()

        text = re.sub(r'\s+', ' ', text).strip()
        return text

    # =======================
    # CHUNKING
    # =======================

    def chunk_text(self, text, tokenizer, max_tokens=900):
        """Split text into token-safe chunks for summarization"""
        tokens = tokenizer(
            text,
            return_tensors="pt",  # PyTorch tensors
            truncation=False
        )["input_ids"][0]

        for i in range(0, len(tokens), max_tokens):
            chunk_tokens = tokens[i:i + max_tokens]
            chunk_text = tokenizer.decode(
                chunk_tokens,
                skip_special_tokens=True
            )

            if len(chunk_text.strip()) > 100:
                yield chunk_text


    # =======================
    # SUMMARIZATION
    # =======================

    def summarize_text(self, text):
        """Summarize text using NLP or fallback method"""
        if not text or len(text) < 300:
            return text[:100] + "..."

        if not USE_TRANSFORMER:
            return text[:500] + "..."

        summaries = []

        tokenizer = summarizer.tokenizer

        # Token-safe chunking (instead of char-based)
        chunks = list(self.chunk_text(text, tokenizer))

        for chunk in chunks:
            input_len = len(chunk.split())

            # Skip very small chunks
            if input_len < 80:
                continue

            max_len = min(130, int(input_len * 0.6))
            min_len = min(50, int(input_len * 0.3))

            # Ensure valid bounds
            if min_len >= max_len:
                min_len = max(20, max_len - 10)

            try:
                s = summarizer(
                    chunk,
                    max_length=max_len,
                    min_length=min_len,
                    do_sample=False,
                    truncation=True
                )
                summaries.append(s[0]["summary_text"])
            except Exception:
                summaries.append(chunk[:300] + "...")

        if not summaries:
            return text[:500] + "..."

        combined = " ".join(summaries)

        # Final summary with safe length calculation
        combined_len = len(combined.split())
        max_len = min(180, int(combined_len * 0.6))
        min_len = min(80, int(combined_len * 0.3))

        if min_len >= max_len:
            min_len = max(40, max_len - 20)

        try:
            final = summarizer(
                combined,
                max_length=max_len,
                min_length=min_len,
                do_sample=False,
                truncation=True
            )
            return final[0]["summary_text"]
        except Exception:
            return combined[:600] + "..."



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

    def process_items(self, items):
        """Process all RSS items and generate results"""
        results = defaultdict(dict)

        for idx, item in enumerate(items, 1):
            score = self.calculate_score(item["title"], item["description"])
            
            if score < 2:
                continue

            print(f"Processing item {idx}: {item['title'][:60]}... (score: {score})")

            pdf_links = self.extract_pdf_links(item["link"])

            if pdf_links:
                try:
                    print(f"  Extracting from PDF: {pdf_links[0]}")
                    text = self.extract_text_from_pdf(pdf_links[0])
                    full_text = text
                    extracted_data = self.extractor.extract_document_fields(full_text)
                    xml_content = self.pdf_to_xml(pdf_links[0], item["title"])
                    print(f"  Extracted {len(text)} characters")
                    summary = self.summarize_text(text)
                    print(f"  Summary generated: {len(summary)} characters")
                except Exception as e:
                    print(f"  Error processing PDF: {e}")
                    summary = f"Error processing PDF: {str(e)}"
            else:
                print(f"  No PDF found")
                summary = "No PDF available"

            results[item["pub_date"]][f"item_{idx}"] = {
                "score": score,
                "title": item["title"],
                "publish_date": item["pub_date"],
                "link": item["link"],
                "pdf_links": pdf_links,
                "summary": summary,
                "extracted_metadata": extracted_data,
                "full_text": full_text,
                "xml_content": xml_content
            }

        return dict(results)

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
        
        print("\n[3/3] Processing items...")
        results = self.process_items(items)
        
        return results


# =======================
# ENTRY POINT
# =======================

if __name__ == "__main__":
    parser = SEBIRSSParser()
    results = parser.run()

    if results:
        output_file = "sebi_results.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print("\n" + "="*70)
        print(f"Results saved to {output_file}")
        print(f"Relevant items: {sum(len(v) for v in results.values())}")
        print("="*70)
    else:
        print("\n No results")