import re
import spacy
from datetime import datetime

class DocumentExtractor:
    def __init__(self):
        """Initialize the DocumentExtractor with spacy model"""
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Spacy model not found. Installing...")
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
            self.nlp = spacy.load("en_core_web_sm")
        
        # Define reference texts for comparison
        self.reference_adjudication = """
        Adjudication Order, Order Number, Noticee, PAN, Exchange, BSE, NSE,
        Violation, Trading, PFUTP Regulations, Investigation Period, 
        Adjudicating Officer, Strike Off, Dissolved, Final Outcome, Dispose,
        Section 15-I, Section 15HA, Show Cause Notice, SCN, Penalty
        """
        
        self.reference_circular = """
        Circular, Technical Glitch, Stock Brokers, Stock Exchanges, Framework,
        Capacity Planning, Software Testing, BCP, DRS, Disaster Recovery,
        Reporting Requirements, LAMA, Monitoring Mechanism, IBT, STWT,
        Samuhik Prativedan Manch, Common Reporting Platform, 10000 clients,
        Financial Disincentive, Compliance, SEBI Circular
        """

    def detect_document_type(self, full_text):
        """Detect if document is Adjudication Order or Circular"""
        adjudication_keywords = ['adjudication order', 'noticee', 'penalty', 'violation', 'scn']
        circular_keywords = ['circular', 'technical glitch', 'framework', 'capacity planning']
        
        text_lower = full_text.lower()
        
        adj_score = sum(1 for kw in adjudication_keywords if kw in text_lower)
        circ_score = sum(1 for kw in circular_keywords if kw in text_lower)
        
        if adj_score > circ_score:
            return "Adjudication Order"
        elif circ_score > adj_score:
            return "Circular"
        else:
            return "Unknown"

    def extract_common_fields(self, full_text, doc):
        """Extract fields common to both document types"""
        common = {
            "sebi_reference": None,
            "date": None,
            "location": None,
            "stock_exchanges_mentioned": [],
            "stock_brokers_mentioned": False,
            "regulatory_framework": [],
            "mumbai_reference": False
        }
        
        # SEBI references
        sebi_refs = re.findall(r'SEBI[/\w\-]+', full_text)
        if sebi_refs:
            common["sebi_reference"] = sebi_refs[0]
        
        # Dates
        date_matches = re.findall(
            r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b',
            full_text
        )
        if date_matches:
            common["date"] = date_matches[0]
        
        # Location
        if "Mumbai" in full_text:
            common["location"] = "Mumbai"
            common["mumbai_reference"] = True
        
        # Stock exchanges
        exchanges = []
        if "BSE" in full_text:
            exchanges.append("BSE")
        if "NSE" in full_text:
            exchanges.append("NSE")
        if "Stock Exchange" in full_text or "stock exchange" in full_text:
            exchanges.append("Stock Exchange (Generic)")
        common["stock_exchanges_mentioned"] = list(set(exchanges))
        
        # Stock brokers
        if "stock broker" in full_text.lower() or "stockbroker" in full_text.lower():
            common["stock_brokers_mentioned"] = True
        
        # Regulatory framework
        for ent in doc.ents:
            if ent.label_ == "LAW":
                common["regulatory_framework"].append(ent.text)
        
        return common

    def extract_adjudication_exclusive_fields(self, full_text, doc):
        """Extract fields exclusive to Adjudication Orders"""
        exclusive = {
            "order_number": None,
            "noticee": None,
            "noticee_pan": None,
            "violation_type": None,
            "investigation_period": None,
            "adjudicating_officer": None,
            "adjudicating_officer_signature": None,
            "final_outcome": None,
            "decision_date": None,
            "penalty_amount": None,
            "regulations_violated": [],
            "company_status": None,
            "show_cause_notice_date": None,
            "proceedings_disposed": False
        }
        
        # Order number
        order_match = re.search(r"Order[/\w\-]+", full_text)
        if order_match:
            exclusive["order_number"] = order_match.group()
        
        # PAN
        pan_match = re.search(r"PAN[:\s–]+([A-Z]{5}[0-9]{4}[A-Z])", full_text)
        if pan_match:
            exclusive["noticee_pan"] = pan_match.group(1)
        
        # Company status
        if "Strike Off" in full_text or "struck off" in full_text.lower():
            exclusive["company_status"] = "Struck Off"
        elif "dissolved" in full_text.lower():
            exclusive["company_status"] = "Dissolved"
        
        # Entities
        for ent in doc.ents:
            if ent.label_ == "ORG":
                if "Private Limited" in ent.text or "Limited" in ent.text:
                    if not exclusive["noticee"]:
                        exclusive["noticee"] = ent.text
            
            elif ent.label_ == "PERSON":
                # Check if mentioned with Officer context
                sent_text = ent.sent.text if hasattr(ent, 'sent') else ""
                if "Officer" in sent_text or "Adjudicating" in sent_text:
                    exclusive["adjudicating_officer"] = ent.text
            
            elif ent.label_ == "DATE":
                sent_text = ent.sent.text if hasattr(ent, 'sent') else ""
                if "Show Cause" in sent_text:
                    exclusive["show_cause_notice_date"] = ent.text
                elif "Date" in sent_text or "signed" in sent_text.lower():
                    exclusive["decision_date"] = ent.text
        
        # Violation type
        violation_patterns = [
            r"trading in ([^.]+)",
            r"matter of ([^.]+)",
            r"violation of ([^.]+)"
        ]
        for pattern in violation_patterns:
            match = re.search(pattern, full_text, re.IGNORECASE)
            if match:
                exclusive["violation_type"] = match.group(1).strip()
                break
        
        # Regulations violated
        reg_matches = re.findall(r"Regulation[s]?\s+\d+\([a-z]\)", full_text, re.IGNORECASE)
        exclusive["regulations_violated"] = list(set(reg_matches))
        
        # Investigation period
        period_match = re.search(
            r"(?:period|IP)[:\s]+([A-Z][a-z]+\s+\d{1,2},?\s+\d{4}\s+to\s+[A-Z][a-z]+\s+\d{1,2},?\s+\d{4})",
            full_text
        )
        if period_match:
            exclusive["investigation_period"] = period_match.group(1)
        
        # Final outcome
        if "dispose" in full_text.lower() or "disposed" in full_text.lower():
            exclusive["proceedings_disposed"] = True
            for sent in doc.sents:
                if "dispose" in sent.text.lower():
                    exclusive["final_outcome"] = sent.text.strip()
                    break
        
        # Penalty amount
        penalty_match = re.search(r"(?:penalty|fine)[:\s]+(?:Rs\.?|INR)?\s*([\d,]+)", full_text, re.IGNORECASE)
        if penalty_match:
            exclusive["penalty_amount"] = penalty_match.group(1)
        
        return exclusive

    def extract_circular_exclusive_fields(self, full_text, doc):
        """Extract fields exclusive to Circulars"""
        exclusive = {
            "circular_number": None,
            "circular_title": None,
            "supersedes_circular": None,
            "effective_date": None,
            "technical_glitch_mentioned": False,
            "client_threshold": None,
            "reporting_time_limit": None,
            "common_reporting_platform": None,
            "capacity_planning_required": False,
            "disaster_recovery_required": False,
            "software_testing_required": False,
            "financial_disincentive_mentioned": False,
            "exemptions_provided": [],
            "applicability_criteria": [],
            "key_modifications": [],
            "contact_person": None,
            "contact_email": None,
            "contact_phone": None
        }
        
        # Circular number
        circ_match = re.search(r"CIRCULAR\s+[\w/\-()]+", full_text)
        if circ_match:
            exclusive["circular_number"] = circ_match.group()
        
        # Supersedes
        supersede_match = re.search(r"supersede[s]?\s+(?:earlier\s+)?(?:SEBI\s+)?circular\s+no[.\s]+([\w/\-]+)", full_text, re.IGNORECASE)
        if supersede_match:
            exclusive["supersedes_circular"] = supersede_match.group(1)
        
        # Effective date
        effective_match = re.search(r"come into effect from\s+([A-Z][a-z]+\s+\d{1,2},?\s+\d{4})", full_text)
        if effective_match:
            exclusive["effective_date"] = effective_match.group(1)
        
        # Technical glitch
        if "technical glitch" in full_text.lower():
            exclusive["technical_glitch_mentioned"] = True
        
        # Client threshold
        threshold_match = re.search(r"(\d+[,\d]*)\s+registered clients", full_text)
        if threshold_match:
            exclusive["client_threshold"] = threshold_match.group(1).replace(',', '')
        
        # Reporting time
        time_match = re.search(r"within\s+(\d+)\s+hours?", full_text)
        if time_match:
            exclusive["reporting_time_limit"] = f"{time_match.group(1)} hours"
        
        # Common reporting platform
        if "Samuhik Prativedan Manch" in full_text or "Common Reporting Platform" in full_text:
            exclusive["common_reporting_platform"] = "Samuhik Prativedan Manch"
        
        # Requirements
        if "capacity planning" in full_text.lower():
            exclusive["capacity_planning_required"] = True
        
        if "disaster recovery" in full_text.lower() or "DRS" in full_text or "DR site" in full_text:
            exclusive["disaster_recovery_required"] = True
        
        if "software testing" in full_text.lower():
            exclusive["software_testing_required"] = True
        
        if "financial disincentive" in full_text.lower():
            exclusive["financial_disincentive_mentioned"] = True
        
        # Exemptions
        exemption_patterns = [
            r"exempted from ([^.]+)",
            r"shall not be considered as ([^.]+)",
            r"not applicable to ([^.]+)"
        ]
        for pattern in exemption_patterns:
            matches = re.findall(pattern, full_text, re.IGNORECASE)
            exclusive["exemptions_provided"].extend(matches)
        
        # Contact information
        email_match = re.search(r"Email ID:\s*([\w.]+@[\w.]+)", full_text)
        if email_match:
            exclusive["contact_email"] = email_match.group(1)
        
        phone_match = re.search(r"Tel\.? No:\s*([\d\s]+)", full_text)
        if phone_match:
            exclusive["contact_phone"] = phone_match.group(1).strip()
        
        # Contact person
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                sent_text = ent.sent.text if hasattr(ent, 'sent') else ""
                if "General Manager" in sent_text or "Manager" in sent_text:
                    exclusive["contact_person"] = ent.text
        
        return exclusive

    def extract_document_fields(self, full_text):
        """Main extraction function that handles both document types"""
        try:
            doc = self.nlp(full_text)
            
            # Detect document type
            doc_type = self.detect_document_type(full_text)
            
            # Extract common fields
            common_fields = self.extract_common_fields(full_text, doc)
            
            # Extract exclusive fields based on document type
            if doc_type == "Adjudication Order":
                exclusive_fields = self.extract_adjudication_exclusive_fields(full_text, doc)
            elif doc_type == "Circular":
                exclusive_fields = self.extract_circular_exclusive_fields(full_text, doc)
            else:
                exclusive_fields = {}
            
            # Combine all data
            data = {
                "document_type": doc_type,
                "common_fields": common_fields,
                "exclusive_fields": exclusive_fields
            }
            
            return data
        
        except Exception as e:
            return {
                "document_type": "Error",
                "common_fields": {},
                "exclusive_fields": {},
                "error": str(e)
            }