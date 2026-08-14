import os
from fpdf import FPDF
import random

class PDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 12)
        self.cell(0, 10, "STRICTLY CONFIDENTIAL - MERGER AND ACQUISITION AGREEMENT", border=0, align="C", new_x="LMARGIN", new_y="NEXT")

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")

def generate_complex_pdf(output_path="local_rag/complex_ma_agreement.pdf"):
    pdf = PDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", size=10)

    # 1. Title Page and Core Definitions
    text = """
AGREEMENT AND PLAN OF MERGER
by and among
NEXUS GLOBAL HOLDINGS INC.,
ASTRAL ACQUISITION SUB LLC,
and
QUANTUM DYNAMICS LLC

Dated as of August 11, 2026

ARTICLE I
THE MERGER
1.1 The Merger. Upon the terms and subject to the conditions set forth in this Agreement, and in accordance with the Delaware Limited Liability Company Act (the "DLLCA"), at the Effective Time, Astral Acquisition Sub LLC ("Merger Sub") shall be merged with and into Quantum Dynamics LLC (the "Company"). Following the Merger, the separate corporate existence of Merger Sub shall cease, and the Company shall continue as the surviving company (the "Surviving Company") and shall succeed to and assume all the rights and obligations of Merger Sub in accordance with the DLLCA.

1.2 Closing. The closing of the Merger (the "Closing") shall take place at 10:00 a.m., Eastern Time, on October 15, 2026 (the "Closing Date"), provided that all conditions set forth in Article VII have been satisfied or waived.

1.3 Effective Time. Subject to the provisions of this Agreement, as soon as practicable on the Closing Date, the parties shall file a certificate of merger with the Secretary of State of the State of Delaware.

ARTICLE II
EFFECT ON THE CAPITAL STOCK OF THE CONSTITUENT CORPORATIONS
2.1 Conversion of Securities. The aggregate consideration to be paid by Nexus Global Holdings Inc. ("Parent") for all outstanding equity interests of the Company shall be $1,450,000,000 (One Billion Four Hundred Fifty Million Dollars) (the "Merger Consideration").
The Merger Consideration shall be paid in cash, minus any applicable Debt, plus any Estimated Working Capital Adjustment.

2.2 Break-Up Fee. In the event that this Agreement is terminated by the Company due to a superior alternative proposal, the Company shall pay to Parent a break-up fee of $50,000,000 (Fifty Million Dollars) within two (2) business days of such termination.

ARTICLE III
REPRESENTATIONS AND WARRANTIES OF THE COMPANY
3.1 Organization. The Company is a limited liability company duly organized, validly existing, and in good standing under the laws of the State of Delaware.
3.2 Capitalization. The authorized equity of the Company consists of 10,000,000 Class A Units.
3.3 Intellectual Property. The Company owns all right, title, and interest in the "Aegis Core" software patents, without any encumbrances.

ARTICLE IV
INDEMNIFICATION
4.1 Survival. The representations and warranties of the Company contained in this Agreement shall survive the Closing for a period of eighteen (18) months, except for Fundamental Representations which shall survive indefinitely.
4.2 Indemnification by the Sellers. The Sellers shall indemnify and hold harmless the Parent Indemnitees against any Losses arising out of (a) any breach of representation or warranty made by the Company, and (b) any pre-Closing Taxes.
"""
    pdf.multi_cell(0, 5, text)
    
    # Generate 45 pages of "legalese" filler to make the document complex and test the chunker's retrieval capabilities
    legal_jargon_snippets = [
        "The foregoing notwithstanding, nothing in Section {sec} shall be construed to limit the obligations under Section {sec2}.",
        "Subject to the limitations set forth in the Disclosure Schedule, the parties agree that standard commercial reasonableness shall apply.",
        "Any disputes arising from the interpretation of Section {sec} shall be subject to mandatory arbitration in Wilmington, Delaware.",
        "The term 'Material Adverse Effect' shall not include any changes in macroeconomic conditions generally affecting the industry.",
        "Each party shall bear its own costs and expenses incurred in connection with the negotiation, preparation, and execution of this Agreement.",
        "Except as otherwise expressly provided herein, all remedies available under this Agreement are cumulative and not exclusive.",
        "No waiver by any party of any default, misrepresentation, or breach of warranty or covenant hereunder shall be deemed to extend to any prior or subsequent default."
    ]
    
    for i in range(5, 50):
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 10, f"ARTICLE {i}: ADDITIONAL COVENANTS AND STIPULATIONS", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", size=10)
        
        page_text = ""
        for _ in range(40):
            snippet = random.choice(legal_jargon_snippets).format(sec=f"{i}.{random.randint(1,9)}", sec2=f"{i+1}.{random.randint(1,9)}")
            page_text += snippet + " "
        
        pdf.multi_cell(0, 5, page_text)

    # Insert one more highly specific fact on page 50
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 10, "ARTICLE 50: SPECIAL REGULATORY CONDITIONS", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", size=10)
    pdf.multi_cell(0, 5, "50.1 CFIUS Approval. The consummation of the Merger is strictly conditioned upon receiving written clearance from the Committee on Foreign Investment in the United States (CFIUS) no later than September 1, 2026. If clearance is not obtained by this date, either party may terminate the Agreement without penalty.")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pdf.output(output_path)
    print(f"Generated {output_path} successfully!")

if __name__ == "__main__":
    generate_complex_pdf()
