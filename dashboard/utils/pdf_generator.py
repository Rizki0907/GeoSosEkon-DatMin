from fpdf import FPDF
import pandas as pd
from pathlib import Path
from datetime import datetime
import os

class ProvincialReport(FPDF):
    def __init__(self, province):
        super().__init__()
        self.province = province
        
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.set_text_color(44, 62, 80)
        self.cell(0, 10, 'GeoSosEkon - Provincial Poverty Report', 0, 1, 'C')
        self.set_font('Arial', 'I', 10)
        self.set_text_color(127, 140, 141)
        self.cell(0, 5, f'Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M")}', 0, 1, 'C')
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.set_text_color(127, 140, 141)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

def generate_pdf(province_name, data_dict):
    pdf = ProvincialReport(province_name)
    pdf.add_page()
    
    # Title
    pdf.set_font('Arial', 'B', 20)
    pdf.set_text_color(52, 152, 219)
    pdf.cell(0, 15, province_name.upper(), 0, 1, 'C')
    pdf.ln(5)
    
    # 1. Macro Overview (BPS Data)
    pdf.set_font('Arial', 'B', 14)
    pdf.set_text_color(44, 62, 80)
    pdf.cell(0, 10, '1. Macro Overview (Latest Year)', 0, 1, 'L')
    pdf.set_font('Arial', '', 12)
    pdf.set_text_color(0, 0, 0)
    
    bps_data = data_dict.get('bps_data', {})
    pdf.cell(0, 8, f"- Poverty Rate : {bps_data.get('poverty_rate', 'N/A')}%", 0, 1, 'L')
    pdf.cell(0, 8, f"- HDI : {bps_data.get('hdi', 'N/A')}", 0, 1, 'L')
    pdf.cell(0, 8, f"- Open Unemployment (TPT) : {bps_data.get('tpt', 'N/A')}%", 0, 1, 'L')
    pdf.ln(5)
    
    # 2. Provincial Typology & Spatial
    pdf.set_font('Arial', 'B', 14)
    pdf.set_text_color(44, 62, 80)
    pdf.cell(0, 10, '2. Spatial & Typology Profile', 0, 1, 'L')
    pdf.set_font('Arial', '', 12)
    pdf.set_text_color(0, 0, 0)
    
    pdf.cell(0, 8, f"- Typology Cluster : {data_dict.get('cluster', 'N/A')}", 0, 1, 'L')
    pdf.cell(0, 8, f"- Spatial Association (LISA) : {data_dict.get('lisa_quadrant', 'N/A')}", 0, 1, 'L')
    pdf.ln(5)
    
    # 3. Poverty Forecast (2025-2026)
    pdf.set_font('Arial', 'B', 14)
    pdf.set_text_color(44, 62, 80)
    pdf.cell(0, 10, '3. Poverty Forecast (2025-2026)', 0, 1, 'L')
    pdf.set_font('Arial', '', 12)
    pdf.set_text_color(0, 0, 0)
    
    forecast = data_dict.get('forecast', {})
    pdf.cell(0, 8, f"- Forecast 2025 : {forecast.get('2025', 'N/A')}%", 0, 1, 'L')
    pdf.cell(0, 8, f"- Forecast 2026 : {forecast.get('2026', 'N/A')}%", 0, 1, 'L')
    pdf.ln(5)
    
    # 4. Regional Sentiment
    pdf.set_font('Arial', 'B', 14)
    pdf.set_text_color(44, 62, 80)
    pdf.cell(0, 10, '4. Regional Public Sentiment Context', 0, 1, 'L')
    pdf.set_font('Arial', '', 12)
    pdf.set_text_color(0, 0, 0)
    
    sentiment = data_dict.get('sentiment', 'No specific regional sentiment data mapped. Assuming global trends.')
    pdf.multi_cell(0, 8, sentiment)
    
    # Generate Output
    try:
        os.makedirs("temp", exist_ok=True)
        out_path = f"temp/Report_{province_name.replace(' ', '_')}.pdf"
        pdf.output(out_path)
        return out_path
    except Exception as e:
        print(f"PDF Gen Error: {e}")
        return None
