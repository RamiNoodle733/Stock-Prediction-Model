import os
from fpdf import FPDF
from PIL import Image

class PDFReport(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'Stock Prediction Report', align='C', ln=True)
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', align='C')

def add_image(pdf, image_path, title):
    if os.path.exists(image_path):
        pdf.add_page()
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, title, ln=True, align='C')
        pdf.ln(10)
        pdf.image(image_path, x=10, y=30, w=190)
    else:
        print(f"Image not found: {image_path}")

def generate_pdf():
    pdf = PDFReport()

    # Add stock price visualizations
    stocks = ['AAPL', 'AMZN', 'GOOGL', 'MSFT', 'TSLA']
    for stock in stocks:
        stock_price_path = f'../results/multiple/{stock}/{stock}_stock_price.png'
        add_image(pdf, stock_price_path, f'{stock} Stock Price')

    # Add model comparison
    model_comparison_path = '../results/multiple/stock_model_comparison.png'
    add_image(pdf, model_comparison_path, 'Model Comparison')

    # Add feature importance
    feature_importance_path = '../results/AAPL_feature_importance.png'
    add_image(pdf, feature_importance_path, 'Feature Importance (AAPL)')

    # Add ablation study
    ablation_study_path = '../results/AAPL_ablation_study.png'
    add_image(pdf, ablation_study_path, 'Ablation Study (AAPL)')

    # Save the PDF
    output_path = '../results/Stock_Prediction_Report.pdf'
    pdf.output(output_path)
    print(f'Report generated: {output_path}')

if __name__ == '__main__':
    generate_pdf()