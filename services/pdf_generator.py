from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak, Table, ListFlowable, ListItem, TableStyle
from reportlab.lib import colors
import os
import io
from io import BytesIO
import base64
from PIL import Image as PILImage

def decode_base64_image(base64_string):
    img_data = base64.b64decode(base64_string)
    return io.BytesIO(img_data)

def compute_column_widths(table_data, page_width, min_col_width=40, max_col_width=120):
    num_cols = len(table_data[0])
    raw_width = page_width / num_cols
    col_width = max(min(raw_width, max_col_width), min_col_width)
    return [col_width] * num_cols

def wrap_table_data(data):
    style = getSampleStyleSheet()['BodyText']
    return [[Paragraph(str(cell), style) for cell in row] for row in data]

def round_numeric_cells(table_data, decimals=2):
    rounded_data = []
    for i, row in enumerate(table_data):
        if i == 0:
            rounded_data.append(row)
        else:
            new_row = []
            for cell in row:
                if isinstance(cell, (float, int)):
                    new_row.append(round(cell, decimals))
                else:
                    new_row.append(cell)
            rounded_data.append(new_row)
    return rounded_data

def create_shap_report_pdf(
    title="SHAP-Agent Report",
    shap_summary_img_base64=None,
    bar_chart_img_base64=None,
    top_influencers_sentence="",
    feature_analysis_points=[],
    key_observations_points=[],
    practical_recommendations=[],
    sample_data=None,
    shap_values=None,
    waterfall_img_base64=None
):
    buffer = BytesIO()
    
    doc = SimpleDocTemplate(buffer, pagesize=letter,
                            rightMargin=72, leftMargin=72,
                            topMargin=72, bottomMargin=72)
    
    styles = getSampleStyleSheet()
    story = []

    # Title
    story.append(Paragraph(title, styles['Title']))
    story.append(Spacer(1, 20))

    # User Data
    story.append(Paragraph("Input Data", styles['Heading2']))
    story.append(Paragraph("Below is a preview of the dataset provided by the user, showing the first five rows and six features. This gives an overview of the features and their values fed into the model.", styles['BodyText']))
    story.append(Spacer(1, 12))

    if sample_data:
        max_cols = 6

        col_count = len(sample_data[0]) if sample_data else 0

        sample_data = [row[:max_cols] for row in sample_data]

        page_width = letter[0] - 72*2 
        col_widths = compute_column_widths(sample_data, page_width)

        wrapped_data = wrap_table_data(sample_data)
        table = Table(wrapped_data, colWidths=col_widths, hAlign='LEFT')
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),  
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),      
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),             
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),       
            ('FONTSIZE', (0, 0), (-1, -1), 6), 
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),    
        ]))
        story.append(table)

        story.append(Paragraph(f"Num. of features: {col_count}", styles['BodyText']))

        story.append(Spacer(1, 30))

    # Raw SHAP Values
    story.append(Paragraph("Raw SHAP Values", styles['Heading2']))
    story.append(Paragraph("This table shows the raw SHAP values computed for these initial samples. Each value indicates how much the corresponding feature contributes to the model’s prediction for that sample.", styles['BodyText']))
    story.append(Spacer(1, 12))

    if shap_values is not None:
        shap_values = [shap_values.columns.to_list()] + shap_values.head(5).values.tolist()

        max_cols = 6
        header = shap_values[0][:max_cols]
        rows = [row[:max_cols] for row in shap_values[1:]]
        trimmed_data = [header] + rows
        trimmed_data = round_numeric_cells(trimmed_data, decimals=4)

        page_width = letter[0] - 72*2
        col_widths = compute_column_widths(trimmed_data, page_width)

        wrapped_data = wrap_table_data(trimmed_data)
        table = Table(wrapped_data, colWidths=col_widths, hAlign='LEFT')
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.whitesmoke),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 6),
            ('GRID', (0, 0), (-1, -1), 0.25, colors.grey),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
        ]))

        story.append(table)
        story.append(PageBreak())


    # Visual Summary
    story.append(Paragraph("Visual Summary", styles['Heading2']))
    story.append(Paragraph("The SHAP plots below highlight how features influence the model’s predictions.", styles['BodyText']))
    story.append(Spacer(1, 12))
    
    # SHAP Summary Image
    if shap_summary_img_base64:
        img_buffer = decode_base64_image(shap_summary_img_base64)
        story.append(Image(img_buffer, width=5*inch, height=4*inch))
    
    # Top Features
    story.append(Paragraph("Top Features", styles['Heading2']))
    story.append(Paragraph("The 5 most impactful features are shown below.", styles['BodyText']))
    
    # Bar Chart Image
    if bar_chart_img_base64:
        img_buffer = decode_base64_image(bar_chart_img_base64)
        story.append(Image(img_buffer, width=5*inch, height=2.5*inch))
    
    story.append(PageBreak())

    # Dependence Plot
    story.append(Paragraph("Waterfall Plot", styles['Heading2']))
    story.append(Paragraph("The waterfall plot visualizes how each feature contributes to pushing the model's prediction from the base value to the final output for an individual instance, showing positive and negative impacts clearly.", styles['BodyText']))
    story.append(Spacer(1, 12))
    

    # Waterfall Image
    if waterfall_img_base64:
        img_buffer = decode_base64_image(waterfall_img_base64)
        story.append(Image(img_buffer, width=5*inch, height=4*inch))
    
    story.append(Spacer(1, 12))

    # Model Insights
    story.append(Paragraph("Model Insights", styles['Heading2']))
    
    # 1. Key Influencers
    story.append(Paragraph("1. Key Influencers", styles['Heading3']))
    story.append(Paragraph(top_influencers_sentence, styles['BodyText']))
    story.append(Spacer(1, 12))
    
    # 2. Feature Breakdown (bullet points)
    story.append(Paragraph("2. Feature Breakdown", styles['Heading3']))
    if feature_analysis_points:
        bullet_list = ListFlowable([
            ListItem(
                Paragraph(point, styles['BodyText']),
                spaceBefore=12
            ) 
            for point in feature_analysis_points],
            bulletType='bullet',
            start='circle'
        )
        story.append(bullet_list)
    story.append(Spacer(1, 12))
    
    # 3. Key Observations (bullet points)
    story.append(Paragraph("3. Observations", styles['Heading3']))
    if key_observations_points:
        bullet_list = ListFlowable([
            ListItem(
                Paragraph(observation, styles['BodyText']),
                spaceBefore=12 
            )
        for observation in key_observations_points],            
        bulletType='bullet',
        start='circle'
    )
        story.append(bullet_list)
    story.append(Spacer(1, 12))
    
    # 4. Recommendations (bullet points)
    story.append(Paragraph("4. Recommendations", styles['Heading3']))
    if practical_recommendations:
        rec_list = ListFlowable([
            ListItem(
                Paragraph(rec, styles['BodyText']),
                spaceBefore=12
            )
            for rec in practical_recommendations],
            bulletType='bullet',
            start='circle'
        )
        story.append(rec_list)
    story.append(Spacer(1, 12))
    
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()
