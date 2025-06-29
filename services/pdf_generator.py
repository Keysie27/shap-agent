from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, ListFlowable, ListItem
from reportlab.lib import colors
import os
import io
from io import BytesIO
import base64
from PIL import Image as PILImage

def decode_base64_image(base64_string):
    img_data = base64.b64decode(base64_string)
    return io.BytesIO(img_data)

def create_shap_report_pdf(
    output_path,
    title="SHAP-Agent Report",
    shap_summary_img_base64=None,
    bar_chart_img_base64=None,
    top_influencers_sentence="",
    feature_analysis_points=[],
    key_observations_points=[],
    practical_recommendations=[],
):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    doc = SimpleDocTemplate(output_path, pagesize=letter,
                            rightMargin=72, leftMargin=72,
                            topMargin=72, bottomMargin=72)
    
    styles = getSampleStyleSheet()
    story = []
    
    # Title
    story.append(Paragraph(title, styles['Title']))
    story.append(Spacer(1, 12))
    
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
        story.append(Spacer(1, 20))
    
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
    

