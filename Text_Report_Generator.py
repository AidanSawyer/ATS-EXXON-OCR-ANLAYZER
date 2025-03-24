'''
Aidan Sawyer

Generates the text report (with Section Intervention scores)
Takes in extracted_data.h5 after Text_Analyzer.py is ran
saves pdf as text_report.pdf
'''

import os
import cv2
import json
import h5py
import numpy as np
import io
from PIL import Image
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

def generate_pdf_report(extracted_data, images_path, output_pdf="report.pdf"):
    """
    Generates a PDF report where each page contains:
      1. A JPG image (resized to 750x900) with section bounding boxes drawn in yellow.
         Each section box has a small red label (the section number) positioned 
         just outside (to the left) of the box.
      2. A list of sections and their corresponding section_intervention_score, printed
         on the right side of the image, with a heading.
      3. A header displaying the page number (e.g., "Page 1") at the top of each page.
    """
    # Determine all page numbers in our data
    pages = sorted({word["page"] for word in extracted_data})
    
    # Set dimensions for the report page:
    img_width, img_height = 750, 900
    header_height = 40  # Extra space at the top for the header
    text_area_width = 300
    page_width = img_width + text_area_width
    page_height = img_height + header_height

    # Create a ReportLab canvas with the new page size
    c = canvas.Canvas(output_pdf, pagesize=(page_width, page_height))
    
    for page in pages:
        image_filename = os.path.join(images_path, f"page_{page}.jpg")
        if not os.path.exists(image_filename):
            print(f"⚠️  Image for page {page} not found, skipping.")
            continue
        
        # Load and resize image
        image = cv2.imread(image_filename)
        if image is None:
            print(f"⚠️  Could not load image for page {page}, skipping.")
            continue

        orig_height, orig_width, _ = image.shape
        image_resized = cv2.resize(image, (img_width, img_height), interpolation=cv2.INTER_AREA)
        
        words_on_page = [word for word in extracted_data if word["page"] == page]
        if not words_on_page:
            print(f"⚠️  No OCR data for page {page}, skipping.")
            continue
        
        # Scaling factors (from original image to resized image)
        scale_x = img_width / orig_width
        scale_y = img_height / orig_height
        
        # Compute section bounding boxes and store a section intervention score per section.
        section_boxes = {}   # section_id -> [xmin, ymin, xmax, ymax]
        section_scores = {}  # section_id -> section_intervention_score
        for word_data in words_on_page:
            section_id = word_data["section"]
            (xmin, ymin), (xmax, ymax) = word_data["bounding_box"]
            xmin_scaled = int(xmin * scale_x)
            ymin_scaled = int(ymin * scale_y)
            xmax_scaled = int(xmax * scale_x)
            ymax_scaled = int(ymax * scale_y)
            
            if section_id not in section_boxes:
                section_boxes[section_id] = [xmin_scaled, ymin_scaled, xmax_scaled, ymax_scaled]
                section_scores[section_id] = word_data["section_intervention_score"]
            else:
                section_boxes[section_id][0] = min(section_boxes[section_id][0], xmin_scaled)
                section_boxes[section_id][1] = min(section_boxes[section_id][1], ymin_scaled)
                section_boxes[section_id][2] = max(section_boxes[section_id][2], xmax_scaled)
                section_boxes[section_id][3] = max(section_boxes[section_id][3], ymax_scaled)
        
        # Draw section boxes on the resized image.
        # Yellow for the box (BGR: (0, 255, 255)) and red for the label (BGR: (0, 0, 255)).
        for section_id, (sxmin, symin, sxmax, symax) in section_boxes.items():
            cv2.rectangle(image_resized, (sxmin, symin), (sxmax, symax), (0, 255, 255), 3)
            # Shift the label 20 pixels left and 5 pixels upward from the top-left corner
            offset = 20
            label_x = max(sxmin - offset, 0)
            label_y = max(symin - 5, 0) + 15  # adding 15 so that the text is legible
            cv2.putText(image_resized, str(section_id), (label_x, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        
        # Convert the image from BGR to RGB and then to a PIL image.
        image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        img_reader = ImageReader(pil_image)
        
        # Draw the header with the page number in the reserved header space.
        c.setFont("Helvetica-Bold", 16)
        c.setFillColorRGB(0, 0, 0)
        header_y = img_height + (header_height / 2)
        c.drawString(10, header_y, f"Page {page}")
        
        # Draw the image on the left side of the page, below the header.
        c.drawImage(img_reader, 0, 0, width=img_width, height=img_height)
        
        # Draw the intervention scores on the right side.
        text_x = img_width + 10
        text_y = img_height - 30  # Start near the top of the image area
        
        # Print heading for the section intervention scores list.
        c.setFont("Helvetica-Bold", 12)
        c.drawString(text_x, text_y, "Section Human Intervention Scores (0-100)")
        text_y -= 20
        
        # Now print each section's score, rounding to the nearest integer.
        c.setFont("Helvetica", 12)
        for section_id, score in sorted(section_scores.items()):
            text_line = f"Section {section_id}: {int(round(score))}"
            c.drawString(text_x, text_y, text_line)
            text_y -= 20  # Adjust spacing as needed
        
        c.showPage()  # End the current page
    
    c.save()
    print(f"Text Report Generator. PDF report generated and saved at {output_pdf}")


script_directory = os.path.dirname(os.path.abspath(__file__))

# Load the image folder path from the JSON file
path = os.path.join(script_directory, "selected_paths.json")
with open(path, "r") as json_file:
    paths_data = json.load(json_file)

image_folder = paths_data.get("folder_path", "")

# Load the extracted_data from the HDF5 file (similar to your GUI code)
h5_file_path = os.path.join(image_folder, "extracted_data.h5")

print("Text Report Generator: Generating Report...")
extracted_data = []
with h5py.File(h5_file_path, "r") as hdf:
    for item in hdf.keys():
        group = hdf[item]
        data = {}
        for key in group.attrs:
            data[key] = group.attrs[key]
        for key in group:
            data[key] = np.array(group[key])
        extracted_data.append(data)



pdf_path = os.path.join(image_folder, "text_report.pdf")
generate_pdf_report(extracted_data, image_folder, pdf_path)
