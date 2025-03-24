
import os
import json
from pdf2image import convert_from_path

path = "selected_paths.json"

with open(path, "r") as json_file:
    paths_data = json.load(json_file)

# Extract the PDF path
pdf_path = paths_data.get("pdf_path", "")
output_folder = paths_data.get("folder_path", "")

if os.path.exists(pdf_path):
    print(f"PDF to Image: File found: {pdf_path}")
else:
    print(f"PDF to Image: File NOT found: {pdf_path}")

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)
print(f"Pdf to Image: Converting PDF to images. Saving at {output_folder}")
# Convert PDF to images
try:
    images = convert_from_path(pdf_path, dpi=300)
except Exception as e:
    print(f"Error: {e}")
    exit(1)

# Get total number of pages
total_pages = len(images)

# Save images & create metadata
for i, img in enumerate(images):
    img_filename = os.path.join(output_folder, f"page_{i+1}.jpg")
    img.save(img_filename, "JPEG")
    print(f"Saved: {img_filename}")

# Save total page count to metadata.json
metadata = {"totalPages": total_pages}
metadata_path = os.path.join(output_folder, "metadata.json")
with open(metadata_path, "w") as f:
    json.dump(metadata, f)

print(f"Pdf To Image: Conversion complete. Metadata saved to {metadata_path}")