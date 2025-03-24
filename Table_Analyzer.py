'''
Aidan Sawyer

This program both analyzes table structual quality and generates the table report
It can detect two types of table errors - however depending on the type of table both of them might not be useful every time
The first error is a boudning box error - if a cell spans multiple rows of the table there is a merging error,
and if a cells were mapped to the same position of the image, we have a splitting error.

The second type is a "word split error". This one is more comprehensive but is only usefuly if the text is already clean
To find a wrod split error, we read the text in the original image, and try to guess using image processing techniques where 
the column boundaries of the tables are. Then, if see words that we think should be in the same column based on these column boundary guesses,
but they are in a different column in the OCR data, then we have a word split error.
This algorithm probably isn't perfect, but I think it can still be helpful in finding tables that need fixed.

After both of these errors are fond and visualized, we generate the report.
'''
import json
import os
import pandas as pd
from IPython.core.display import display, HTML
from IPython.display import Image
import matplotlib.pyplot as plt
import cv2
import sys
import numpy as np


#get the paths for the images and ocr data
script_directory = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_directory)
path = "selected_paths.json"

with open(path, "r") as json_file:
    paths_data = json.load(json_file)

ocr_json_path = paths_data.get("ocr_path", "")
image_folder = paths_data.get("folder_path", "")

with open(ocr_json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

print("Table Analyzer: Intitial Table Data Sucesfully Extracted")
# Extract tables from JSON
tables = data.get("analyzeResult", {}).get("tables", [])

# Default DPI fallback
default_dpi = 300

# List to store table JSON and corresponding cropped images
table_list = []

# Process each table and store its JSON and cropped image
for i, table in enumerate(tables):
    # Extract page number
    page_number = -1
    if "boundingRegions" in table and table["boundingRegions"]:
        page_number = table["boundingRegions"][0].get("pageNumber", -1)

    # Locate the corresponding page image
    image_path = os.path.join(image_folder, f"page_{page_number}.jpg")

    cropped_table = None  # Default value if no image is found

    if os.path.exists(image_path):
        # Load the full-page image
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB for proper display
        img_height, img_width, _ = img.shape  # Get image dimensions in pixels

        # Extract DPI and page dimensions from JSON
        dpi = default_dpi
        page_width_inches = None
        page_height_inches = None

        if "metadata" in data.get("analyzeResult", {}):
            metadata = data["analyzeResult"]["metadata"]
            dpi = metadata.get("dpi", dpi)  # Use metadata DPI if available
            page_width_inches = metadata.get("width", None)  # Width in inches
            page_height_inches = metadata.get("height", None)  # Height in inches

        # Compute DPI if not explicitly given
        if page_width_inches and page_height_inches:
            dpi_x = img_width / page_width_inches
            dpi_y = img_height / page_height_inches
            dpi = (dpi_x + dpi_y) / 2  # Average DPI for better accuracy

        INCH_TO_PIXELS = dpi  # Conversion factor

        # Extract the bounding box from JSON (stored as a polygon in inches)
        if "boundingRegions" in table and table["boundingRegions"]:
            polygon = table["boundingRegions"][0]["polygon"]  # Format: [x1, y1, x2, y2, ..., xN, yN]

            # Convert polygon from inches to pixels
            pixel_polygon = [(int(x * INCH_TO_PIXELS), int(y * INCH_TO_PIXELS)) for x, y in zip(polygon[0::2], polygon[1::2])]

            # Get bounding box coordinates
            x_coords = [p[0] for p in pixel_polygon]
            y_coords = [p[1] for p in pixel_polygon]
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)

            # Crop the table from the full-page image
            cropped_table = img[y_min:y_max, x_min:x_max]

    # Store table JSON and cropped image in the list
    table_list.append({
        "table_json": table,
        "cropped_image": cropped_table,
        "page_number": page_number
    })

print(f"Tabel Analyzer: Extracted {len(table_list)} tables successfully.")
if(len(table_list) == 0):
    print("No Tables were found in this file, so no Table Report will be generated")
    print("Exiting Table_Analyzer")
    sys.exit()
#Find all Bounding Box Merging/Splitting Errors:
def find_bounding_box_merging_splitting_errors(table_entry, table_index):
    """
    Visualizes merging and splitting errors on the cropped table image.
    Also updates the table_entry dictionary with:
      - "error_images": a list containing the visualization image.
      - "error_count": the total number of errors detected.

    Parameters:
        table_entry (dict): Dictionary containing table JSON and its cropped image.
        table_index (int): Index of the table in table_list (1-based indexing).

    Returns:
        None (updates table_entry in place).
    """
    table_json = table_entry["table_json"]
    cropped_image = table_entry["cropped_image"]

    if cropped_image is None:
        print(f"Table {table_index}: No image available for visualization.")
        table_entry["box_error_images"] = []
        table_entry["box_error_count"] = 0
        return

    num_rows = table_json["rowCount"]
    num_cols = table_json["columnCount"]
    cells = table_json.get("cells", [])

    # Make a copy of the image to draw on
    img_copy = cropped_image.copy()

    # Set colors for merged and split cells
    merged_color = (255, 0, 0)  # Red for merged cells
    split_color = (0, 0, 255)   # Blue for split cells

    # Counters for errors
    merged_count = 0
    split_count = 0

    # Lists to keep track of cells
    merged_cells = []
    split_cells = []

    for cell in cells:
        row_start = cell["rowIndex"]
        col_start = cell["columnIndex"]
        row_span = cell.get("rowSpan", 1)
        col_span = cell.get("columnSpan", 1)

        # Calculate approximate bounding box (assuming uniform spacing)
        x1 = int(col_start * (img_copy.shape[1] / num_cols))
        y1 = int(row_start * (img_copy.shape[0] / num_rows))
        x2 = int((col_start + col_span) * (img_copy.shape[1] / num_cols))
        y2 = int((row_start + row_span) * (img_copy.shape[0] / num_rows))

        if row_span > 1 or col_span > 1:
            # Merged cell: draw a rectangle and count it as an error.
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), merged_color, 2)
            merged_cells.append((row_start, col_start))
            merged_count += 1
        else:
            # Split cell: if the cell position is already recorded, mark the error.
            if (row_start, col_start) in split_cells:
                cv2.circle(img_copy, (int((x1 + x2) / 2), int((y1 + y2) / 2)), 5, split_color, -1)
                split_count += 1
            else:
                split_cells.append((row_start, col_start))

    # Total error count is the sum of merged and split errors.
    error_count = merged_count + split_count

    # Update the table_entry dictionary with the visualization and error count.
    table_entry["box_error_images"] = [img_copy]  # storing the image in a list
    table_entry["box_error_count"] = error_count


for idx, table in enumerate(table_list):
    find_bounding_box_merging_splitting_errors(table, idx + 1)  # 1-based index

print("Table Analyzer: Bounding Box Merging/Splitting Errors Analyzed")
print("Table Analyzer: Now Analyzing Word Merging/Splitting Errors")

#Need to find visual column boundaries now to do the rest of analysis.

def detect_and_draw_column_dividers(image, table_index, whitespace_threshold=30, blob_threshold=5, min_blob_size=10, save_path=None):
    """
    Detects column divisions in a table image by identifying large whitespace gaps,
    counts the number of black blobs (text regions) in each column (including left & right edges),
    and keeps merging sections until each has at least `blob_threshold` blobs.

    Parameters:
        image (numpy array): The cropped table image.
        table_index (int): Index of the table in table_list (1-based index).
        whitespace_threshold (int): Minimum width (in pixels) for a whitespace gap to be considered a column separator.
        blob_threshold (int): Minimum number of text blobs required to keep a divider.
        min_blob_size (int): Minimum size (in pixels) of a black region to be considered a valid text blob.
        save_path (str, optional): Path to save the processed image. If None, it won't save.

    Returns:
        list: Refined detected column divider positions in pixels.
    """
    if image is None:
        print(f"Error: Table {table_index} does not have a valid image.")
        return []

     # **Ensure Image Is Grayscale Before Processing**
    if len(image.shape) == 3:  # If the image has 3 channels (color), convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image  # Already grayscale

    # Get image dimensions
    height, width = gray.shape

    # Compute the average pixel intensity for each column
    column_intensity = np.mean(gray, axis=0)

    # Normalize values (scale between 0 and 255)
    column_intensity = (column_intensity - np.min(column_intensity)) / (np.max(column_intensity) - np.min(column_intensity)) * 255

    # Identify potential column dividers
    threshold = 220  # Define white space threshold (higher means more whitespace)
    is_white_space = column_intensity > threshold

    # Detect large continuous whitespace regions
    column_boundaries = []
    in_whitespace = False
    whitespace_start = 0

    for i in range(1, width - 1):
        if is_white_space[i] and not in_whitespace:
            # Entering a whitespace region
            whitespace_start = i
            in_whitespace = True
        elif not is_white_space[i] and in_whitespace:
            # Exiting a whitespace region
            whitespace_end = i
            in_whitespace = False
            
            # Only consider this a column boundary if the whitespace is wide enough
            whitespace_width = whitespace_end - whitespace_start
            if whitespace_width > whitespace_threshold:
                column_center = (whitespace_start + whitespace_end) // 2
                column_boundaries.append(column_center)

    # ** Ensure the leftmost and rightmost sections are included as columns **
    if column_boundaries and column_boundaries[0] > 0:
        column_boundaries.insert(0, 0)  # Add left boundary
    if column_boundaries and column_boundaries[-1] < width:
        column_boundaries.append(width)  # Add right boundary

    # ** Step 2: Convert image to binary (thresholding) **
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)  # Invert to make text white on black background

    # ** Step 3: Count black blobs (text regions) per column **
    def count_blobs(left, right):
        """ Counts the number of valid text blobs in a given column region. """
        column_region = binary[:, left:right]
        contours, _ = cv2.findContours(column_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_blobs = [cnt for cnt in contours if cv2.contourArea(cnt) > min_blob_size]
        return len(valid_blobs)

    # Initial blob counts
    blob_counts = [count_blobs(column_boundaries[i], column_boundaries[i + 1]) for i in range(len(column_boundaries) - 1)]
    
    #print(f"Initial column boundaries: {column_boundaries}")

    # ** Step 4: Keep Merging Low-Blob Columns Until Threshold is Met **
    while any(count < blob_threshold for count in blob_counts) and len(column_boundaries) > 2:
        new_boundaries = [column_boundaries[0]]  # Always keep the first boundary
        new_blob_counts = []
        i = 0
        merged = False

        while i < len(blob_counts):
            if blob_counts[i] < blob_threshold:
                # Merge with the closest column (prefer merging forward)
                if i < len(blob_counts) - 1:
                    # Merge with the next column
                    merged_boundary = (column_boundaries[i + 1] + column_boundaries[i + 2]) // 2
                    new_boundaries.append(merged_boundary)
                    new_blob_counts.append(blob_counts[i] + blob_counts[i + 1])
                    merged = True
                    i += 2  # Skip next column since it's merged
                elif len(new_boundaries) > 1:
                    # Merge with the previous column (if at the end)
                    new_boundaries[-1] = (column_boundaries[i] + column_boundaries[i + 1]) // 2
                    new_blob_counts[-1] += blob_counts[i]
                    merged = True
                    i += 1  # Move to the next column
            else:
                # Keep the column as is
                new_boundaries.append(column_boundaries[i + 1])
                new_blob_counts.append(blob_counts[i])
                i += 1

        if not merged:  # Stop if no merges happened
            break

        column_boundaries = new_boundaries
        blob_counts = new_blob_counts

    # ** Ensure At Least One Column Exists **
    if len(column_boundaries) < 2:
        column_boundaries = [0, width]  # Ensure at least one full-width column

    # ** Step 5: Draw detected column dividers on the image **
    output_image = image.copy()
    for x in column_boundaries:
        cv2.line(output_image, (x, 0), (x, height), (0, 0, 255), 2)  # Red line

    # Save the resulting image if a save path is provided
    if save_path:
        cv2.imwrite(save_path, output_image)


    return column_boundaries

# Loop through all table images in table_list and process them

whitespace_threshold = 25 # Minimum whitespace width required to be a column divider
blob_threshold = 20  # Keep merging until at least this many text blobs exist in a section
min_blob_size = 10  # Ignore blobs smaller than this size (to filter noise)

for idx, table_entry in enumerate(table_list):
    table_image = table_entry["cropped_image"]  # Get the cropped table image

    if table_image is not None:
        column_positions = detect_and_draw_column_dividers(table_image, idx + 1, whitespace_threshold, blob_threshold, min_blob_size)
        #print(f"Table {idx + 1}: Refined Column Dividers at Positions:", column_positions)
        table_entry["column_positions"] = column_positions
    else:
        table_entry["column_positions"] = [0]
        print(f"WARNING: Table {idx + 1}: No image available.")

#Now that we have columns drawn, we can actually detect text merging/splitting errors
import cv2
import numpy as np
import matplotlib.pyplot as plt
import easyocr
from collections import defaultdict

def preprocess_image(image):
    """
    Converts image to grayscale (if needed), applies thresholding, and noise removal.
    """
    if len(image.shape) == 3:  # If the image has 3 channels (color), convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image  # Already grayscale

    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Optional: Noise removal
    binary = cv2.medianBlur(binary, 3)

    return binary


reader = easyocr.Reader(['en'])  # 'en' = English, can add other languages if needed

def extract_text_by_column(image, column_boundaries):
    """
    Extracts text from each detected column using EasyOCR.
    
    Parameters:
        image (numpy array): Preprocessed grayscale table image.
        column_boundaries (list): List of detected column boundaries (x-coordinates).

    Returns:
        dict: Dictionary where keys are column indices, values are lists of words in that column.
    """
    column_text = defaultdict(list)

    for col_idx, (x_start, x_end) in enumerate(zip(column_boundaries[:-1], column_boundaries[1:])):
        # Crop the image to this column
        column_image = image[:, x_start:x_end]  

        # Run EasyOCR on the cropped column image
        results = reader.readtext(column_image)

        # Store detected words
        for bbox, text, confidence in results:
            column_text[col_idx].append(text.strip())

    return column_text

def compare_columns(image_columns, json_ocr, table_index, max_distance=2):
    """
    Compares text extracted from the image with JSON OCR data to detect column splitting,
    while ignoring columns that are too far apart. Also prints words that got split.

    Parameters:
        image_columns (dict): Extracted words by column from the image.
        json_ocr (dict): JSON OCR table data.
        table_index (int): Index of the table in table_list.
        max_distance (int): Maximum column distance allowed for a valid split.

    Returns:
        list: List of detected split errors, including words that were split.
    """
    ocr_cells = json_ocr.get("cells", [])
    json_column_text = defaultdict(list)

    # Extract text per column from JSON
    for cell in ocr_cells:
        col_idx = cell["columnIndex"]
        text = cell.get("content", "").strip()
        json_column_text[col_idx].append(text)

    # Compare extracted image text to JSON text
    split_errors = []

    for img_col_idx, img_words in image_columns.items():
        matched_columns = defaultdict(set)  # Store words found in each OCR column

        # Check which JSON columns contain these words
        for json_col_idx, json_words in json_column_text.items():
            for word in img_words:
                if word in json_words:
                    matched_columns[json_col_idx].add(word)

        # Filter out distant column splits
        filtered_columns = {col: words for col, words in matched_columns.items() if abs(col - img_col_idx) <= max_distance}

        # If words from one image column appear in multiple nearby OCR columns, it's a split error
        if len(filtered_columns) > 1:
            split_errors.append((img_col_idx, filtered_columns))

    # Report Errors
    print(f"Table {table_index}: Found {len(split_errors)} split column(s)")
    
    table_list[table_index - 1]["word_split_error_count"] = len(split_errors)
    table_list[table_index - 1]["word_split_descriptions"] = [None] * len(split_errors)
    split_error_count = 0
    for img_col, json_cols in split_errors:
        split_details = "\n    ".join([f"→ OCR Column {col}: {list(words)}" for col, words in json_cols.items()])
        #print(f"  - Column {img_col} in the image was split into multiple OCR columns:\n    {split_details}")
        table_list[table_index - 1]["word_split_descriptions"][split_error_count] = f"  - Column {img_col} in the image was split into multiple OCR columns:\n    {split_details}"
        split_error_count += 1
    return split_errors



def visualize_splits(image, column_boundaries, split_errors, table_index):
    """
    Draws detected columns and highlights split column errors.
    """
    # Ensure image is in color format for visualization
    if len(image.shape) == 2:  # If grayscale, convert to BGR
        output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        output_image = image.copy()

    height, width = output_image.shape[:2]

    # Draw detected columns (Green)
    for x in column_boundaries:
        cv2.line(output_image, (x, 0), (x, height), (0, 255, 0), 2)

    # Highlight split errors (Red)
    for img_col_idx, json_cols in split_errors:
        x_center = (column_boundaries[img_col_idx] + column_boundaries[img_col_idx + 1]) // 2
        cv2.putText(output_image, f"Split Error", (x_center, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    table_entry["word_split_image"] = [output_image]


split_threshold = 0.3  # Detect splits affecting at least 30% of rows
max_distance = 2  # Only consider splits within ±2 columns

table_count = len(table_list)

for idx, table_entry in enumerate(table_list):
    table_image = table_entry["cropped_image"]  # Get table image
    json_ocr = table_entry["table_json"]  # Get JSON OCR table

    print(f"Table Analyzer: Analyzing Table {idx+1}/{len(table_list)}")

    if table_image is not None and json_ocr:
        # Preprocess image
        preprocessed_image = preprocess_image(table_image)

        # Detect columns
        column_boundaries = table_entry["column_positions"]

        # Extract words from image columns
        image_column_text = extract_text_by_column(preprocessed_image, column_boundaries)

        # Compare extracted text with OCR JSON data (EXCLUDING FAR COLUMNS)
        split_results = compare_columns(image_column_text, json_ocr, idx + 1, max_distance=max_distance)

        # Visualize detected split errors
        visualize_splits(table_image, column_boundaries, split_results, idx + 1)
    else:
        print(f"WARNING: Table {idx + 1}: No image or OCR table available.")

#NOw create the PDF report
import json
import pandas as pd
import matplotlib.pyplot as plt
import io
import numpy as np
from PIL import Image
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph
import textwrap
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from io import BytesIO


def crop_image_by_threshold(img, threshold=240, margin=0):
    """
    Crops an image by finding the bounding box of pixels whose grayscale value is below the threshold.
    - img: a PIL Image.
    - threshold: pixels with a value >= threshold are considered white.
    - margin: additional pixels to include around the content.
    
    Returns:
      The cropped image.
    """
    gray = img.convert('L')
    arr = np.array(gray)
    rows = np.where(np.any(arr < threshold, axis=1))[0]
    cols = np.where(np.any(arr < threshold, axis=0))[0]
    if rows.size == 0 or cols.size == 0:
        return img
    top = max(rows[0] - margin, 0)
    bottom = min(rows[-1] + margin, arr.shape[0])
    left = max(cols[0] - margin, 0)
    right = min(cols[-1] + margin, arr.shape[1])
    return img.crop((left, top, right, bottom))

def df_to_image(df, title="", fontsize=12,
                base_cell_width=1.0,   # minimum cell width in inches
                base_cell_height=0.6,  # cell height in inches (header + each row)
                char_width_factor=0.15 # additional inches per character
               ):
    """
    Renders a pandas DataFrame as an image using matplotlib, then crops out extra white space
    using a NumPy–based threshold method.
    
    Returns:
      (cropped_img, page_width_pts, page_height_pts)
    """
    num_cols = len(df.columns) if not df.empty else 1
    num_rows = len(df) + 1

    col_widths = []
    for col in df.columns:
        header_len = len(str(col))
        cell_lens = df[col].astype(str).apply(len).tolist() if not df.empty else [0]
        max_len = max([header_len] + cell_lens)
        col_width = max(base_cell_width, char_width_factor * max_len)
        col_widths.append(col_width)
        
    total_width_in = sum(col_widths)
    total_height_in = num_rows * base_cell_height

    fig, ax = plt.subplots(figsize=(total_width_in, total_height_in))
    ax.axis('off')
    
    rel_col_widths = [w/total_width_in for w in col_widths]
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        colWidths=rel_col_widths,
        cellLoc='center',
        loc='upper center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(fontsize)

    if title:
        # Title for the table page.
        ax.set_title(title, fontsize=fontsize + 2, pad=2)

    fig.subplots_adjust(top=1.0, bottom=0.0, left=0.0, right=1.0)

    buf = io.BytesIO()
    dpi = 300
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=dpi)
    plt.close(fig)
    buf.seek(0)
    
    img = Image.open(buf)
    img_cropped = crop_image_by_threshold(img, threshold=240, margin=0)
    
    w_pixels, h_pixels = img_cropped.size
    page_width_points = (w_pixels / dpi) * 72
    page_height_points = (h_pixels / dpi) * 72
    
    return img_cropped, page_width_points, page_height_points

def create_pdf(table_data, output_filename="output.pdf"):
    """
    For each dictionary in table_data, creates two pages:
      - Page 1: The table (from table_json) on a custom-sized page that exactly fits
                the cropped table image (with title "Table on Page _").
      - Page 2: The remaining info on a dynamically sized page.
               On this page, a box error section is preceded by the title:
                   "bounding box merge/split errors (merge= red, split=blue)"
               and a word split section is preceded by the title:
                   "word split errors".
               If word split descriptions come as a list, they are joined into a string.
               The page height is extended if needed so all content fits.
    """
    default_page_size = letter  # we'll use letter width for Page 2
    c = canvas.Canvas(output_filename)
    styles = getSampleStyleSheet()
    style = styles["Normal"]
    max_cell_width = 40

    for entry in table_data:
        # ----------------------------
        # Page 1: Render table image.
        # ----------------------------
        table_json = entry["table_json"]
        page_number = entry["page_number"]
        num_rows = table_json["rowCount"]
        num_cols = table_json["columnCount"]
        
        table_data = [["" for _ in range(num_cols)] for _ in range(num_rows)]
        for cell in table_json.get("cells", []):
            row, col = cell["rowIndex"], cell["columnIndex"]
            content = cell.get("content", "").strip()
            wrapped_content = textwrap.fill(content, width=max_cell_width)
            table_data[row][col] = wrapped_content
        
        df = pd.DataFrame(table_data)
        
        # Render table using matplotlib
        fig_height = num_rows * 0.6 + 1.5  # 1.5 for title
        fig, ax = plt.subplots(figsize=(num_cols * 2, fig_height))
        ax.axis("off")

        # Reserve space at the top for title
        top_margin = 0.15  # Fraction of figure height to leave at top
        table_bottom = 0.05  # Bottom margin
        table_height = 1 - top_margin - table_bottom

        # Draw the title
        fig.text(0.5, 1 - top_margin / 2, f"Table on Page {page_number}", ha='center', va='center', fontsize=14, weight='bold')

        # Place the table within the adjusted bounding box
        table = ax.table(
            cellText=df.values,
            colLabels=None,
            loc='upper center',
            cellLoc='left',
            bbox=[0, table_bottom, 1.0, table_height]  # x, y, width, height (fractions)
        )


        font_size = 8
        points_per_inch = 72

        # Prepare wrapped data and track row heights
        wrapped_data = []
        row_heights_in = []

        fig, ax = plt.subplots(figsize=(num_cols * 2, num_rows * 0.6 + 1.5))  # Temporary figure to measure text
        ax.axis("off")

        for i in range(num_rows):
            wrapped_row = []
            max_height = 0  # Track max text height per row
            for j in range(num_cols):
                content = df.iat[i, j]
                wrapped = textwrap.fill(content, width=max_cell_width)
                wrapped_row.append(wrapped)

                # Create a temporary text object to measure its size
                text_obj = ax.text(0, 0, wrapped, fontsize=font_size)
                fig.canvas.draw()
                bbox = text_obj.get_window_extent(renderer=fig.canvas.get_renderer())
                text_height = bbox.height / points_per_inch  # Convert pixels to inches
                max_height = max(max_height, text_height)  # Track tallest text in row

            wrapped_data.append(wrapped_row)
            row_heights_in.append(max_height + 0.15)  # Add small buffer

        plt.close(fig)  # Close temp figure

        # Create the final figure with proper heights
        fig_height = sum(row_heights_in) + 1.0  # +1 for title
        fig, ax = plt.subplots(figsize=(num_cols * 2, fig_height))
        ax.axis("off")

        # Add title
        plt.text(0.5, 1 - 0.03, f"Table on Page {page_number}", fontsize=14, weight='bold', ha='center', va='top', transform=fig.transFigure)

        # Draw table
        table = plt.table(cellText=wrapped_data, colLabels=None, cellLoc='left', loc='center')

        # Apply dynamic row heights
        for i, height in enumerate(row_heights_in):
            for j in range(num_cols):
                cell = table[i, j]
                cell.set_fontsize(font_size)
                cell.get_text().set_verticalalignment('top')
                cell.set_height(height / fig_height)  # Scale height properly

        # Save to image buffer
        buf = BytesIO()
        plt.savefig(buf, format="png", bbox_inches='tight', dpi=150)
        buf.seek(0)

        # Insert into PDF
        img = Image.open(buf)
        page_width_points, page_height_points = img.size
        table_img_reader = ImageReader(buf)

        c.setPageSize((page_width_points, page_height_points))
        c.drawImage(
            table_img_reader,
            0, 0,
            width=page_width_points,
            height=page_height_points,
            preserveAspectRatio=True
        )
        c.showPage()

        plt.close(fig)


        # ----------------------------
        # Page 2: Draw remaining info.
        # ----------------------------
        # We want to dynamically set the page height so all content fits.
        # We'll use the letter width for this page.
        page_width_letter, _ = default_page_size
        margin = 36  # 0.5 inch margin
        gap = 12
        # Fixed heights and titles:
        box_title = "bounding box merge/split errors (merge= red, split=blue)"
        box_title_height = 14  # points
        box_image_height = 300  # points
        word_title = "word split errors"
        word_title_height = 14  # points
        word_image_height = 300  # points

        # Prepare the text details.
        wsd = entry.get("word_split_descriptions", "")
        if isinstance(wsd, list):
            wsd = "\n".join(wsd)
        
        if wsd == "":
            wsd = "No word split errors."
        other_text = (
            f"<b>Page Number:</b> {entry['page_number']}<br/>"
            f"<b>Box Error Count:</b> {entry['box_error_count']}<br/>"
            f"<b>Word Split Error Count:</b> {entry['word_split_error_count']}<br/>"
            f"<b>Word Split Descriptions:</b> {wsd}"
        )
        # Wrap the text with a very high available height to get its natural height.
        usable_width = page_width_letter - 2 * margin
        para = Paragraph(other_text, style)
        _, text_height = para.wrap(usable_width, 10000)

        # Compute total required page height:
        # Top margin + box title + gap + box image + gap + word title + gap + word image + gap + text + bottom margin.
        required_page_height = (margin +
                                box_title_height + gap +
                                box_image_height + gap +
                                word_title_height + gap +
                                word_image_height + gap +
                                text_height + margin)
        # Set the page height to the required value.
        page2_size = (page_width_letter, required_page_height)
        c.setPageSize(page2_size)
        # Set starting y_position from top.
        y_position = required_page_height - margin

        # Draw box error title.
        c.setFont("Helvetica-Bold", 14)
        c.drawString(margin, y_position - box_title_height, box_title)
        y_position -= (box_title_height + gap)
        
        # Draw Box Error Image.
        box_img = entry["box_error_images"]
        if isinstance(box_img, list):
            box_img = box_img[0] if box_img else None
        if box_img is not None:
            try:
                if isinstance(box_img, np.ndarray):
                    box_img = Image.fromarray(box_img)
                box_img = ImageReader(box_img)
                c.drawImage(
                    box_img,
                    margin, y_position - box_image_height,
                    width=usable_width,
                    height=box_image_height,
                    preserveAspectRatio=True
                )
            except Exception as e:
                c.drawString(
                    margin, y_position - box_image_height/2,
                    f"Error loading Box Error Image: {e}"
                )
            y_position -= (box_image_height + gap)
        else:
            c.drawString(margin, y_position - 15, "No Box Error Image")
            y_position -= (15 + gap)
        
        # Draw word split title.
        c.setFont("Helvetica-Bold", 14)
        c.drawString(margin, y_position - word_title_height, word_title)
        y_position -= (word_title_height + gap)
        
        # Draw Word Split Image.
        word_split_img = entry["word_split_image"]
        if isinstance(word_split_img, list):
            word_split_img = word_split_img[0] if word_split_img else None
        if word_split_img is not None:
            try:
                if isinstance(word_split_img, np.ndarray):
                    word_split_img = Image.fromarray(word_split_img)
                word_split_img = ImageReader(word_split_img)
                c.drawImage(
                    word_split_img,
                    margin, y_position - word_image_height,
                    width=usable_width,
                    height=word_image_height,
                    preserveAspectRatio=True
                )
            except Exception as e:
                c.drawString(
                    margin, y_position - word_image_height/2,
                    f"Error loading Word Split Image: {e}"
                )
            y_position -= (word_image_height + gap)
        else:
            c.drawString(margin, y_position - 15, "No Word Split Image")
            y_position -= (15 + gap)
        
        # Reset to normal font for text.
        c.setFont("Helvetica", 12)
        para = Paragraph(other_text, style)
        w, h = para.wrap(usable_width, y_position - margin)
        para.drawOn(c, margin, y_position - h)
        
        c.showPage()
    
    c.save()

#get file location from selected_paths and create pdf and store report

print("Table Analyzer: Analysis COmplete. Generating report.")
script_directory = os.path.dirname(os.path.abspath(__file__))

path = os.path.join(script_directory, "selected_paths.json")
with open(path, "r") as json_file:
    paths_data = json.load(json_file)

image_folder = paths_data.get("folder_path", "")

table_report_path = os.path.join(image_folder, "table_report.pdf")
create_pdf(table_list, output_filename=table_report_path)
print(f"Table Analyzer: Report saved to {table_report_path}")
