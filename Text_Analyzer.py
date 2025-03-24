'''
Aidan Sawyer

This Program does the following in this order:
- extracts ocr data form given file
- seperates each image of each pdf page into pixel arrays of each word an pairs it with its corresponding OCR word 
- uses some image processing techniques on each of these image to crop, denoise, and greyscale them
- uses these preprocessed images, along with their paired OCR data, to calculate their smuge and fade
- adds the smudge and fade to the word data (extracted_data list of dictionaries)
- groups the words into sections based on where they are on the page
- computes section averages for each section and adds those to word data
- uses all of this to calculate final human intervention score for each word and each section
- publishes all of this in extracted_data.h5 to use in Text_Report_Generator.py and Text_Viewer.py
'''


import json
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os
import h5py


#get the paths for the images and ocr data
script_directory = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_directory)

path = "selected_paths.json"

with open(path, "r") as json_file:
    paths_data = json.load(json_file)

ocr_json_path = paths_data.get("ocr_path", "")
image_folder = paths_data.get("folder_path", "")

extracted_data = []
word_index = 0  # Unique counter for each word

# Load OCR data
with open(ocr_json_path, "r", encoding="utf-8") as f:
    ocr_data = json.load(f)

print("Text Analyzer: OCR and PDF Files sucesfully found")

# Process each page in the OCR results
for page in ocr_data["analyzeResult"]["pages"]:
    page_number = page["pageNumber"]
    page_width = page["width"]  # Detected scanned width (e.g., 8.5139 inches)
    page_height = page["height"]  # Detected scanned height (e.g., 11.3611 inches)

    print(f"Text Analyzer: Processing Page {page_number}")

    # Load the corresponding scanned page image
    image_path = os.path.join(image_folder, f"page_{page_number}.jpg")  # ✅ FIXED

    if not os.path.exists(image_path):
        print(f"WARNING: Image not found for Page {page_number}")
        continue

    image = cv2.imread(image_path)
    img_height, img_width, _ = image.shape  # Get image dimensions

    # Loop through words and extract bounding boxes
    for word in page["words"]:
        word_text = word["content"]
        polygon = word["polygon"]  # 8 values (x1, y1, x2, y2, ... x4, y4)
        confidence = word.get("confidence", -1)  # Get confidence (default -1 if missing)

        # Convert bounding box coordinates from OCR scale to image pixels
        x_coords = [polygon[i] / page_width * img_width for i in range(0, 8, 2)]
        y_coords = [polygon[i + 1] / page_height * img_height for i in range(0, 8, 2)]

        # Get bounding box (xmin, ymin, xmax, ymax)
        xmin, xmax = int(min(x_coords)), int(max(x_coords))
        ymin, ymax = int(min(y_coords)), int(max(y_coords))

        # Crop the bounding box region from the image
        cropped_pixels = image[ymin:ymax, xmin:xmax]  # Extract pixels

        # Convert cropped region to a NumPy array
        pixel_array = cropped_pixels if cropped_pixels is not None else None

        # Store extracted data with a unique word index
        extracted_data.append({
            "word_index": word_index,  # Unique index for every word
            "word": word_text,
            "page": page_number,  # Include the page number
            "bounding_box": [(xmin, ymin), (xmax, ymax)],  # Top-left & bottom-right corners
            "pixels": pixel_array,
            "confidence": confidence,  # Include confidence score
        })

        word_index += 1  # Increment unique word index

print("\nText Analyzer: initial data extraction finished. Beggining text analysis")

'''
AT THIS POINT IN THE PROGRAM, WE HAVE THE EXTRACTED DATA FROM THE OCR FILE.
NOW WE NEED TO ACTUALLY ANALAYZE EVERYTHING AND ADD INFO TO THE EXTRACTED DATA
'''

'''
FUNCTIONS FOR TEXT ANALYSIS
'''

def denoise_image(image_array):
    """Denoises a grayscale image while preserving gaps between text."""
    
    denoised = cv2.fastNlMeansDenoising(image_array, None, 10, 7, 21)
    weak_denoised = cv2.fastNlMeansDenoising(image_array, None, 3, 7, 21)
    if len(image_array.shape) == 3:
        image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)

    # ✅ Adaptive Thresholding Instead of Otsu
    #binary_image = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
     #                                    cv2.THRESH_BINARY, 11, 2)

    _, binary_image = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, weak_binary_image =  cv2.threshold(weak_denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # ✅ Use an Elliptical Kernel Instead of Square
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    # ✅ Perform MORPH_CLOSE Before MORPH_OPEN
    cleaned = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)  # Fill small gaps
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)  # Remove small noise

    return cleaned, weak_binary_image

def crop_binary_image(binary_image):
    """
    Crops completely black or white rows/columns from the edges of a binary image.
    
    Args:
        binary_image (numpy.ndarray): Binary (black & white) image.

    Returns:
        numpy.ndarray: Cropped binary image.
    """
    
    # Step 1: Find valid rows (not completely black or white)
    row_sums = np.sum(binary_image, axis=1)  # Sum along columns
    valid_rows = np.where((row_sums > 0) & (row_sums < binary_image.shape[1] * 255))[0]

    # Step 2: Find valid columns (not completely black or white)
    col_sums = np.sum(binary_image, axis=0)  # Sum along rows
    valid_cols = np.where((col_sums > 0) & (col_sums < binary_image.shape[0] * 255))[0]

    # ✅ Check if there are valid rows/columns
    if valid_rows.size == 0 or valid_cols.size == 0:
        print("Warning: No valid content found! Returning original image.")
        return binary_image  # Return original if everything is black/white

    # Step 3: Determine cropping boundaries
    row_start, row_end = valid_rows[0], valid_rows[-1]
    col_start, col_end = valid_cols[0], valid_cols[-1]

    # Step 4: Crop the image
    cropped_image = binary_image[row_start:row_end+1, col_start:col_end+1]

    return cropped_image

def process_black_sections(clean_image, weak_clean_image, min_blob_size=10, min_size = 3):
    """Processes an image to count distinct sections of connected black pixels."""

    # ✅ Ensure black pixels are treated as foreground (invert the image if necessary)
    inverted_clean_image = cv2.bitwise_not(clean_image)  # Flip black ↔ white
    inverted_weak_image = cv2.bitwise_not(weak_clean_image)

    #inverted_raw_image = cv2.filter2D(inverted_raw_image, -1, np.ones((3,3), np.float32) / 9)

    num_labels, labels, stats_clean, _ = cv2.connectedComponentsWithStats(inverted_clean_image, connectivity=8) 
    num_labels, labels, stats_weak, _ = cv2.connectedComponentsWithStats(inverted_weak_image, connectivity=4) 

    # ✅ Count valid black sections (ignore background label 0)
    large_sections = sum(1 for stat in stats_clean[1:] if stat[cv2.CC_STAT_AREA] >= min_blob_size)
    small_sections = sum(1 for stat in stats_weak[1:] if stat[cv2.CC_STAT_AREA] >= min_size) #maybe add in a max size, but probably not nessacary

    #large sections is for detecting smudge, small_sections (which also includes large sections) is for detecting fade. maybe change some paramater names

    return large_sections, small_sections  # No need to subtract 1 anymore

def count_spaces_in_word(gray_word_image, threshold=192):
    """
    Counts the number of spaces (gaps) in a grayscale word image.
    
    Args:
        gray_word_image (numpy.ndarray): Grayscale image of the word.
        threshold (int): Intensity value to classify as background vs text.
    
    Returns:
        int: Estimated number of spaces.
        list: X-positions where spaces occur.
    """

    # Step 1: Compute the horizontal intensity profile
    column_intensity = np.mean(gray_word_image, axis=0)  # Average intensity per column

    # Step 2: Convert to binary (text = 0, background = 1)
    binary_profile = column_intensity > threshold  

    # Step 3: Count transitions (white → black and back)
    transitions = np.diff(binary_profile.astype(int))
    space_positions = np.where(transitions == -1)[0]  # X-positions of detected spaces
    space_count = len(space_positions)  # Number of spaces

    return space_count, space_positions

'''
Smudge is a (tuneable) weighted average of two measurements: the measurement of distinct ink blobs (letters) and
the amount of "spaces" we see in the image of the text. If we saw the Word HELLO, and each letter was completely surrounded by whitespace,
we would see zero smudge, since there would be 5 distinct black sections and 4 spaces.
If we saw this word written in cursive however, we would expect to see only one distinct black section, but we would still see every space
if the cursive was neat (since the connections between letters would not have a lot of black pixels), so we wouldn't determined it to be
completely smudged.
'''
def calculate_smudge(black_sections, space_count, word_length):
    """Calculates the smudge value based on black sections and space count."""

    section_component = 1 - min(1, black_sections/word_length)
    
    if(word_length <= 1): #a little hacky, but prevents division by zero
        word_length = 2
        
    space_component = 1 - min(1, space_count/(word_length-1)) #if a word has 4 letters, in theory there are 3 spaces

    return 0.7*section_component + 0.3*space_component #arbitrary weighted average but im going with it for now. This is tuenable

'''
Intuitevely, Fade can be thought of as a measure of the number of small ink sections that aren't big enough to constitue a letter
It is calculated based on the amount of letters we think we should be seeing
so if we see much more ink blobs than there are letters, we would say that that text is "faded"
if the word HELLO is what we think we see in the text, we would expect to see 5 distinct ink sections
but if we actually saw 15 sections(assuming they were all large enough to survive the denoising process)
then we would say that that text is faded
'''
def calculate_fade(black_sections, small_black_sections, word_length):

    fade = 0
    if small_black_sections <= black_sections: #there are not any small sections that are not also normal sections
        fade = 0
    else:
        extra_sections = small_black_sections - black_sections
        m = min(3, extra_sections / word_length) #this is tuneable, but if you change the 3 to a 4, you would need to change the multiplier in the next line
        fade = round(0.3333 * m, 3)

    return fade

'''
APPLY TEXT ANALYSIS
'''
total_words = len(extracted_data)
for i in range(total_words):

    if(i % 100 == 0):
        print(f"Analyzing Text: {i}/{total_words} words completed")

    word_data = extracted_data[i]
    word_index = word_data["word_index"]
    word_text = word_data["word"]
    word_length = len(word_text)
    confidence = word_data.get("confidence", -1)
    gray_pixels = cv2.cvtColor(word_data["pixels"], cv2.COLOR_BGR2GRAY)
    

    if gray_pixels is None or gray_pixels.size == 0:
        print(f"WARNING: Skipping word '{word_text}' (empty pixel array)")
        continue
    
    
    clean_image, weak_clean_image = denoise_image(gray_pixels)
    cropped_image = crop_binary_image(clean_image)
    num_sections, num_small_sections = process_black_sections(cropped_image, weak_clean_image)
    avg_greyscale_by_column = np.mean(gray_pixels, axis=0)
    space_count, space_positions = count_spaces_in_word(cropped_image)
    smudge = calculate_smudge(num_sections, space_count, word_length)  # Calculate smudge score
    fade = calculate_fade(num_sections, num_small_sections, word_length)
    # Store smudge value in extracted data
    extracted_data[i]["smudge"] = smudge
    extracted_data[i]["fade"] = fade

print("Text Analyzer: Word Analysis Complete. Beggining Section Analysis.")
'''
CODE TO GROUP WORDS INTO SECTIONS AND ADD THAT INFO TO EXTRACTED DATA
'''

def group_words_into_sections(extracted_data, x_threshold=20, y_threshold=7):
    """
    Groups words into sections based on their spatial proximity.

    Args:
        extracted_data (list of dict): The OCR-extracted word data with bounding boxes.
        x_threshold (int): Maximum horizontal gap (pixels) to be considered part of the same section.
        y_threshold (int): Maximum vertical gap (pixels) to be considered part of the same section.

    Returns:
        list of dict: Updated extracted_data with an added "section" key.
    """

    # Sort words by Y-coordinate (top to bottom)
    extracted_data.sort(key=lambda word: word["bounding_box"][0][1])  # Sort by ymin

    sections = []
    section_id = 0  # Unique section number

    for word in extracted_data:
        xmin, ymin = word["bounding_box"][0]  # Top-left corner of word
        xmax, ymax = word["bounding_box"][1]  # Bottom-right corner

        added_to_section = False

        # Check if word belongs to an existing section
        for section in sections:
            for sec_word in section:
                sec_xmin, sec_ymin = sec_word["bounding_box"][0]
                sec_xmax, sec_ymax = sec_word["bounding_box"][1]

                # Compute horizontal & vertical distance
                x_distance = abs(xmin - sec_xmax)
                y_distance = abs(ymin - sec_ymax)

                if x_distance <= x_threshold and y_distance <= y_threshold:
                    section.append(word)
                    word["section"] = section[0]["section"]  # Assign same section ID
                    added_to_section = True
                    break
            if added_to_section:
                break

        # If the word wasn't added to an existing section, create a new one
        if not added_to_section:
            word["section"] = section_id
            sections.append([word])
            section_id += 1  # Increment for next section

    # ✅ Second Pass: Merge Overlapping Sections
    merged_sections = merge_overlapping_sections(sections)

    # ✅ Assign new section IDs after merging
    for new_section_id, section in enumerate(merged_sections):
        for word in section:
            word["section"] = new_section_id

    return extracted_data


def merge_overlapping_sections(sections):
    """
    Merges sections if their bounding boxes overlap.

    Args:
        sections (list of list): List of word groups forming sections.

    Returns:
        list: Merged section list with no unused sections.
    """
    merged = True
    while merged:
        merged = False
        new_sections = []

        while sections:
            section = sections.pop(0)  # Take first section
            section_bbox = get_section_bounding_box(section)
            merged_section = section  # Start with current section

            i = 0
            while i < len(sections):
                other_section = sections[i]
                other_bbox = get_section_bounding_box(other_section)

                if check_overlap(section_bbox, other_bbox):
                    # Merge this section into merged_section
                    merged_section.extend(other_section)
                    del sections[i]  # Remove merged section
                    section_bbox = get_section_bounding_box(merged_section)  # Update bounding box
                    merged = True  # Flag that we need another merging pass
                else:
                    i += 1  # Only move forward if no merge happened

            new_sections.append(merged_section)  # Store fully merged section

        sections = new_sections  # Replace with merged sections before next pass

    return sections



def get_section_bounding_box(section):
    """Calculates the bounding box of a section."""
    sxmin = min(word["bounding_box"][0][0] for word in section)
    symin = min(word["bounding_box"][0][1] for word in section)
    sxmax = max(word["bounding_box"][1][0] for word in section)
    symax = max(word["bounding_box"][1][1] for word in section)
    return (sxmin, symin, sxmax, symax)


def check_overlap(bbox1, bbox2, min_overlap_ratio=0.15):
    """Checks if two bounding boxes overlap beyond a certain threshold."""
    x1_min, y1_min, x1_max, y1_max = bbox1
    x2_min, y2_min, x2_max, y2_max = bbox2

    # Compute intersection area
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    inter_width = max(0, inter_x_max - inter_x_min)
    inter_height = max(0, inter_y_max - inter_y_min)
    inter_area = inter_width * inter_height

    # Compute individual section areas
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)

    # Compute the overlap ratio
    min_area = min(area1, area2)
    if min_area == 0:
        return False  # Prevent division by zero

    overlap_ratio = inter_area / min_area

    # Only merge if overlap is significant
    return overlap_ratio >= min_overlap_ratio

extracted_data = group_words_into_sections(extracted_data)
print("Text Analyzer: Words Sucesfully Grouped Into Sections")

def compute_section_averages(extracted_data):
    """
    Computes section-wise averages for smudge and confidence, 
    and directly adds them to extracted_data.

    Args:
        extracted_data (list of dict): List of word dictionaries.

    Modifies:
        extracted_data: Adds "section_avg_smudge" and "section_avg_confidence" to each word.
    """

    # Step 1: Organize words into their respective sections
    section_data = {}  # Stores section-level lists of smudge & confidence values
    for word in extracted_data:
        section_id = word.get("section")
        page_number = word.get("page")

        if section_id is None:
            continue  # Skip words without a section

        section_key = (page_number, section_id)

        if section_key not in section_data:
            section_data[section_key] = {"smudge": [], "fade": [], "confidence": []}

        if "smudge" in word:
            section_data[section_key]["smudge"].append(word["smudge"])

        if "fade" in word:
            section_data[section_key]["fade"].append(word["fade"])

        if "confidence" in word:
            section_data[section_key]["confidence"].append(word["confidence"])
        

    # Step 2: Compute averages for each section
    section_averages = {}
    for section_key, values in section_data.items():
        avg_smudge = np.mean(values["smudge"]) if values["smudge"] else 0  # Handle empty list case
        avg_fade = np.mean(values["fade"]) if values["fade"] else 0
        avg_confidence = np.mean(values["confidence"]) if values["confidence"] else 0

        section_averages[section_key] = {
            "avg_smudge": avg_smudge,
            "avg_confidence": avg_confidence,
            "avg_fade": avg_fade
        }

    # Step 3: Add section averages directly to extracted_data
    for word in extracted_data:
        section_id = word.get("section")
        page_number = word.get("page")

        if section_id is not None:
            section_key = (page_number, section_id)
            if section_key in section_averages:
                word["section_avg_smudge"] = section_averages[section_key]["avg_smudge"]
                word["section_avg_fade"] = section_averages[section_key]["avg_fade"]
                word["section_avg_confidence"] = section_averages[section_key]["avg_confidence"]



compute_section_averages(extracted_data)
print("Text Analyzer: Section Analysis Complete. Computing final scores.")


'''
FINALLY, WE WANT TO COMPUTE OUR FINAL HUMAN INTERVENTION SCORES FOR BOTH WORDS AND SECTIONS
(WHERE SECTION SCORES ARE JUST AVERAGE OF EACH WORDS SCORE)

'''
def normalized_power(x, n=2):
    """
    Compute the normalized power function:
      P(Data|Error) = x^n / (x^n + (1-x)^n)
    where x is max(smudge, fade) and n defaults to 2.
    """
    return (x ** n) / ((x ** n) + ((1 - x) ** n))

def compute_human_intervention_score(confidence, smudge, fade, n=1.7):
    """
    Compute the human intervention score based on:
      - Prior: 1 - confidence
      - Likelihood for Error: normalized_power(max(smudge, fade), n)
      - Likelihood for No Error: 1 - max(smudge, fade)
    The final score is the posterior probability multiplied by 100.
    
    Parameters:
        confidence (float): OCR confidence (0.0 to 1.0).
        smudge (float): Smudge value (0.0 to 1.0).
        fade (float): Fade value (0.0 to 1.0).
        n (int): Exponent parameter for normalized power function (default 2).
    
    Returns:
        float: Human intervention score (0 to 100).
    """
    # Prior error probability
    prior_error = 1 - confidence

    # Use max(smudge, fade) as our variable x
    x = max(smudge, fade)

    # Likelihood P(Data|Error) using the normalized power function
    likelihood_error = normalized_power(x, n)

    # Likelihood P(Data|No Error)
    likelihood_no_error = 1 - x

    # Calculate the evidence P(Data)
    evidence = likelihood_error * prior_error + likelihood_no_error * (1 - prior_error)

    # Avoid division by zero
    if evidence == 0:
        posterior_error = 0
    else:
        # Bayes theorem: P(Error|Data) = (P(Data|Error) * P(Error)) / P(Data)
        posterior_error = (likelihood_error * prior_error) / evidence

    # Multiply by 100 to get a score from 0 to 100
    intervention_score = posterior_error * 100
    return intervention_score

def compute_intervention_scores(extracted_data):
    for word in extracted_data:
        word_intervention_score = compute_human_intervention_score(word["confidence"], word["smudge"], word["fade"])
        word["word_intervention_score"] = word_intervention_score

compute_intervention_scores(extracted_data)

def compute_section_intervention_scores(extracted_data):
    """
    Computes section-wise averages for smudge and confidence, 
    and directly adds them to extracted_data.

    Args:
        extracted_data (list of dict): List of word dictionaries.

    Modifies:
        extracted_data: Adds section_intervention_score
    """

    # Step 1: Organize words into their respective sections
    section_data = {}  # Stores section-level lists of smudge & confidence values
    for word in extracted_data:
        section_id = word.get("section")
        page_number = word.get("page")

        if section_id is None:
            continue  # Skip words without a section

        section_key = (page_number, section_id)

        if section_key not in section_data:
            section_data[section_key] = {"word_intervention_score": []}

        if "word_intervention_score" in word:
            section_data[section_key]["word_intervention_score"].append(word["word_intervention_score"])
        

    # Step 2: Compute averages for each section
    section_averages = {}
    for section_key, values in section_data.items():
        avg_intervention_score = np.mean(values["word_intervention_score"]) if values["word_intervention_score"] else 0  # Handle empty list case

        section_averages[section_key] = {
            "section_intervention_score": avg_intervention_score,
        }

    # Step 3: Add section averages directly to extracted_data
    for word in extracted_data:
        section_id = word.get("section")
        page_number = word.get("page")

        if section_id is not None:
            section_key = (page_number, section_id)
            if section_key in section_averages:
                word["section_intervention_score"] = section_averages[section_key]["section_intervention_score"]
                
compute_section_intervention_scores(extracted_data)
print("Text Analyzer: Human Intervention Scores Sucesfully Computed. Saving Data...")

'''
NOW THAT EXTRACTED DATA HAS EVERYTHING WE WANT IN IT
WE WILL PUT IT INTO A JSON FILE TO BE USED BY THE BOUNDING BOX VIEWER
WE SHOULD ALSO PROBABLY GENERATE A REPORT
'''

script_directory = os.path.dirname(os.path.abspath(__file__))
h5_file_path = os.path.join(image_folder, "extracted_data.h5")

import numpy as np

with h5py.File(h5_file_path, "w") as hdf:
    for i, d in enumerate(extracted_data):
        grp = hdf.create_group(f"item_{i}")  # Each dictionary entry as a group
        for key, value in d.items():
            if isinstance(value, np.ndarray):
                grp.create_dataset(key, data=value)  # Store NumPy arrays correctly
            else:
                grp.attrs[key] = value  # Store other data as attributes

print(f"Text Analyzer: Data Sucesfully saved at {h5_file_path}")
print("Text Analysis FINISHED")