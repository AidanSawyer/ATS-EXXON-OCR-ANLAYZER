'''
Aidan Sawyer

takes in data from extracted_data.h5
creates GUI to view all of the granular measurements of text quality
'''

import cv2
import tkinter as tk
from tkinter import Label, Button, Frame, Canvas
from PIL import Image, ImageTk
import os
import numpy as np
import json

class BoundingBoxViewer:
    def __init__(self, root, extracted_data, images_path):
        self.root = root
        self.extracted_data = extracted_data
        self.images_path = images_path
        self.page_number = 1  # Start at page 1
        self.hovered_word = None  # Track which word is being hovered over
        self.hovered_section = None
        self.last_update_time = 0  # Track last update time
        self.update_delay = 50  # Delay in milliseconds (throttles updates)
        self.pending_update = None  # Store pending update event
        self.word_boxes = []
        self.max_page = max(d["page"] for d in extracted_data)

        # Set window size
        self.root.geometry("1200x950")  # Adjusted for better fit
        self.root.title("Bounding Box Viewer")

        # Main frame to hold image & info panel
        self.main_frame = Frame(root)
        self.main_frame.pack(pady=5, side="left")

        # Navigation buttons (Above the image)
        self.button_frame = Frame(self.main_frame)
        self.button_frame.pack(pady=5)

        self.prev_button = Button(self.button_frame, text="⬅️ Previous", command=self.prev_page, width=10, height=2)
        self.prev_button.pack(side="left", padx=10)

        self.next_button = Button(self.button_frame, text="Next ➡️", command=self.next_page, width=10, height=2)
        self.next_button.pack(side="right", padx=10)

        # Image canvas
        self.canvas = Canvas(self.main_frame, width=750, height=900)
        self.canvas.pack()

        # Bind mouse movement event
        self.canvas.bind("<Motion>", self.on_mouse_move)

        # Right panel for hovered word details
        self.section_boxes = None
        self.info_frame = Frame(root, width=200, height=900)
        self.info_frame.pack(side="left", padx=10)

        self.word_label = Label(self.info_frame, text="Hover over a word...", font=("Arial", 14))
        self.word_label.pack(pady=10)

        self.confidence_label = Label(self.info_frame, text="Confidence: N/A", font=("Arial", 12))
        self.confidence_label.pack(pady=5)
        self.section_label = Label(self.info_frame, text="Section: N/A", font=("Arial", 12))
        self.section_label.pack(pady=5)
        
        self.smudge_label = Label(self.info_frame, text="Smudge: N/A", font=("Arial", 12))
        self.smudge_label.pack(pady=5)

        self.fade_label = Label(self.info_frame, text="Fade: N/A", font=("Arial", 12))
        self.fade_label.pack(pady=5)

        self.section_avg_label = Label(self.info_frame, text="Section Avg Smudge: N/A", font=("Arial", 12))
        self.section_avg_label.pack(pady=5)

        self.section_conf_label = Label(self.info_frame, text="Section Avg Confidence: N/A", font=("Arial", 12))
        self.section_conf_label.pack(pady=5)

        self.section_fade_label = Label(self.info_frame, text="Section Avg Fade: N/A", font=("Arial", 12))
        self.section_fade_label.pack(pady=5)

        self.word_image_label = Label(self.info_frame)  # Placeholder for cropped word image
        self.word_image_label.pack(pady=10)

        self.word_intervention_label = Label(self.info_frame, text="Word Human Intervention Score: N/A", font=("Arial", 12))
        self.word_intervention_label.pack(pady=10)

        self.section_intervention_label = Label(self.info_frame, text="Section Human Intervention Score: N/A", font=("Arial", 12))
        self.section_intervention_label.pack(pady=10)

        self.tk_image = None  # Ensure reference exists at class level
        self.root.after(100, self.update_display)  # Delay execution slightly

    def update_display(self):
        """Loads image, draws bounding boxes, and updates the display."""
        self.image_filename = self.images_path + f"/page_{self.page_number}.jpg"

        if not os.path.exists(self.image_filename):
            print(f"Error: Could not find {self.image_filename}")
            return

        image = cv2.imread(self.image_filename)

        if image is None:
            print(f"Error: Could not load {self.image_filename}")
            return

        self.orig_height, self.orig_width, _ = image.shape

        #Resize the image to fit the canvas without zooming in
        image_resized = cv2.resize(image, (750, 900), interpolation=cv2.INTER_AREA)

        words_on_page = [word for word in self.extracted_data if word["page"] == self.page_number]

        if not words_on_page:
            print(f"No words found for page {self.page_number}")
            return

        image_resized = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)

        #Find and store section bounding boxes
        self.section_boxes = {}
        for word_data in words_on_page:
            section_id = word_data["section"]
            (xmin, ymin), (xmax, ymax) = word_data["bounding_box"]

            if section_id not in self.section_boxes:
                self.section_boxes[section_id] = [xmin, ymin, xmax, ymax]
            else:
                # Expand section bounding box
                self.section_boxes[section_id][0] = min(self.section_boxes[section_id][0], xmin)
                self.section_boxes[section_id][1] = min(self.section_boxes[section_id][1], ymin)
                self.section_boxes[section_id][2] = max(self.section_boxes[section_id][2], xmax)
                self.section_boxes[section_id][3] = max(self.section_boxes[section_id][3], ymax)

        #Draw word bounding boxes (RED)
        for word_data in words_on_page:
            (xmin, ymin), (xmax, ymax) = word_data["bounding_box"]

            # Scale bounding box for resized image
            scale_x = 750 / self.orig_width
            scale_y = 900 / self.orig_height

            xmin_display = int(xmin * scale_x)
            ymin_display = int(ymin * scale_y)
            xmax_display = int(xmax * scale_x)
            ymax_display = int(ymax * scale_y)

            cv2.rectangle(image_resized, (xmin_display, ymin_display), (xmax_display, ymax_display), (255, 0, 0), 2)

        #Draw section bounding boxes (YELLOW)
        for section_id, (sxmin, symin, sxmax, symax) in self.section_boxes.items():
            sxmin_display = int(sxmin * scale_x)
            symin_display = int(symin * scale_y)
            sxmax_display = int(sxmax * scale_x)
            symax_display = int(symax * scale_y)

            cv2.rectangle(image_resized, (sxmin_display, symin_display), (sxmax_display, symax_display), (255, 255, 0), 3)

        # Convert to a PIL image for Tkinter display
        self.tk_image = ImageTk.PhotoImage(Image.fromarray(image_resized))
        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_image)

        # Store words for hover detection
        self.word_boxes = words_on_page

    def next_page(self):
        """Move to the next page and update display."""
        if self.page_number < self.max_page:
            self.page_number += 1
            self.update_display()
        pass

    def prev_page(self):
        """Move to the previous page and update display."""
        if self.page_number > 1:
            self.page_number -= 1
            self.update_display()

    def on_mouse_move(self, event):
        """Detect if the mouse is hovering over a word and throttle updates."""
        x, y = event.x, event.y

        #Cancel previous pending update to prevent unnecessary updates
        if self.pending_update:
            self.root.after_cancel(self.pending_update)

        #Schedule a new update (delayed by self.update_delay)
        self.pending_update = self.root.after(self.update_delay, lambda: self.process_hover(x, y))

    def process_hover(self, x, y):
        """Handles hover logic separately to avoid lag."""
        self.pending_update = None  #Reset pending update flag

        for word_data in self.word_boxes:
            (xmin, ymin), (xmax, ymax) = word_data["bounding_box"]

            scale_x = 750 / self.orig_width
            scale_y = 900 / self.orig_height

            xmin_scaled = int(xmin * scale_x)
            ymin_scaled = int(ymin * scale_y)
            xmax_scaled = int(xmax * scale_x)
            ymax_scaled = int(ymax * scale_y)

            if xmin_scaled <= x <= xmax_scaled and ymin_scaled <= y <= ymax_scaled:
                if self.hovered_word != word_data["word"]:  
                    self.hovered_word = word_data["word"]
                    self.display_word_info(word_data)
                return

        # If not hovering over any word, clear display
        self.hovered_word = None
        self.hovered_section = None
        self.word_label.config(text="Hover over a word...")
        self.confidence_label.config(text="Confidence: N/A")
        self.section_label.config(text="Section: N/A")
        self.smudge_label.config(text="Smudge: N/A")
        self.fade_label.config(text="Fade: N/A")
        self.word_image_label.config(image="")
        self.section_avg_label.config(text="Section Avg Smudge: N/A")
        self.section_conf_label.config(text="Section Avg Confidence: N/A")
        self.section_fade_label.config(text="Section Avg Fade: N/A")
        self.word_intervention_label.config(text="Word Human Intervention Score: N/A")
        self.section_intervention_label.config(text="Section Human Intervention Score: N/A")


    def display_word_info(self, word_data):
        """Update right panel with word details and cropped image."""
        self.word_label.config(text=f"Word: {word_data['word']}")
        self.confidence_label.config(text=f"Confidence: {word_data['confidence']:.2f}")
        self.section_label.config(text=f"Section: {word_data['section']}")
        self.section_avg_label.config(text=f"Section Avg Smudge: {round(word_data['section_avg_smudge'],3)}")
        self.section_fade_label.config(text=f"Section Avg Fade: {round(word_data['section_avg_fade'],3)}")
        self.section_conf_label.config(text=f"Section Avg Confidence: {round(word_data['section_avg_confidence'],3)}")
        self.section_intervention_label.config(text=f"Section Human Intervention Score:{round(word_data['section_intervention_score'],3)}")

        #Handle missing "smudge" values
        smudge_value = word_data.get("smudge", "N/A")
        self.smudge_label.config(text=f"Smudge: {round(smudge_value,3)}")

        #Handle missing "fade" values
        fade_value = word_data.get("fade", "N/A")
        self.fade_label.config(text=f"Fade: {round(fade_value,3)}")

        #Handle missing "Intervention score" values
        word_intervention_value = word_data.get("word_intervention_score", "N/A")
        self.word_intervention_label.config(text=f"Word Intervention Score: {round(word_intervention_value,3)}")

        # Extract word image
        image = cv2.imread(self.image_filename)
        (xmin, ymin), (xmax, ymax) = word_data["bounding_box"]
        cropped_image = image[ymin:ymax, xmin:xmax]

    # Convert to grayscale for display
        if cropped_image.size != 0:
            gray_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
            word_pil = Image.fromarray(gray_image)
            word_pil = word_pil.resize((150, 50), Image.Resampling.LANCZOS)  # Resize for UI

            # Display the word image
            self.word_tk_image = ImageTk.PhotoImage(word_pil)
            self.word_image_label.config(image=self.word_tk_image)
        else:
            self.word_image_label.config(image="")


import h5py

script_directory = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(script_directory, "selected_paths.json")

with open(path, "r") as json_file:
    paths_data = json.load(json_file)

image_folder = paths_data.get("folder_path", "")

h5_file_path = os.path.join(image_folder, "extracted_data.h5")


with h5py.File(h5_file_path, "r") as hdf:
    extracted_data = []
    for item in hdf.keys():
        group = hdf[item]
        data = {}
        # Get keys stored as attributes
        for key in group.attrs:
            data[key] = group.attrs[key]
        # Get keys stored as datasets
        for key in group:
            data[key] = np.array(group[key])
        extracted_data.append(data)


root = tk.Tk()
BoundingBoxViewer(root, extracted_data, images_path=image_folder)  
root.mainloop()
