import tkinter as tk
from tkinter import filedialog, messagebox
import json
import os
import subprocess
import threading

def browse_ocr():
    filename = filedialog.askopenfilename(title="Select OCR Data File")
    ocr_entry.delete(0, tk.END)
    ocr_entry.insert(0, filename)

def browse_pdf():
    filename = filedialog.askopenfilename(title="Select PDF File", filetypes=[("PDF Files", "*.pdf")])
    pdf_entry.delete(0, tk.END)
    pdf_entry.insert(0, filename)

def browse_folder():
    foldername = filedialog.askdirectory(title="Select Folder to Store Data")
    folder_entry.delete(0, tk.END)
    folder_entry.insert(0, foldername)

def submit():
    ocr_path = ocr_entry.get()
    pdf_path = pdf_entry.get()
    folder_path = folder_entry.get()
    
    root.update_idletasks() 

    # Ensure the folder path is provided
    if not folder_path:
        submit_label.config(text="Please select a folder to store data", fg="red")
        return

    # Create the folder if it does not exist
    try:
        os.makedirs(folder_path, exist_ok=True)
    except Exception as e:
        submit_label.config(text=f"Error creating folder: {e}", fg="red")
        return

    # Define the full path where selected_paths.json will be stored
    script_directory = os.path.dirname(os.path.abspath(__file__))
    json_file_path = os.path.join(script_directory, "selected_paths.json")

    # Store paths in the JSON file
    paths_data = {
        "ocr_path": ocr_path,
        "pdf_path": pdf_path,
        "folder_path": folder_path
    }

    try:
        with open(json_file_path, "w") as json_file:
            json.dump(paths_data, json_file, indent=4)
    except Exception as e:
        submit_label.config(text=f"Error saving JSON: {e}", fg="red")
        return

    # Check if files exist
    if not os.path.isfile(ocr_path):
        submit_label.config(text="OCR file not found", fg="red")
        return
    if not os.path.isfile(pdf_path):
        submit_label.config(text="PDF file not found", fg="red")
        return

    submit_label.config(text="Files Found Succesfully. \n Converting PDF to Image.\n This may take a minute", fg="blue")

    # Run process_data.py and wait for it to finish
    
    pdfToImage_thread = threading.Thread(target=run_pdfToImage)
    pdfToImage_thread.start()
    

def run_pdfToImage():
    try:
        script_directory = os.path.dirname(os.path.abspath(__file__))
        subprocess.run(["python", "PDF_To_Image.py"], cwd=script_directory, check=True)
        submit_label.config(text="PDF's Succesfully \nconverted to images.\nYou are now ready \nfor analysis", fg="green")
    except subprocess.CalledProcessError as e:
        submit_label.config(text=f"Error: Image Conversion Failed!", fg="red")


# Create main window
root = tk.Tk()
root.title("Data Path Selection")
root.geometry("500x250")


# Row 0-2: Labels and Entry Fields
tk.Label(root, text="Path to OCR data:").grid(row=0, column=1, sticky="w", padx=10, pady=5)
ocr_entry = tk.Entry(root, width=50)
ocr_entry.grid(row=0, column=2, columnspan=3, padx=5, pady=5)
tk.Button(root, command=browse_ocr, text="Browse").grid(row=0, column=5, padx=5)

tk.Label(root, text="Path to PDF:").grid(row=1, column=1, sticky="w", padx=10, pady=5)
pdf_entry = tk.Entry(root, width=50)
pdf_entry.grid(row=1, column=2, columnspan=3, padx=5, pady=5)
tk.Button(root, command=browse_pdf, text="Browse").grid(row=1, column=5, padx=5)

tk.Label(root, text="Folder to store data:").grid(row=2, column=1, sticky="w", padx=10, pady=5)
folder_entry = tk.Entry(root, width=50)
folder_entry.grid(row=2, column=2, columnspan=3, padx=5, pady=5)
tk.Button(root, command=browse_folder, text="Browse").grid(row=2, column=5, padx=5)

# Row 3: Centered Submit Button (More Vertical Space)
submit_button = tk.Button(root, command=submit, text="Submit" )
submit_button.grid(row=3, column=2, pady=15)  # Increased bottom padding

# Row 4: Label Below Submit Button (More Vertical Space)
submit_label = tk.Label(root, text="Status will appear here", fg="blue")
submit_label.grid(row=4, column=2, pady=(5, 40))  # More space before bottom buttons

# Run the application
root.mainloop()

