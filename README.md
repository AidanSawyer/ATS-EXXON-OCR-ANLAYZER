# ATS-EXXON-OCR-ANLAYZER
My Code for Exxon OCR Wazoku Challenge

How to Use:
Download Repository and Unzip
In your downloaded folders directory (unzipped folder) open command prompt and run "pip install -r requirements.txt" to install all python dependencies

The four .bat files are what execute seperate parts of the program. They can be compbined as nessacary but for demonstration purposes I decided to keep them seperate.
<b>The one that must be ran first is Launch_Single_Document_Interface.bat<\b>
All of the other ones can be ran independently.
When you Run Launch_Single_Document_Interface.bat, it will ask you for 2 file locations: The OCR File being analyzed, and its accompanying pdf.
The third textbox is where you create a new folder to store the reports and data. This folder will be created if it does not already exist yet, and once the other programs
are finished running, you can navigate to this folder and view the reprorts.
For the most granular analysis of the document text, Run Launch_Text_Viewer.bat <b>Note: this can only be done after you have succesfully ran Generate_Text_Report.bat<\b>
