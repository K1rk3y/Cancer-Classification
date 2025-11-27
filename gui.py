## File structure
## data --> "volume_x_slice_n.h5"

import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
import numpy as np
import os
import h5py
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import cv2

class ImageGUI:
   
    def __init__(self, master):
         self.master = master
         GUI_title = "Glioma Classification from MRI Images"
         self.master.title(GUI_title)
         
         # Specify fonts
         self.custom_font0 = ("Arial", 16, "bold", "underline") # Title
         self.custom_font1 = ("Arial", 12, "bold") # Heading
         self.custom_font2 = ("Arial", 12) # Body

         # Initiate global variables
         self.slice_directory = None
         self.h5_files = None

         self.channel_idx = 1
         self.annotation_value = "OFF"
         self.alpha = 0.5

         self.slice_value = 0
         self.slice_slider_min = 0
         self.slice_slider_max = 154

         self.volume_value = 1
         self.volume_slider_min = 1
         self.volume_slider_max = 369
    
         # Create a frame for the GUI and center it
         self.frame = tk.Frame(self.master)
         self.frame.grid(row=0, column=0, padx=10, pady=10)
         self.frame.grid_rowconfigure(0, weight=1)
         self.frame.grid_columnconfigure(0, weight=1)
         
         # Create a border for the GUI
         self.border = tk.Frame(self.frame, borderwidth=2, relief="groove")
         self.border.grid(row=0, column=0, sticky="nsew")
         
         # Create in-window title
         self.title_label = tk.Label(self.border, text=GUI_title, font=self.custom_font0)
         self.title_label.grid(row=0, column=0, columnspan=4, padx=5, pady=5)
        
         # Create a "Load Slice Directory" button
         self.loadslicedirectory_button = tk.Button(self.border, text="Load Slice Directory", command=self.load_slice_directory, font=self.custom_font1)
         self.loadslicedirectory_button.grid(row=1, column=0, columnspan=2, padx=5, pady=5)

         # Label volume slider widget
         self.volume_slider_label = tk.Label(self.border, text="Select Volume:", font=self.custom_font1)
         self.volume_slider_label.grid(row=2, column=0, padx=5, pady=5)

         self.volume_slider = tk.Scale(self.border, from_=self.volume_slider_min, to=self.volume_slider_max, orient=tk.HORIZONTAL, length=200,  command=self.update_volume_value, font=self.custom_font2)
         self.volume_slider.grid(row=2, column=1, padx=5, pady=5)

         # Create annotation dropdown widget
         self.annotation_label = tk.Label(self.border, text="Annotation:", font=self.custom_font1)
         self.annotation_label.grid(row=3, column=0, padx=5, pady=5)
        
         annotation_options = ["OFF", "ON"]                                       # <----------------------- Define here or in function or global variable?
         self.annotation_dropdown = ttk.Combobox(self.border, values=annotation_options, font=self.custom_font2)
         self.annotation_dropdown.set(annotation_options[0]) # Set initial value to OFF
         self.annotation_dropdown.bind("<<ComboboxSelected>>", self.update_annotation_value) # Bind update_annotation_value function to change in annotation_dropdown
         self.annotation_dropdown.grid(row=3, column=1, padx=5, pady=5)

         # Create a channel selection dropdown widget
         self.channel_label = tk.Label(self.border, text="Select Channel:", font=self.custom_font1)
         self.channel_label.grid(row=4, column=0, padx=5, pady=5)
        
         channel_options = ["T2-FLAIR", "T1", "T1-Gd", "T2"]                                       # <----------------------- Define here or in function or global variable?
         self.channel_dropdown = ttk.Combobox(self.border, values=channel_options, font=self.custom_font2)
         self.channel_dropdown.set(channel_options[1]) # Set initial value to T1
         self.channel_dropdown.bind("<<ComboboxSelected>>", self.update_channel_selection) # Bind update_channel_selection function to change in channel_dropdown
         self.channel_dropdown.grid(row=4, column=1, padx=5, pady=5)

         # Label slice slider widget
         self.slicer_slider_label = tk.Label(self.border, text="Select Slice:", font=self.custom_font1)
         self.slicer_slider_label.grid(row=5, column=0, padx=5, pady=5)

         self.slice_slider = tk.Scale(self.border, from_=self.slice_slider_min, to=self.slice_slider_max, orient=tk.HORIZONTAL, length=200,  command=self.update_slice_value, font=self.custom_font2)
         self.slice_slider.grid(row=5, column=1, padx=5, pady=5)

         # Extract conventional features button
         self.extractconventionalfeatures_button = tk.Button(self.border, text="Extract Conventional Features", command=self.extract_conventional_features, font=self.custom_font1)
         self.extractconventionalfeatures_button.grid(row=6, column=0, columnspan=2, padx=5, pady=5)

         # Extract radiomic features button
         self.extractradiomicfeatures_button = tk.Button(self.border, text="Extract Radiomic Features", command=self.extract_radiomic_features, font=self.custom_font1)
         self.extractradiomicfeatures_button.grid(row=7, column=0, columnspan=2, padx=5, pady=5)
         
         # Create a blank canvas on which to show image
         self.canvas = tk.Canvas(self.border, width=241, height=241, bg="black")
         self.canvas.grid(row=1, rowspan=7, column=2, columnspan=2, padx=5, pady=5)
         
         

    ## ___________________________________________________________________________________ Show image function
    def show_image(self):
        # Required image
        file_directory = f"{self.slice_directory}/volume_{self.volume_value}_slice_{self.slice_value}.h5"

        # Print error and don't refresh if image not found
        if not os.path.exists(file_directory):
            print("ERROR: File not found! Image not updated")

        else:
            figure, axes = plt.subplots()
            figure.patch.set_facecolor('#F0F0F0')

            # Close previous figure
            plt.close(figure)

            # Create new figure
            with h5py.File(file_directory, 'r') as file:
                image = file["image"]

                # Make numpy array
                imageArray = np.array(image)
                # Normalise
                normimageArray = cv2.normalize(imageArray[:,:,self.channel_idx], None, norm_type=cv2.NORM_MINMAX)
                # Show image from selected channel
                axes.imshow(normimageArray, cmap='gray')
            
                # Check if annotation is ON
                if self.annotation_value == "ON":
                    # Annotation is ON then Overlay mask
                    mask = file["mask"]
                    # Make a composite of all 3 channels of the mask segmentation
                    mask_comp = np.sum(mask, axis=2)

                    # Ensure compatible with cv2.normalise
                    if mask_comp.dtype != np.float32:
                        mask_comp = mask_comp.astype(np.float32)

                    # Make numpy array
                    maskcompArray = np.array(mask_comp)
                    # Normalise
                    normmaskcompArray = cv2.normalize(maskcompArray, None, norm_type=cv2.NORM_MINMAX)

                    # Overlay mask
                    axes.imshow(normmaskcompArray*255, cmap='gray', alpha=self.alpha)

                axes.axis("off")
                figure.tight_layout()

                # Create tkinter canvas
                canvas = FigureCanvasTkAgg(figure, master=self.canvas)
                canvas.draw()

                # Pack new canvas into GUI window
                canvas.get_tk_widget().grid(row=1, rowspan=5, column=2, columnspan=2)

    ## ___________________________________________________________________________________ Load slice directory function
    def load_slice_directory(self):
        print()
        print("Selecting slice directory...")
        self.slice_directory = filedialog.askdirectory()
        print("    Directory selected!")
        print(f"    output: slice_directory = '{self.slice_directory}'")

        # Get min and max slice values
        self.h5_files = os.listdir(self.slice_directory)
        # Filter only .h5 files
        self.h5_files = [file for file in self.h5_files if file.endswith(".h5")]
                 
        # Update image
        self.show_image()

    ## ___________________________________________________________________________________ Function to handle annotation dropdown 
    def update_annotation_value(self, event):
        # Retrieve annotation value from dropdown
        self.annotation_value = self.annotation_dropdown.get()

        if self.slice_directory is not None:
            # Update image
            self.show_image()

    ## ___________________________________________________________________________________ Function to handle channel dropdown 
    def update_channel_selection(self, event):
        # Retrieve channel selection from dropdown
        self.channel_options = ["T2-FLAIR", "T1", "T1-Gd", "T2"]

        channel_selection = self.channel_dropdown.get()

        # Get index to encode channel selection from slice array
        self.channel_idx = self.channel_options.index(channel_selection)

        if self.slice_directory is not None:
            # Update image
            self.show_image()
    
    ## ___________________________________________________________________________________ Function to update slice value
    def update_slice_value(self, event):
        # Retrieve slice value from slider
        self.slice_value = self.slice_slider.get()

        if self.slice_directory is not None:
            # Update image
            self.show_image()
    
    ## ___________________________________________________________________________________ Function to update slice value
    def update_volume_value(self, event):
        # Retrieve slice value from slider
        self.volume_value = self.volume_slider.get()

        if self.slice_directory is not None:
            # Update image
            self.show_image()

    ## ___________________________________________________________________________________ Extract conventional features function
    def extract_conventional_features(self):
        print()
        print("extract_conventional_features(self)")
        print("    output: calculate and save conventional features to csv")

    ## ___________________________________________________________________________________ Extract conventional features function
    def extract_radiomic_features(self):
        print()
        print("extract_radiomic_features(self)")
        print("    output: calculate and save radiomic features to csv")

# Run
root = tk.Tk()
app = ImageGUI(root)
root.mainloop()
