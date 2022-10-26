import cv2
import time
import numpy as np
import warnings
from PIL import Image
import streamlit as st
import pandas as pd
from OM import *
import tkinter as tk
from tkinter import filedialog

warnings.simplefilter('ignore')


## Setting parameters and variables##
font = cv2.FONT_HERSHEY_SIMPLEX
root = tk.Tk()
root.withdraw()

root.wm_attributes('-topmost',1)

st.title("Defect Detection in AM with ML and AI")

## Processing a single image
st.subheader("Process single image")
img_file_buffer = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "tif"])

show = st.checkbox(label = "Show Plot")

if img_file_buffer is not None:
    image = np.array(Image.open(img_file_buffer))

else:
    demo_image = r'Data\Optical Images - JAM Lab\H13\A1\bottom_1.tif' #DEMO_IMAGE
    image = np.array(Image.open(demo_image))

defect_stats, fig = process(image, show=show)
if show:
    st.pyplot(fig=fig)
st.markdown(defect_stats)

## Processing a folder
st.subheader("Process single folder")

## Processing multiple folders
st.subheader("Process multiple folders")
st.subheader("Select folder")
folder_sel = st.button("Folder picker")
show_all = st.checkbox(label = "Show Cumulative Plot")

if folder_sel:
    dir = st.text_input('Selected Folder: ', filedialog.askdirectory(master=root))
    stats, fig = process_all_images(dir)
    complete_stats = sum_stats(stats)
    st.markdown("stats")
    st.markdown(complete_stats)
    
    if show_all:
        st.pyplot(fig=fig)

## TODO ##
# plots for each class
# display in a table the stats
# display progress
# instructions
# single folder processing

# sem
