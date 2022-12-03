import cv2
import time
import numpy as np
import warnings
from PIL import Image
import streamlit as st
import pandas as pd
from utils import *
import tkinter as tk
from tkinter import filedialog

warnings.simplefilter('ignore')


## Setting parameters and variables##
font = cv2.FONT_HERSHEY_SIMPLEX
root = tk.Tk()
root.withdraw()

root.wm_attributes('-topmost',1)

st.title("Defect Detection in Additive Manufacturing with ML and AI")
st.markdown("""---""") 



type = st.select_slider("Select type of Microscope Imaging:", ["OM", "SEM"], value="OM", )
# st.markdown(
#     f'''
#         <style>
#          .stSlider {{
#             width: 200px;
#          }}
#          </style>
#          <div class="stSlider"></div>
#     ''',
#     unsafe_allow_html=True
# )

if type == "OM":
    om = OM()
    st.header("Optical Microscopy")

    ## Processing a single image
    st.subheader("Process single image")
    img_file_buffer = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "tif"])

    show = st.checkbox(label = "Show Plot")

    if img_file_buffer is not None:
        image = np.array(Image.open(img_file_buffer))

    else:
        demo_image = r'Data\Optical Images - JAM Lab\H13\A1\bottom_1.tif' #DEMO_IMAGE
        image = np.array(Image.open(demo_image))
        st.write("Sample output:")

    defect_stats, fig = om.process(image, show=show)
    if show:
        st.pyplot(fig=fig)

    stats = (pd.DataFrame(defect_stats, index=[0]))
    st.write(stats)
    st.markdown("""---""") 

    ## Processing a folder
    st.subheader("Process single folder")
    # st.write("Enter Parameter:")
    condition = st.text_input("Enter Parameter:")
    if condition:
        st.write("Select folder")
        folder_sel1 = st.button("Pick folder with images of " + str(condition))
        show_all_single = st.checkbox(label = "Show Cumulative Plot for " + str(condition))

        if folder_sel1:
            dir_single = st.text_input('Selected Folder: ', filedialog.askdirectory(master=root))
            stats_single, fig_single = om.process_all_images_single(dir_single, condition)
            complete_stats_single = om.sum_stats(stats_single)
            stats_tot_single = pd.DataFrame(stats_single)
            st.write(stats_tot_single)
            if show_all_single:
                st.pyplot(fig=fig_single)
    st.markdown("""---""") 

    ## Processing multiple folders
    st.subheader("Process multiple folders")
    st.write("Select folder")
    folder_sel2 = st.button("Pick folder with images of different parameters")
    show_all = st.checkbox(label = "Show Cumulative Plot for all parameters")

    if folder_sel2:
        dir = st.text_input('Selected Folder: ', filedialog.askdirectory(master=root))
        stats, fig = om.process_all_images(dir)
        complete_stats = om.sum_stats(stats)
        stats_tot_all = pd.DataFrame(complete_stats)
        st.write(stats_tot_all)
        
        if show_all:
            st.pyplot(fig=fig)
    st.markdown("""---""") 

if type == "SEM":
    sem = SEM()
    st.header("Scanning Electron Microscopy")

    ## Processing a single image
    st.subheader("Process single image")
    img_file_buffer = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "tif"])

    show = st.checkbox(label = "Show Plot")

    if img_file_buffer is not None:
        image = np.array(Image.open(img_file_buffer))

    else:
        demo_image = r"Data\Jag images\SST SIDE VIEW (SEM)\A\A2 retouch.tif" #DEMO_IMAGE
        st.write("Sample output:")
        image = np.array(Image.open(demo_image))

    defect_stats, fig = sem.process(image, show=show)
    if show:
        st.pyplot(fig=fig)

    stats = (pd.DataFrame(defect_stats, index=[0]))
    st.write(stats)
    st.markdown("""---""") 

    ## Processing a folder
    st.subheader("Process single folder")
    # st.write("Enter Parameter:")
    condition = st.text_input("Enter Parameter:")
    if condition:
        st.write("Select folder")
        folder_sel1 = st.button("Pick folder with images of " + str(condition))
        show_all_single = st.checkbox(label = "Show Cumulative Plot for " + str(condition))

        if folder_sel1:
            dir_single = st.text_input('Selected Folder: ', filedialog.askdirectory(master=root))
            stats_single, fig_single = sem.process_all_images_single(dir_single, condition)
            complete_stats_single = sem.sum_stats(stats_single)
            stats_tot_single = pd.DataFrame(stats_single)
            st.write(stats_tot_single)
            if show_all_single:
                st.pyplot(fig=fig_single)
    st.markdown("""---""") 

    ## Processing multiple folders
    st.subheader("Process multiple folders")
    st.write("Select folder")
    folder_sel2 = st.button("Pick folder with images of different parameters")
    show_all = st.checkbox(label = "Show Cumulative Plot for all parameters")

    if folder_sel2:
        dir = st.text_input('Selected Folder: ', filedialog.askdirectory(master=root))
        stats, fig = sem.process_all_images(dir)
        complete_stats = sem.sum_stats(stats)
        stats_tot_all = pd.DataFrame(complete_stats)
        st.write(stats_tot_all)
        
        if show_all:
            st.pyplot(fig=fig)
    st.markdown("""---""") 

## TODO ##
# area thresh slider

