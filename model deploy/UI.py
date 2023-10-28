import streamlit as st
import pandas as pd
from PIL import Image

# Load data
df1 = pd.read_csv('E:\group 16\csv files\output (2).csv')
df2 = pd.read_csv('E:\group 16\csv files\gallery.csv')
csv_data = pd.merge(df1, df2, on='seller_img_id')

# Define the path to the gallery folder
gallery_folder_path = ""

# Streamlit app
st.set_page_config(
    page_title='Similar Image',
    page_icon=':mag:',
    layout='wide',
    initial_sidebar_state='expanded'
)
st.title('Similar Image')
st.markdown(
    """
    <style>
    .stTitle {
        text-align: center;
        font-size: 36px;
        margin-bottom: 30px;
    }
    .stRadio > div {
        display: flex;
        justify-content: center;
        align-items: center;
    }
    .stImageContainer {
        display: flex;
        justify-content: center;
        align-items: center;
        margin-bottom: 30px;
    }
    .stImage {
        max-width: 100%;
        height: auto;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Define the desired width and height for the resized images
image_width = 150
image_height = 150

# Calculate the number of columns based on the available width
num_columns = 5

# Calculate the number of rows required
images_per_page = 50
num_rows = (images_per_page + num_columns - 1) // num_columns

# Calculate the total number of pages required
total_pages = (len(csv_data) + images_per_page - 1) // images_per_page

# Create a row for the page selector and images
col1, col2 = st.columns([2, 3])

# Page selector
with col1:
    tabs = st.radio(" ", range(1, total_pages + 1))

# Calculate the range of images to display for the selected page
start_index = (tabs - 1) * images_per_page
end_index = min(start_index + images_per_page, len(csv_data))

# Display the image path above the collage images
with col2:
    image_path = "queries/eager-pink-raven-of-agility.jpeg"
    st.image(image_path, caption="Query Image", width=300)
    # Create a row for the collage images
    image_columns = st.columns(num_columns)

    # Loop through each row in the CSV data for the selected page
    for index in range(start_index, end_index):
        img_id = csv_data.loc[index, 'seller_img_id']
        img_path = csv_data.loc[index, 'img_path']
        product_id = csv_data.loc[index, 'product_id']
                
        # Construct the full path to the image
        full_img_path = gallery_folder_path + img_path
                
        # Open and resize the image to the desired dimensions
        image = Image.open(full_img_path)
        resized_image = image.resize((image_width, image_height))
                
        # Determine the column to place the image in
        col_index = index % num_columns
        with image_columns[col_index]:
            st.image(resized_image, caption=f'Product ID: {product_id}\nImage ID: {img_id}', width=image_width)
