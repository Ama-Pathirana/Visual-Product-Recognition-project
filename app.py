import gradio as gr
import numpy as np
import similarSearch

# Function to find similar image IDs for a newly updated image
def find_similar_images(new_image):
    # Placeholder for your code to find similar image IDs based on the new_image
    similar_ids = similarSearch.predict(new_image)  # Replace with the actual logic for finding similar IDs
    print(similar_ids)
    return similar_ids

# Define Gradio interface
iface = gr.Interface(
    fn=find_similar_images, 
    inputs=gr.inputs.Image(label="Upload Image"), 
    outputs=gr.outputs.Textbox(label="Similar Image IDs"),
    title="Image Similarity Finder",
    description="Upload an image to find similar image IDs in the database."
)

# Launch the Gradio interface
iface.launch()
