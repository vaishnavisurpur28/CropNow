import gradio as gr
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# Load models (make sure .h5 files are in the same folder or models/ directory in your repo)
crop_weed_model = load_model("C:\Users\USER\OneDrive\Desktop\CropNow\dlproject\models")
pest_model = load_model("C:\Users\USER\OneDrive\Desktop\CropNow\dlproject\models")
disease_model = load_model("C:\Users\USER\OneDrive\Desktop\CropNow\dlproject\models")

# Class labels
crop_weed_classes = ['Tomato Plant', 'Weed', 'Pest', 'Random Image']
pest_classes = ['Invalid', 'Aphids', 'Armyworm', 'Beetle', 'Fruit Fly', 'Mosquito', 'Red Palm Weevil', 'Thrips', 'Tomato Leaf Miner']
disease_classes = [
    'Tomato Bacterial Spot', 'Tomato Early Blight', 'Tomato Late Blight', 'Tomato Leaf Mold',
    'Tomato Mosaic Virus', 'Tomato septoria leaf spot', 'Tomato Spider Mites',
    'Tomato leaf yellow_curl virus', 'Tomato Target Spot', 'Tomato Healthy', 'Random Image'
]

# Information dictionaries
weed_info = {
    "Symptoms": "Unwanted plant growing among crops that compete for water and nutrients.",
    "Solution": "Use manual weeding or herbicides. Employ crop rotation to manage weed growth."
}

pest_info = {
    "Aphids": ("Soft-bodied insects that suck plant sap.", "Use neem oil or insecticidal soap."),
    "Armyworm": ("Larvae that feed on foliage and fruit.", "Use Bacillus thuringiensis (Bt) sprays."),
    "Beetle": ("Chew leaves, stems, and fruits.", "Handpick or apply suitable insecticides."),
    "Fruit Fly": ("Lay eggs in fruit, leading to rot.", "Use pheromone traps or bait sprays."),
    "Mosquito": ("Not typically harmful to tomato plants.", "Maintain dry surroundings."),
    "Red Palm Weevil": ("Bores into plant tissue.", "Destroy infected plants and use traps."),
    "Thrips": ("Cause silvering of leaves and fruit scarring.", "Use sticky traps and insecticides."),
    "Tomato Leaf Miner": ("Tunnel into leaves creating white trails.", "Use biological control or neem oil.")
}

disease_info = {
    "Tomato Bacterial Spot": ("Water-soaked spots on leaves that turn brown with yellow halos.",
                              "Use certified seeds, avoid overhead watering, and apply copper-based fungicides."),
    "Tomato Early Blight": ("Dark concentric rings on lower leaves forming target-like pattern.",
                            "Remove infected leaves, rotate crops, and use chlorothalonil fungicide."),
    "Tomato Late Blight": ("Dark greasy lesions on leaves; white mold under humid conditions.",
                           "Destroy infected plants, improve air circulation, and apply fungicides."),
    "Tomato Leaf Mold": ("Yellow spots on upper leaf surfaces with mold underneath.",
                         "Ensure good airflow, avoid leaf wetting, and apply fungicides."),
    "Tomato Mosaic Virus": ("Mottled light/dark green leaves, curling.",
                            "Remove infected plants, disinfect tools, and control aphids."),
    "Tomato septoria leaf spot": ("Small, circular spots with gray centers and dark margins.",
                                  "Remove lower leaves, avoid overhead watering, apply fungicide."),
    "Tomato Spider Mites": ("Tiny speckles and webbing under leaves.",
                            "Spray miticides or insecticidal soap, maintain leaf moisture."),
    "Tomato leaf yellow_curl virus": ("Upward curling leaves, yellowing, and stunted growth.",
                                      "Use resistant varieties, control whiteflies, remove infected plants."),
    "Tomato Target Spot": ("Circular brown spots with concentric rings.",
                           "Remove infected leaves, use proper fungicides."),
    "Tomato Healthy": ("No symptoms. Plant is healthy.", "No treatment needed.")
}

# Main classifier
def classify_image(img):
    img_resized = img.resize((224, 224))
    img_array = image.img_to_array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Step 1: Crop/Weed/Pest/Random detection
    crop_pred = crop_weed_model.predict(img_array)[0]
    crop_label = crop_weed_classes[np.argmax(crop_pred)]

    if crop_label == "Weed":
        return crop_label, weed_info["Symptoms"], weed_info["Solution"]

    elif crop_label == "Pest":
        pest_pred = pest_model.predict(img_array)[0]
        pest_label = pest_classes[np.argmax(pest_pred)]
        if pest_label == "Invalid":
            return "Invalid Pest Image", "Could not recognize a valid pest.", "Try uploading a clear image of the pest."
        else:
            symptoms, solution = pest_info.get(pest_label, ("No info available", "No solution found."))
            return pest_label, symptoms, solution

    elif crop_label == "Tomato Plant":
        disease_pred = disease_model.predict(img_array)[0]
        disease_label = disease_classes[np.argmax(disease_pred)]
        symptoms, solution = disease_info.get(disease_label, ("No info available", "No solution found."))
        return disease_label, symptoms, solution

    else:
        return "Unknown Image", "Not a valid crop/pest/weed image.", "Please upload a clear image."

# Gradio Interface
interface = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type="pil"),
    outputs=[
        gr.Text(label="Detected Class"),
        gr.Text(label="Symptoms"),
        gr.Text(label="Solution")
    ],
    title="CropNow - Unified Crop, Pest & Disease Classifier",
    description="Upload an image to detect if it's a tomato plant, pest, or weed. If it's a tomato plant, pest or disease, symptoms and treatment will be provided."
)

if __name__ == "__main__":
    interface.launch(server_name="0.0.0.0", server_port=8080)
