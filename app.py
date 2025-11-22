import streamlit as st
from PIL import Image

# Try to import ultralytics; show a helpful Streamlit message if it's missing so
# the deployed app logs aren't redacted and users see actionable guidance.
ultralytics_import_error = None
try:
    from ultralytics import YOLO
except Exception as e:
    YOLO = None
    ultralytics_import_error = e

st.set_page_config(page_title="Smart City Object Detection", layout="wide")

# Title
st.title("Smart City Public Safety: YOLO Object Detection Dashboard")

if YOLO is not None:
    # Load model when ultralytics is available
    model = YOLO("yolov8n.pt")
else:
    # Inform the user and stop execution so Streamlit shows a clear error message
    st.error(
        "Missing dependency: `ultralytics`. Please ensure `requirements.txt` includes `ultralytics` and `torch`."
    )
    st.caption(f"Import error: {ultralytics_import_error}")
    st.stop()

# File uploader
uploaded = st.file_uploader("Upload a street image", type=["jpg", "jpeg", "png"])

if uploaded:
    # Show input image
    image = Image.open(uploaded)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Run YOLO
    st.subheader("Detecting objects...")
    result = model(image)[0]

    # Save output
    result.save("result.jpg")

    # Display results
    st.image("result.jpg", caption="Detected Objects", use_column_width=True)

    # Count objects
    labels = result.boxes.cls.cpu().numpy()
    names = result.names
    count_dict = {}

    for l in labels:
        cls_name = names[int(l)]
        count_dict[cls_name] = count_dict.get(cls_name, 0) + 1

    st.subheader("Object Counts")
    st.json(count_dict)
