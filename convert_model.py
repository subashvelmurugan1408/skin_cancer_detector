from tensorflow.keras.models import load_model

# Load your working model
model = load_model("final_model.keras")

# Save as H5
model.save("model.h5")

print("✅ Model converted to H5 successfully!")