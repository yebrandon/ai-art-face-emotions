import datasets
import os
import csv
import numpy as np
from deepface import DeepFace

IMAGES_DIR_PATH = "images/"
SUBSET_NAME = "2m_random_50k"
results_count = {
    "angry": 0,
    "disgust": 0,
    "fear": 0,
    "happy": 0,
    "sad": 0,
    "surprise": 0,
    "neutral": 0,
    "several_faces_found": 0,
    "face_not_found": 0,
    "total_images_successful": 0,
    "total_images_processed": 0,
}

# Load the DiffusionDB dataset with the specified subset
dataset = datasets.load_dataset(
    path="poloclub/diffusiondb", name=SUBSET_NAME, split="train"
)

# Create images folder if it doesn't already exist
if not (os.path.exists(IMAGES_DIR_PATH)):
    os.mkdir(IMAGES_DIR_PATH)

# Save and label each image in the dataset if face is detected
print("Attempting to analyze " + str(len(dataset)) + " images...")
with open(file="image_results.csv", mode="w", encoding="utf-8", newline="") as csv_file:
    writer = csv.writer(csv_file)

    for i in range(len(dataset)):
        PIL_image = dataset[i]["image"]
        numpy_image = np.array(PIL_image)
        try:
            results_count["total_images_processed"] += 1
            result = DeepFace.analyze(
                img_path=(numpy_image), actions="emotion", silent=True
            )
        except ValueError:
            results_count["face_not_found"] += 1
            continue

        if len(result) > 1:
            results_count["several_faces_found"] += 1
            continue

        # Write prompt, dominant emotion, and emotion confidences for each image file to CSV
        emotion_confidences = result[0]["emotion"]
        dom_emotion = result[0]["dominant_emotion"]
        dom_emotion_confidence = emotion_confidences[dom_emotion]
        results_count[dom_emotion] += 1
        results_count["total_images_successful"] += 1

        writer.writerow(
            [
                i,
                dataset[i]["prompt"],
                dom_emotion,
                dom_emotion_confidence,
                emotion_confidences,
            ]
        )

        # Save image in images directory
        file_path = IMAGES_DIR_PATH + str(i) + ".jpg"
        PIL_image.save(file_path)
        print("Saved " + file_path)

# Write tally of images emotions detected to CSV
print("Writing summary of results...")
with open(file="image_results_summary.csv", mode="w", newline="") as csv_file:
    writer = csv.writer(csv_file)
    for key in results_count.keys():
        writer.writerow([key, results_count[key]])

print("Done! " + str(results_count["total_images_successful"]) + " images saved.")
