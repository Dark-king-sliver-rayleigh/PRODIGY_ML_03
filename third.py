import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def load_images_from_folder(folder, label, img_size=(128, 128)):
    """
    Loads images from a folder, resizes them, and assigns labels.
    """
    images = []
    labels = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path)
        if img is not None:
            img_resized = cv2.resize(img, img_size)
            images.append(img_resized)
            labels.append(label)
    return images, labels

butterfly_path = "butterfly"
dolphin_path = "dolphin"

butterflies, butterfly_labels = load_images_from_folder(butterfly_path, label=0) 
dolphins, dolphin_labels = load_images_from_folder(dolphin_path, label=1) 

images = np.array(butterflies + dolphins)
labels = np.array(butterfly_labels + dolphin_labels)


images_flattened = images.reshape(len(images), -1)

scaler = StandardScaler()
images_flattened = scaler.fit_transform(images_flattened)

X_train, X_test, y_train, y_test = train_test_split(images_flattened, labels, test_size=0.2, random_state=42)

svm = SVC(kernel='linear', C=1.0, random_state=42)
svm.fit(X_train, y_train)

y_pred = svm.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Accuracy Score:", accuracy_score(y_test, y_pred))

def visualize_prediction(images, labels, predictions, num_images=10):
    """
    Displays a few test images with their true and predicted labels.
    """
    plt.figure(figsize=(10, 10))
    for i in range(num_images):
        idx = np.random.randint(0, len(images))
        img = images[idx].reshape(128, 128, 3)
        true_label = "Butterfly" if labels[idx] == 0 else "Dolphin"
        predicted_label = "Butterfly" if predictions[idx] == 0 else "Dolphin"
        plt.subplot(2, 5, i + 1)
        plt.imshow(cv2.cvtColor(img.astype("uint8"), cv2.COLOR_BGR2RGB))
        plt.title(f"True: {true_label}\nPred: {predicted_label}")
        plt.axis("off")
    plt.tight_layout()
    plt.show()

visualize_prediction(X_test.reshape(-1, 128, 128, 3), y_test, y_pred)
