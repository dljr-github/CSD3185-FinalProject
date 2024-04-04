import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import time
import pickle
import os
import matplotlib.pyplot as plt

# Tells us what joints are connected by a line
# Each key is a tuple of two integers, where each integer is the index of a keypoint
# Each value is a string that represents the color of the line
EDGES = {
    (0, 1): 'm',  # nose to left eye
    (0, 2): 'c',  # nose to right eye
    (1, 3): 'r',  # left eye to left ear
    (2, 4): 'g',  # right eye to right ear
    (0, 5): 'b',  # nose to left shoulder
    (0, 6): 'y',  # nose to right shoulder
    (5, 7): 'k',  # left shoulder to left elbow
    (7, 9): 'w',  # left elbow to left wrist
    (6, 8): 'p',  # right shoulder to right elbow
    (8, 10): 'o',  # right elbow to right wrist
    (5, 6): 'n',  # left shoulder to right shoulder
    (5, 11): 'm',  # left shoulder to left hip
    (6, 12): 'c',  # right shoulder to right hip
    (11, 12): 'r',  # left hip to right hip
    (11, 13): 'g',  # left hip to left knee
    (13, 15): 'b',  # left knee to left ankle
    (12, 14): 'y',  # right hip to right knee
    (14, 16): 'k'  # right knee to right ankle
}

# Load the MoveNet model from TensorFlow Hub
model = hub.load('https://tfhub.dev/google/movenet/singlepose/thunder/4')
movenet = model.signatures['serving_default']

def crop_frame(frame, keypoints, crop_padding=50):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y,x,1]))
    
    min_x = int(np.min(shaped[:,1]))
    max_x = int(np.max(shaped[:,1]))
    min_y = int(np.min(shaped[:,0]))
    max_y = int(np.max(shaped[:,0]))
    
    # Add padding to the crop region
    min_x = max(0, min_x - crop_padding)
    max_x = min(x, max_x + crop_padding)
    min_y = max(0, min_y - crop_padding)
    max_y = min(y, max_y + crop_padding)
    
    cropped_frame = frame[min_y:max_y, min_x:max_x]
    return cropped_frame

def draw_keypoints(frame, keypoints, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y,x,1]))
    
    for kp in shaped:
        ky, kx, kp_conf = kp
        if kp_conf > confidence_threshold:
            cv2.circle(frame, (int(kx), int(ky)), 6, (0,255,0), -1)

def draw_connections(frame, keypoints, edges, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y,x,1]))
    
    for edge, color in edges.items():
        p1, p2 = edge
        y1, x1, c1 = shaped[p1]
        y2, x2, c2 = shaped[p2]
        
        if (c1 > confidence_threshold) & (c2 > confidence_threshold):
            color_code = colors[color]
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), color_code, 4)

def draw_instructions(frame, class_name, remaining_time, next_class = None):
    cv2.putText(frame, f"Collecting data for class: {class_name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(frame, f"Time remaining: {remaining_time:.1f} seconds", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    if next_class:
        cv2.putText(frame, f"Next class: {next_class}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

def draw_text(frame, text, x, y):
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

ASKTOLOAD = True

# Define colors for different edges
colors = {'m': (255, 0, 0),   # magenta
          'c': (0, 255, 0),   # cyan
          'r': (255, 0, 255), # red
          'g': (0, 255, 255), # green
          'b': (255, 255, 0), # blue
          'y': (0, 0, 255),   # yellow
          'k': (255, 255, 255), # white
          'w': (0, 0, 0),     # black
          'p': (255, 128, 0), # orange
          'o': (128, 0, 255), # purple
          'n': (0, 128, 255)} # navy

# Open the webcam
cap = cv2.VideoCapture(0)

# Create the window
cv2.namedWindow('Movenet Singlepose', cv2.WINDOW_NORMAL)

# Initialize variables for data collection
keypoints_data = []
labels = []

# Collect data for each class
classes = ["standing", "sitting", "t-posing", "crane pose", "running", "warrior yoga pose", "tree yoga pose", "goddess yoga pose"]
num_classes = len(classes)

# Load existing data if available
data_file = "keypoints_data.pkl"
if os.path.exists(data_file):
    with open(data_file, "rb") as f:
        keypoints_data, labels = pickle.load(f)

# Ask the user if they want to collect more data or run the classification
user_input = input("Do you want to collect more data? (y/n): ")

if user_input.lower() == 'y':
    ASKTOLOAD = False
    # Wait for 15 seconds before starting data collection
    print("Starting data collection in 15 seconds...")
    print(f"First class: {classes[0]}")
    time.sleep(15)
    # Collect new data
    for i in range(num_classes):
        class_name = classes[i]
        start_time = time.time()
        while cap.isOpened() and time.time() - start_time < 30:  # Collect data for 30 seconds
            ret, frame = cap.read()
            img = frame.copy()
            img = tf.image.resize_with_pad(tf.expand_dims(img, axis=0), 256, 256)
            input_img = tf.cast(img, dtype=tf.int32)
            results = movenet(input_img)
            keypoints_with_scores = results['output_0'].numpy()[:,:51].reshape((17,3))
            keypoints_data.append(keypoints_with_scores.flatten())
            labels.append(i)  # Class label
            draw_connections(frame, keypoints_with_scores, EDGES, 0.3)
            draw_keypoints(frame, keypoints_with_scores, 0.3)
            remaining_time = 30 - (time.time() - start_time)
            
            # Get next class, if available, else put None
            if i < num_classes - 1:
                next_class = classes[(i + 1) % num_classes]
            else:
                next_class = None
            
            draw_instructions(frame, class_name, remaining_time, next_class)
            cv2.imshow('Movenet Singlepose', frame)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        
        print(f"Data collection for class {class_name} completed. Taking a 15-second break.")
        time.sleep(15)  # Take a 15-second break before collecting data for the next class
    
    # Save collected data
    with open(data_file, "wb") as f:
        pickle.dump((keypoints_data, labels), f)

# Convert data and labels to numpy arrays
keypoints_data_array = np.array(keypoints_data)
labels_array = np.array(labels)

# Create a StandardScaler instance
scaler = StandardScaler()

# Scale the input features
keypoints_data_scaled = scaler.fit_transform(keypoints_data_array)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(keypoints_data_scaled, labels_array, test_size=0.2, random_state=42)

# Define hyperparameter grids for each classifier
svm_params = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
rf_params = {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20]}
lr_params = {'C': [0.1, 1, 10], 'penalty': ['l2'], 'max_iter': [1000]}
knn_params = {'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance']}
dt_params = {'max_depth': [None, 5, 10], 'min_samples_split': [2, 5, 10]}

# Define the filenames for saving the trained models and best hyperparameters
svm_model_file = "svm_model.pkl"
svm_params_file = "svm_params.pkl"
rf_model_file = "rf_model.pkl"
rf_params_file = "rf_params.pkl"
lr_model_file = "lr_model.pkl"
lr_params_file = "lr_params.pkl"
knn_model_file = "knn_model.pkl"
knn_params_file = "knn_params.pkl"
gnb_model_file = "gnb_model.pkl"
dt_model_file = "dt_model.pkl"
dt_params_file = "dt_params.pkl"

# Check if trained models exist
if os.path.exists(svm_model_file) and os.path.exists(rf_model_file) and os.path.exists(lr_model_file) and \
   os.path.exists(knn_model_file) and os.path.exists(gnb_model_file) and os.path.exists(dt_model_file) and \
   os.path.exists(svm_params_file) and os.path.exists(rf_params_file) and os.path.exists(lr_params_file) and \
   os.path.exists(knn_params_file) and os.path.exists(dt_params_file) and ASKTOLOAD:
    user_input = input("Trained models found. Do you want to load them? (y/n): ")
    if user_input.lower() == 'y':
        with open(svm_model_file, "rb") as f:
            svm_classifier = pickle.load(f)
        with open(svm_params_file, "rb") as f:
            svm_best_params = pickle.load(f)
        with open(rf_model_file, "rb") as f:
            rf_classifier = pickle.load(f)
        with open(rf_params_file, "rb") as f:
            rf_best_params = pickle.load(f)
        with open(lr_model_file, "rb") as f:
            lr_classifier = pickle.load(f)
        with open(lr_params_file, "rb") as f:
            lr_best_params = pickle.load(f)
        with open(knn_model_file, "rb") as f:
            knn_classifier = pickle.load(f)
        with open(knn_params_file, "rb") as f:
            knn_best_params = pickle.load(f)
        with open(gnb_model_file, "rb") as f:
            gnb_classifier = pickle.load(f)
        with open(dt_model_file, "rb") as f:
            dt_classifier = pickle.load(f)
        with open(dt_params_file, "rb") as f:
            dt_best_params = pickle.load(f)
    else:
        # Train the models if the user chooses not to load them
        svm_grid = GridSearchCV(SVC(probability=True), svm_params, cv=5, n_jobs=-1)
        svm_grid.fit(X_train, y_train)
        svm_classifier = svm_grid.best_estimator_
        svm_best_params = svm_grid.best_params_

        rf_grid = GridSearchCV(RandomForestClassifier(), rf_params, cv=5, n_jobs=-1)
        rf_grid.fit(X_train, y_train)
        rf_classifier = rf_grid.best_estimator_
        rf_best_params = rf_grid.best_params_

        lr_grid = GridSearchCV(LogisticRegression(max_iter=1000), lr_params, cv=5, n_jobs=-1)
        lr_grid.fit(X_train, y_train)
        lr_classifier = lr_grid.best_estimator_
        lr_best_params = lr_grid.best_params_

        knn_grid = GridSearchCV(KNeighborsClassifier(), knn_params, cv=5, n_jobs=-1)
        knn_grid.fit(X_train, y_train)
        knn_classifier = knn_grid.best_estimator_
        knn_best_params = knn_grid.best_params_

        gnb_classifier = GaussianNB()
        gnb_classifier.fit(X_train, y_train)

        dt_grid = GridSearchCV(DecisionTreeClassifier(), dt_params, cv=5, n_jobs=-1)
        dt_grid.fit(X_train, y_train)
        dt_classifier = dt_grid.best_estimator_
        dt_best_params = dt_grid.best_params_

        # Save the trained models and best hyperparameters
        with open(svm_model_file, "wb") as f:
            pickle.dump(svm_classifier, f)
        with open(svm_params_file, "wb") as f:
            pickle.dump(svm_best_params, f)
        with open(rf_model_file, "wb") as f:
            pickle.dump(rf_classifier, f)
        with open(rf_params_file, "wb") as f:
            pickle.dump(rf_best_params, f)
        with open(lr_model_file, "wb") as f:
            pickle.dump(lr_classifier, f)
        with open(lr_params_file, "wb") as f:
            pickle.dump(lr_best_params, f)
        with open(knn_model_file, "wb") as f:
            pickle.dump(knn_classifier, f)
        with open(knn_params_file, "wb") as f:
            pickle.dump(knn_best_params, f)
        with open(gnb_model_file, "wb") as f:
            pickle.dump(gnb_classifier, f)
        with open(dt_model_file, "wb") as f:
            pickle.dump(dt_classifier, f)
        with open(dt_params_file, "wb") as f:
            pickle.dump(dt_best_params, f)
else:
    # Train the models if they don't exist
    svm_grid = GridSearchCV(SVC(probability=True), svm_params, cv=5, n_jobs=-1)
    svm_grid.fit(X_train, y_train)
    svm_classifier = svm_grid.best_estimator_
    svm_best_params = svm_grid.best_params_

    rf_grid = GridSearchCV(RandomForestClassifier(), rf_params, cv=5, n_jobs=-1)
    rf_grid.fit(X_train, y_train)
    rf_classifier = rf_grid.best_estimator_
    rf_best_params = rf_grid.best_params_

    lr_grid = GridSearchCV(LogisticRegression(max_iter=1000), lr_params, cv=5, n_jobs=-1)
    lr_grid.fit(X_train, y_train)
    lr_classifier = lr_grid.best_estimator_
    lr_best_params = lr_grid.best_params_

    knn_grid = GridSearchCV(KNeighborsClassifier(), knn_params, cv=5, n_jobs=-1)
    knn_grid.fit(X_train, y_train)
    knn_classifier = knn_grid.best_estimator_
    knn_best_params = knn_grid.best_params_

    gnb_classifier = GaussianNB()
    gnb_classifier.fit(X_train, y_train)

    dt_grid = GridSearchCV(DecisionTreeClassifier(), dt_params, cv=5, n_jobs=-1)
    dt_grid.fit(X_train, y_train)
    dt_classifier = dt_grid.best_estimator_
    dt_best_params = dt_grid.best_params_

    # Save the trained models and best hyperparameters
    with open(svm_model_file, "wb") as f:
        pickle.dump(svm_classifier, f)
    with open(svm_params_file, "wb") as f:
        pickle.dump(svm_best_params, f)
    with open(rf_model_file, "wb") as f:
        pickle.dump(rf_classifier, f)
    with open(rf_params_file, "wb") as f:
        pickle.dump(rf_best_params, f)
    with open(lr_model_file, "wb") as f:
        pickle.dump(lr_classifier, f)
    with open(lr_params_file, "wb") as f:
        pickle.dump(lr_best_params, f)
    with open(knn_model_file, "wb") as f:
        pickle.dump(knn_classifier, f)
    with open(knn_params_file, "wb") as f:
        pickle.dump(knn_best_params, f)
    with open(gnb_model_file, "wb") as f:
        pickle.dump(gnb_classifier, f)
    with open(dt_model_file, "wb") as f:
        pickle.dump(dt_classifier, f)
    with open(dt_params_file, "wb") as f:
        pickle.dump(dt_best_params, f)

# Evaluate classifiers
svm_predictions = svm_classifier.predict(X_test)
rf_predictions = rf_classifier.predict(X_test)
lr_predictions = lr_classifier.predict(X_test)
knn_predictions = knn_classifier.predict(X_test)
gnb_predictions = gnb_classifier.predict(X_test)
dt_predictions = dt_classifier.predict(X_test)

svm_accuracy = accuracy_score(y_test, svm_predictions)
rf_accuracy = accuracy_score(y_test, rf_predictions)
lr_accuracy = accuracy_score(y_test, lr_predictions)
knn_accuracy = accuracy_score(y_test, knn_predictions)
gnb_accuracy = accuracy_score(y_test, gnb_predictions)
dt_accuracy = accuracy_score(y_test, dt_predictions)

print("SVM Accuracy:", svm_accuracy)
print("Random Forest Accuracy:", rf_accuracy)
print("Logistic Regression Accuracy:", lr_accuracy)
print("K-Nearest Neighbors Accuracy:", knn_accuracy)
print("Gaussian Naive Bayes Accuracy:", gnb_accuracy)
print("Decision Tree Accuracy:", dt_accuracy)

# Open a file to write the classification reports
with open("classification_reports.txt", "w") as f:
    f.write("Classification Reports:\n")
    f.write("The classification report shows the following metrics for each class:\n")
    f.write("- Precision: The ability of the classifier not to label a negative sample as positive.\n")
    f.write("- Recall: The ability of the classifier to find all the positive samples.\n")
    f.write("- F1-score: A harmonic mean of precision and recall, providing a single metric for both.\n")
    f.write("- Support: The number of samples per class.\n")
    f.write("\nNote: Accuracy is not included in the classification report by default.\n")
    f.write("Accuracy is the ratio of correctly classified samples to the total number of samples.\n")
    f.write("\nThe report also includes two types of averages:\n")
    f.write("- Weighted Average (weighted avg): Calculates the average metric by weighting the metric of each class by its support (number of samples).\n")
    f.write("  This gives more importance to the metrics of classes with a larger number of samples.\n")
    f.write("- Macro Average (macro avg): Calculates the arithmetic mean of the metric values across all classes, treating each class equally.\n")
    f.write("  This does not consider the number of samples per class.\n\n")

    f.write("\nSVM Classification Report (Best Hyperparameters: {}):\n".format(svm_best_params))
    report = classification_report(y_test, svm_predictions, target_names=classes, digits=4, output_dict=True)
    for cls, metrics in report.items():
        if cls == 'accuracy':
            continue
        f.write(f"Class: {cls}\n")
        f.write(f"Precision: {metrics['precision']:.4f} (Out of {metrics['support']} samples, how many were correctly classified as positive)\n")
        f.write(f"Recall: {metrics['recall']:.4f} (Out of the true positive samples, how many were correctly classified)\n")
        f.write(f"F1-score: {metrics['f1-score']:.4f} (Harmonic mean of precision and recall)\n")
        f.write(f"Support: {metrics['support']}\n")
        f.write("\n")

    f.write("\nRandom Forest Classification Report (Best Hyperparameters: {}):\n".format(rf_best_params))
    report = classification_report(y_test, rf_predictions, target_names=classes, digits=4, output_dict=True)
    for cls, metrics in report.items():
        if cls == 'accuracy':
            continue
        f.write(f"Class: {cls}\n")
        f.write(f"Precision: {metrics['precision']:.4f} (Out of {metrics['support']} samples, how many were correctly classified as positive)\n")
        f.write(f"Recall: {metrics['recall']:.4f} (Out of the true positive samples, how many were correctly classified)\n")
        f.write(f"F1-score: {metrics['f1-score']:.4f} (Harmonic mean of precision and recall)\n")
        f.write(f"Support: {metrics['support']}\n")
        f.write("\n")

    f.write("\nLogistic Regression Classification Report (Best Hyperparameters: {}):\n".format(lr_best_params))
    report = classification_report(y_test, lr_predictions, target_names=classes, digits=4, output_dict=True)
    for cls, metrics in report.items():
        if cls == 'accuracy':
            continue
        f.write(f"Class: {cls}\n")
        f.write(f"Precision: {metrics['precision']:.4f} (Out of {metrics['support']} samples, how many were correctly classified as positive)\n")
        f.write(f"Recall: {metrics['recall']:.4f} (Out of the true positive samples, how many were correctly classified)\n")
        f.write(f"F1-score: {metrics['f1-score']:.4f} (Harmonic mean of precision and recall)\n")
        f.write(f"Support: {metrics['support']}\n")
        f.write("\n")

    f.write("\nK-Nearest Neighbors Classification Report (Best Hyperparameters: {}):\n".format(knn_best_params))
    report = classification_report(y_test, knn_predictions, target_names=classes, digits=4, output_dict=True)
    for cls, metrics in report.items():
        if cls == 'accuracy':
            continue
        f.write(f"Class: {cls}\n")
        f.write(f"Precision: {metrics['precision']:.4f} (Out of {metrics['support']} samples, how many were correctly classified as positive)\n")
        f.write(f"Recall: {metrics['recall']:.4f} (Out of the true positive samples, how many were correctly classified)\n")
        f.write(f"F1-score: {metrics['f1-score']:.4f} (Harmonic mean of precision and recall)\n")
        f.write(f"Support: {metrics['support']}\n")
        f.write("\n")

    f.write("\nGaussian Naive Bayes Classification Report:\n")
    report = classification_report(y_test, gnb_predictions, target_names=classes, digits=4, output_dict=True)
    for cls, metrics in report.items():
        if cls == 'accuracy':
            continue
        f.write(f"Class: {cls}\n")
        f.write(f"Precision: {metrics['precision']:.4f} (Out of {metrics['support']} samples, how many were correctly classified as positive)\n")
        f.write(f"Recall: {metrics['recall']:.4f} (Out of the true positive samples, how many were correctly classified)\n")
        f.write(f"F1-score: {metrics['f1-score']:.4f} (Harmonic mean of precision and recall)\n")
        f.write(f"Support: {metrics['support']}\n")
        f.write("\n")

    f.write("\nDecision Tree Classification Report (Best Hyperparameters: {}):\n".format(dt_best_params))
    report = classification_report(y_test, dt_predictions, target_names=classes, digits=4, output_dict=True)
    for cls, metrics in report.items():
        if cls == 'accuracy':
            continue
        f.write(f"Class: {cls}\n")
        f.write(f"Precision: {metrics['precision']:.4f} (Out of {metrics['support']} samples, how many were correctly classified as positive)\n")
        f.write(f"Recall: {metrics['recall']:.4f} (Out of the true positive samples, how many were correctly classified)\n")
        f.write(f"F1-score: {metrics['f1-score']:.4f} (Harmonic mean of precision and recall)\n")
        f.write(f"Support: {metrics['support']}\n")
        f.write("\n")

    print("Classification reports exported to classification_reports.txt")

# Visualize confusion matrices
svm_cm = confusion_matrix(y_test, svm_predictions)
rf_cm = confusion_matrix(y_test, rf_predictions)
lr_cm = confusion_matrix(y_test, lr_predictions)
knn_cm = confusion_matrix(y_test, knn_predictions)
gnb_cm = confusion_matrix(y_test, gnb_predictions)
dt_cm = confusion_matrix(y_test, dt_predictions)

plt.figure(figsize=(16, 6))
plt.subplot(2, 3, 1)
plt.imshow(svm_cm, cmap='Blues')
plt.title('SVM Confusion Matrix')
plt.xticks(np.arange(len(classes)), classes, rotation=90)
plt.yticks(np.arange(len(classes)), classes)
plt.colorbar()

plt.subplot(2, 3, 2)
plt.imshow(rf_cm, cmap='Greens')
plt.title('Random Forest Confusion Matrix')
plt.xticks(np.arange(len(classes)), classes, rotation=90)
plt.yticks(np.arange(len(classes)), classes)
plt.colorbar()

plt.subplot(2, 3, 3)
plt.imshow(lr_cm, cmap='Reds')
plt.title('Logistic Regression Confusion Matrix')
plt.xticks(np.arange(len(classes)), classes, rotation=90)
plt.yticks(np.arange(len(classes)), classes)
plt.colorbar()

plt.subplot(2, 3, 4)
plt.imshow(knn_cm, cmap='Purples')
plt.title('K-Nearest Neighbors Confusion Matrix')
plt.xticks(np.arange(len(classes)), classes, rotation=90)
plt.yticks(np.arange(len(classes)), classes)
plt.colorbar()

plt.subplot(2, 3, 5)
plt.imshow(gnb_cm, cmap='Oranges')
plt.title('Gaussian Naive Bayes Confusion Matrix')
plt.xticks(np.arange(len(classes)), classes, rotation=90)
plt.yticks(np.arange(len(classes)), classes)
plt.colorbar()

plt.subplot(2, 3, 6)
plt.imshow(dt_cm, cmap='YlOrBr')
plt.title('Decision Tree Confusion Matrix')
plt.xticks(np.arange(len(classes)), classes, rotation=90)
plt.yticks(np.arange(len(classes)), classes)
plt.colorbar()

plt.tight_layout()
plt.show()

# Real-time classification
while cap.isOpened():
    ret, frame = cap.read()
    img = frame.copy()
    img = tf.image.resize_with_pad(tf.expand_dims(img, axis=0), 256, 256)
    input_img = tf.cast(img, dtype=tf.int32)
    results = movenet(input_img)
    keypoints_with_scores = results['output_0'].numpy()[:,:51].reshape((17,3))
    flattened_keypoints = keypoints_with_scores.flatten().reshape(1, -1)
    scaled_keypoints = scaler.transform(flattened_keypoints)  # Scale the keypoints for real-time classification
    svm_prediction = svm_classifier.predict(scaled_keypoints)
    svm_confidence = svm_classifier.predict_proba(scaled_keypoints)
    rf_prediction = rf_classifier.predict(scaled_keypoints)
    rf_confidence = rf_classifier.predict_proba(scaled_keypoints)
    lr_prediction = lr_classifier.predict(scaled_keypoints)
    lr_confidence = lr_classifier.predict_proba(scaled_keypoints)
    knn_prediction = knn_classifier.predict(scaled_keypoints)
    knn_confidence = knn_classifier.predict_proba(scaled_keypoints)
    gnb_prediction = gnb_classifier.predict(scaled_keypoints)
    gnb_confidence = gnb_classifier.predict_proba(scaled_keypoints)
    dt_prediction = dt_classifier.predict(scaled_keypoints)
    dt_confidence = dt_classifier.predict_proba(scaled_keypoints)
    draw_connections(frame, keypoints_with_scores, EDGES, 0.3)
    draw_keypoints(frame, keypoints_with_scores, 0.3)
    
    draw_text(frame, f"SVM: {classes[svm_prediction[0]]} ({svm_confidence[0][svm_prediction[0]]:.2f})", 10, 30)
    draw_text(frame, f"RF: {classes[rf_prediction[0]]} ({rf_confidence[0][rf_prediction[0]]:.2f})", 10, 60)
    draw_text(frame, f"LR: {classes[lr_prediction[0]]} ({lr_confidence[0][lr_prediction[0]]:.2f})", 10, 90)
    draw_text(frame, f"KNN: {classes[knn_prediction[0]]} ({knn_confidence[0][knn_prediction[0]]:.2f})", 10, 120)
    draw_text(frame, f"GNB: {classes[gnb_prediction[0]]} ({gnb_confidence[0][gnb_prediction[0]]:.2f})", 10, 150)
    draw_text(frame, f"DT: {classes[dt_prediction[0]]} ({dt_confidence[0][dt_prediction[0]]:.2f})", 10, 180)
    cv2.imshow('Movenet Singlepose', frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
    
    if cv2.getWindowProperty('Movenet Singlepose', cv2.WND_PROP_VISIBLE) < 1:
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()