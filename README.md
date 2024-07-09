# American Sign Language Detection - Image Classification
This project performs image classification to detect American Sign Language (ASL) letters from a dataset.

**Libraries** \
The notebook imports the following libraries for data manipulation, image processing, model building, and evaluation:

pandas (pd) \
numpy (np) \
seaborn (sns) \
matplotlib (plt) \
OpenCV (cv2) \
scikit-image (skimage) \
tensorflow (tf) \
keras (keras) \
sklearn (various submodules) 


**Data Import and Preprocessing** 
* **Data Path:** The notebook defines the path to the training data directory.
* **Function for Data Loading:** A function get_data is defined to load images from the training directory. It iterates through subfolders corresponding to each ASL letter and resizes the images to a fixed size. It also assigns labels based on the folder name (e.g., 'A' for label 0, 'B' for label 1, etc.).
* **Train-Test Split:** The script splits the loaded data into training and testing sets using train_test_split from scikit-learn.
* **One-Hot Encoding:** The categorical labels are one-hot encoded using to_categorical from Keras for machine learning compatibility.

**Model Building**
* **Model Architecture:** A Convolutional Neural Network (CNN) architecture is built using Keras' Sequential API. It includes convolutional layers with ReLU activation, pooling layers, a flattening layer, a dense layer with ReLU activation, and a final dense layer with softmax activation for multi-class classification (29 classes for all ASL letters and additional classes for special characters like space and deletion).
* **Model Compilation:** The model is compiled with the Adam optimizer, categorical cross-entropy loss function, and accuracy metric.
* **Early Stopping:** Early stopping is implemented using EarlyStopping from Keras to prevent overfitting. Training stops when the validation loss doesn't improve for a certain number of epochs.
* **Model Training:** The model is trained on the training data with a specified number of epochs and batch size.
  
**Model Evaluation**
* **Training History:** The script retrieves the training history from the model object to analyze metrics like loss and accuracy over epochs.
* **Performance Metrics:** The model is evaluated on the testing data using model.evaluate. This provides metrics like loss and accuracy on unseen data.
* **Classification Report:** A classification report is generated using classification_report from scikit-learn to analyze precision, recall, and F1-score for each ASL letter class.
* **Confusion Matrix:** A confusion matrix is visualized using heatmap from seaborn to understand how often the model predicted each class correctly or incorrectly.
