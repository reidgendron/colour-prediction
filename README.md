# Colour Prediction
This is a small machine learning project in order to classify the colours of the pixels in an image based on a training set of RGB colour codes.

# Dependencies
- Python 3.10.9
- Pandas
- NumPy
- Scikit-Learn
- Matplotlib

# Project Report

### Introduction
The objective of this machine learning project is to classify RGB values into basic labelled colors using a Gaussian classification model. The RGB values represent the intensity of red, green, and blue colors in a pixel. The goal is to predict the basic labelled color category based on the input RGB values.

### Data Collection
A dataset of 3,950 RGB values and their corresponding basic labelled colors was collected. The dataset was split into 75% for training and 25% for testing.

### Data Preprocessing
To ensure uniformity, the RGB values were standardized to a range of [0, 1] by dividing each value by 255. This step ensured that any leverage from differing scales was minimized and facilitated better model learning from the data.

### Model Selection
A Gaussian classification model was chosen for this project. Gaussian classification models assume a Gaussian distribution for the data. The *SciKit-Learn* library in *Python* was utilized for model training and testing.

### Model Training
The Gaussian classification model was trained on the training data using the maximum likelihood approach. This approach estimated the parameters of the Gaussian distribution that best fit the data. The model was trained with RGB values as input and basic labelled colors as output.

### Model Evaluation
#### Test Set
- The performance of the model was evaluated on the testing data. The model achieved an accuracy of 54.4% in correctly classifying the RGB values into their respective basic labelled colors. Adjusting the model to gain better precision is definitely something to look into at this point.

#### New Data
- Testing the model on new data resulted in a visually appealing result. For the new data, a slice of LAB colours was generated in order to produce a nice gradient. The gradient was then converted to RGB values using the *SciKit-Image* package *lab2rgb*. Visually, the accuracy looks good for the model's intended purpose.

### Conclusion
In conclusion, a Gaussian classification model was successfully developed for RGB value classification into basic labelled colors. The model demonstrated a fair accuracy of 54.5% on the testing data (reminder: we are working with colour data), indicating its potential for practical applications. Future work could involve exploring alternative generative models and experimenting with different hyperparameters to further enhance the model's performance. Additionally, a larger dataset may help to improve the model's performance, as well as incorporating the confidence field of the dataset into the model.
