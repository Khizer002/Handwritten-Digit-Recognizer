Handwritten Digit Recognition using Machine Learning

This is my semester project where I built a digit recognition system using the MNIST dataset. I compared different machine learning algorithms to see which one works best.

What I Did:

I used 4 different algorithms to recognize handwritten digits (0-9):
- K-Nearest Neighbors (KNN)
- Logistic Regression  
- Decision Tree
- Random Forest

Results:

KNN gave me the best accuracy of 96.7% which is pretty good! Random Forest was close behind with similar accuracy. Decision Tree didn't perform as well, probably because of overfitting.

The dataset had 60,000 images of handwritten digits. I split it 70% for training and 30% for testing.

Technologies I Used:

- Python
- pandas for data handling
- numpy for calculations
- scikit-learn for ML algorithms
- matplotlib for graphs

How to Run This:

1. Clone this repo
2. Install the required libraries: pip install -r requirements.txt
3. Download the MNIST dataset from Kaggle (link below)
4. Save it as dataset.csv in the project folder
5. Run: python main.py

What the Code Does:

- Loads the dataset and checks for missing values
- Shows class distribution (how many of each digit)
- Scales pixel values from 0-255 to 0-1
- Splits data into training and testing sets
- Trains all 4 algorithms
- Calculates accuracy, precision, recall, and F1-score
- Creates confusion matrices to see where models make mistakes
- Generates graphs to compare performance

Dataset:

MNIST dataset from Kaggle: https://www.kaggle.com/datasets/oddrationale/mnist-in-csv

What I Learned:

- Feature scaling really matters! KNN and Logistic Regression improved a lot after scaling
- Random Forest handles image data really well
- Confusion matrices are super helpful to see which digits get confused
- Decision Trees can overfit easily on pixel data

Feel free to use this code for your own projects or learning!
