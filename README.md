# Handwritten Digit Recognizer 🔢

A machine learning project that recognizes handwritten digits using the MNIST dataset. Compares three different algorithms — KNN, Logistic Regression, and Decision Tree — and hits **96.7% accuracy**. Includes full model evaluation with confusion matrices and visualizations.

---

## What it does

- Loads and preprocesses the MNIST dataset (70,000 handwritten digit images)
- Trains three ML models and compares their performance
- Evaluates each model with accuracy scores and confusion matrices
- Visualizes results using matplotlib — graphs saved in the `Graphs/` folder

---

## Algorithms Compared

| Algorithm | Notes |
|-----------|-------|
| K-Nearest Neighbors (KNN) | Distance-based classification |
| Logistic Regression | Linear model with multi-class support |
| Decision Tree | Tree-based classifier |

**Best accuracy achieved: 96.7%**

---

## Why I built this

MNIST is the classic ML benchmark for a reason — it's real image data but approachable enough to experiment with multiple algorithms. The goal was to understand how different classifiers perform on the same dataset, not just get one model working, and to practice proper model evaluation with confusion matrices rather than just printing accuracy.

---

## Tech Stack

| Library | Usage |
|---------|-------|
| Python | Core language |
| scikit-learn | ML algorithms, train/test split, evaluation metrics |
| matplotlib | Visualization and confusion matrix plots |
| NumPy | Data handling and array operations |

---

## How to run it

**1. Clone the repo**
```bash
git clone https://github.com/KhizerAhmad/Handwritten-Digit-Recognizer.git
cd Handwritten-Digit-Recognizer
```

**2. Install dependencies**
```bash
pip install scikit-learn matplotlib numpy
```

**3. Run it**
```bash
python main.py
```

The script will train all three models, print accuracy scores, and save confusion matrix graphs to the `Graphs/` folder.

---

## Project Structure

```
Handwritten-Digit-Recognizer/
│
├── main.py          # Full ML pipeline — preprocessing, training, evaluation
├── Graphs/          # Generated confusion matrices and visualizations
└── .gitignore
```

---

## Results

- Models trained on 60,000 samples, tested on 10,000
- Confusion matrices generated for each algorithm
- Visual comparison of model performance saved to `Graphs/`

---

## Sample Output

```
KNN Accuracy:                 96.7%
Logistic Regression Accuracy: 92.x%
Decision Tree Accuracy:       87.x%
```

> *(Add your actual numbers here if different)*

---

## Screenshot

<img width="797" height="570" alt="Accuracy Comaparison" src="https://github.com/user-attachments/assets/6a89a055-1072-42c5-a5b7-010b8b6b30dc" />


---

## Author

**Khizer Ahmad** — built this to get hands-on with ML classification algorithms, proper model evaluation, and data visualization on a real benchmark dataset.

Feel free to fork it and try adding SVM or a neural network to the comparison.
