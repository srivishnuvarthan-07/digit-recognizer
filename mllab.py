'''1st program lda,qda

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Load Iris dataset
iris = load_iris()
X = iris.data  # Features
y = iris.target  # Labels
target_names = iris.target_names

# Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ---- Linear Discriminant Function (LDA) ----
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)
y_pred_lda = lda.predict(X_test)
print("=== LDF / LDA Results ===")
print("Accuracy:", accuracy_score(y_test, y_pred_lda))
print(classification_report(y_test, y_pred_lda, target_names=target_names))

# ---- Quadratic Discriminant Function (QDA) ----
qda = QuadraticDiscriminantAnalysis()
qda.fit(X_train, y_train)
y_pred_qda = qda.predict(X_test)
print("=== ODF / QDA Results ===")
print("Accuracy:", accuracy_score(y_test, y_pred_qda))
print(classification_report(y_test, y_pred_qda, target_names=target_names))
'''

''' 2nd program id3
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create and train the Decision Tree classifier (ID3)
model = DecisionTreeClassifier(criterion='entropy')  # ID3 uses entropy
model.fit(X_train, y_train)

# Predict the labels for test data
y_pred = model.predict(X_test)

# Evaluate accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# Predict a new sample
sample = [[5.1, 3.5, 1.4, 0.2]]  # Example input
prediction = model.predict(sample)
print("Predicted class:", iris.target_names[prediction[0]])
'''
'''3rd program svm 
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Load the iris dataset
iris = datasets.load_iris()
X = iris.data[:, :2]  # Use only the first two features for visualization
y = iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create an SVM classifier with a linear kernel
svm_model = SVC(kernel='linear', C=1.0)
svm_model.fit(X_train, y_train)

# Predict on the test set
y_pred = svm_model.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Optional: Visualize decision boundaries
def plot_decision_boundaries(X, y, model):
    h = .02  # step size in the mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k')
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')
    plt.title('SVM Decision Surface (Linear Kernel)')
    plt.show()

plot_decision_boundaries(X_train, y_train, svm_model)
'''
'''4 th program random forest

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Sample text data
texts = [
    "I love this movie!",        # positive
    "This film was great!",      # positive
    "What a wonderful story!",   # positive
    "I hated this movie.",       # negative
    "This was a terrible film.", # negative
    "I don't like it at all."    # negative
]
labels = [1, 1, 1, 0, 0, 0]  # 1 = positive, 0 = negative

# Convert text to numeric features
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.3, random_state=42)

# Train Random Forest
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Output results
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
'''



'''5th program

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names

# Split data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create and train KNN classifier
k = 3
knn = KNeighborsClassifier(k)
knn.fit(X_train, y_train)

# Predict on test set
y_pred = knn.predict(X_test)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}\n")

# Print correct and incorrect predictions
print("Correct Predictions:")
for i in range(len(y_test)):
    if y_pred[i] == y_test[i]:
        print(f"Sample {i}: Predicted = {target_names[y_pred[i]]}, Actual = {target_names[y_test[i]]}")

print("\nIncorrect Predictions:")
for i in range(len(y_test)):
    if y_pred[i] != y_test[i]:
        print(f"Sample {i}: Predicted = {target_names[y_pred[i]]}, Actual = {target_names[y_test[i]]}")
'''
''' 6th program
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination

# Step 1: Load the Kaggle heart dataset
df = pd.read_csv("heart.csv")

# Step 2: Preprocess: convert target to binary
df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)

# Discretize continuous features
cont_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
disc = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='uniform')
df[cont_cols] = disc.fit_transform(df[cont_cols])

# Use relevant features
df = df[['age', 'sex', 'cp', 'chol', 'fbs', 'thalach', 'exang', 'oldpeak', 'target']]

# Step 3: Build Bayesian Network structure
model = BayesianNetwork([
    ('age', 'chol'),
    ('sex', 'chol'),
    ('chol', 'target'),
    ('cp', 'target'),
    ('thalach', 'target'),
    ('exang', 'target'),
    ('oldpeak', 'target')
])

# Step 4: Train model
model.fit(df, estimator=MaximumLikelihoodEstimator)

# Step 5: Inference (diagnosis)
infer = VariableElimination(model)
result = infer.query(
    variables=['target'],
    evidence={'cp': 2, 'thalach': 1, 'exang': 1, 'oldpeak': 2}
)

# Step 6: Print diagnosis
print("Heart Disease Diagnosis Probability:")
print(result)
'''
''' 7 th program
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# Load dataset (Setosa=0 vs Versicolor=1)
iris = load_iris()
X = iris.data[:100, :2]  # first 2 features
y = iris.target[:100]    # binary labels: 0 and 1

# Preprocessing
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Add bias term
X_train = np.c_[np.ones((X_train.shape[0], 1)), X_train]
X_test = np.c_[np.ones((X_test.shape[0], 1)), X_test]

# Initialize weights
weights = np.zeros(X_train.shape[1])
learning_rate = 0.1
epochs = 10

# Training loop
for epoch in range(epochs):
    for i in range(len(X_train)):
        z = np.dot(X_train[i], weights)
        y_pred = 1 if z >= 0 else 0
        error = y_train[i] - y_pred
        weights += learning_rate * error * X_train[i]

# Testing
y_pred_test = []
for i in range(len(X_test)):
    z = np.dot(X_test[i], weights)
    y_pred_test.append(1 if z >= 0 else 0)

# Results
print("Accuracy:", accuracy_score(y_test, y_pred_test))
print("\nClassification Report:\n", classification_report(y_test, y_pred_test))
'''

'''9th program
import numpy as np
from scipy import stats

# Sample data for two groups
group1 = [23, 21, 18, 25, 30]
group2 = [32, 35, 29, 31, 27]

# Perform the two-sample t-test
t_stat, p_value = stats.ttest_ind(group1, group2)

# Significance level
alpha = 0.05

# Decision based on p-value
if p_value < alpha:
    result = "Reject the null hypothesis: There is a significant difference between the groups."
else:
    result = "Fail to reject the null hypothesis: No significant difference between the groups."

print(f"T-statistic: {t_stat}")
print(f"P-value: {p_value}")
print(result)
'''
'''10 th program'''
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Load the Breast Cancer dataset
data = load_breast_cancer()
X, y = data.data, data.target

# Create a pipeline with scaling and logistic regression
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=10000))

# Perform k-fold cross-validation with k=10
k = 10
cv_scores = cross_val_score(model, X, y, cv=k)

# Print the cross-validation scores and the average accuracy
print(f"Cross-validation scores (for each fold): {cv_scores}")
print(f"Average accuracy: {np.mean(cv_scores)}")

