from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ------------------------------
# 1. Load dataset
# ------------------------------
url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
df = pd.read_csv(url, sep='\t', names=['label', 'message'])

print("Dataset head:")
print(df.head())

# ------------------------------
# 2. Basic EDA plot: label counts
# ------------------------------
label_counts = df['label'].value_counts()
plt.figure()
plt.bar(label_counts.index, label_counts.values)
plt.title("Class Distribution (ham vs spam)")
plt.xlabel("Label")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# ------------------------------
# 3. Train-test split
# ------------------------------
X = df['message']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ------------------------------
# 4. Vectorization
# ------------------------------
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ------------------------------
# 5. Model training
# ------------------------------
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# ------------------------------
# 6. Evaluation
# ------------------------------
predictions = model.predict(X_test_vec)
acc = accuracy_score(y_test, predictions)
print("Accuracy:", acc)

def check_spam(msg):
    msg_vec = vectorizer.transform([msg])
    pred = model.predict(msg_vec)[0]
    return pred

test_msg ="http://free-gift-online-login-security.com"
print("Test message:", test_msg)
print("Prediction:", check_spam(test_msg))

# ------------------------------
# 7. Matplotlib: Confusion Matrix
# ------------------------------
cm = confusion_matrix(y_test, predictions, labels=['ham', 'spam'])
print("Confusion Matrix:\n", cm)

plt.figure()
plt.imshow(cm, interpolation='nearest')
plt.title("Confusion Matrix")
plt.colorbar()
tick_marks = np.arange(2)
plt.xticks(tick_marks, ['ham', 'spam'])
plt.yticks(tick_marks, ['ham', 'spam'])

# Add numbers on squares
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i, j],
                 ha="center", va="center",
                 color="white" if cm[i, j] > thresh else "black")

plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.tight_layout()
plt.show()

# ------------------------------
# 8. Matplotlib: ROC Curve
# ------------------------------
# Need binary labels (1=spam, 0=ham)
y_test_binary = (y_test == 'spam').astype(int)
y_prob = model.predict_proba(X_test_vec)[:, 1]  # probability of spam

fpr, tpr, thresholds = roc_curve(y_test_binary, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.3f})")
plt.plot([0, 1], [0, 1], linestyle='--')  # diagonal line
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Spam Detection")
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()
