import pandas as pd 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from joblib import dump 

#step 1 Load data
data_path = 'data.csv'
df = pd.read_csv(r"C:\Users\user\Desktop\BRCA\data.csv")

#step 2 Divide the data into features and target variable
X = df.drop("vital.status",axis=1)
y = df["vital.status"]

#step 3 Stratify data by the target variable and split it into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y,train_size=0.85,random_state=40)

#Step 4 Train a Random Forest classifier
rf = RandomForestClassifier(random_state=30)
rf.fit(X_train, y_train)

#Step 5 Make predictions on the test set
y_pred_prob = rf.predict_proba(X_test)[:,1]

#  Calculate the ROC curve and AUC (Area Under the Curve)
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
roc_auc = roc_auc_score(y_test,y_pred_prob)

#Step 7 Plot the ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.savefig(r'C:\Users\user\Desktop\BRCA\roc_curve.png')  # save .png
plt.show()

#Step 8 Validate the model using K-Fold cross-validation
cv = StratifiedKFold(n_splits=10)
scores= cross_val_score(RandomForestClassifier(random_state=42), X, y, cv=cv, scoring="accuracy")
print(f"Accuracy:{scores.mean():.4f}(+/-{scores.std()*2:.4f})")

#Step 9 Feature importance ranking
importance = pd.Series(data=rf.feature_importances_, index=X_train.columns)
importance = importance.sort_values(ascending=False)
print(importance)

#Step 10 Export the trained model
dump(rf, r'C:\Users\user\Desktop\BRCA\random_forest_model.joblib')