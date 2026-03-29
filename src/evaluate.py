from sklearn.metrics import accuracy_score, classification_report, roc_curve, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    print("\n=========== MODEL EVALUATION ===========\n")

    # ✅ Accuracy
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}\n")

    # ✅ Classification Report
    print("Classification Report:\n")
    print(classification_report(y_test, y_pred))

    # ✅ Confusion Matrix (numbers)
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:\n", cm)

    # ✅ ROC-AUC Score
    y_prob = model.predict_proba(X_test)[:,1]
    roc_auc = roc_auc_score(y_test, y_prob)
    print(f"\nROC-AUC Score: {roc_auc:.4f}")

    # ✅ Confusion Matrix Graph
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()


def plot_roc(model, X_test, y_test):
    y_prob = model.predict_proba(X_test)[:,1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)

    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, label="ROC Curve")
    plt.plot([0,1], [0,1], linestyle='--')  # baseline
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid()
    plt.show()