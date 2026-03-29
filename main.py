from src.preprocessing import load_data, clean_data, feature_engineering, encode_data
from src.train import train_model, save_model
from src.evaluate import evaluate_model, plot_roc
from src.visualize import plot_graphs

from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt

# Load data
df = load_data("data/churn.csv")

# Preprocessing
df = clean_data(df)
df = feature_engineering(df)
df = encode_data(df)

# Split features & target
X = df.drop("Exited", axis=1)
y = df["Exited"]

# Handle imbalance
smote = SMOTE()
X, y = smote.fit_resample(X, y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = train_model(X_train, y_train)

# Feature Importance
importances = model.feature_importances_
plt.barh(X.columns, importances)
plt.title("Feature Importance")
plt.show()

# Evaluate model
evaluate_model(model, X_test, y_test)

# ROC Curve
plot_roc(model, X_test, y_test)

# Save model
save_model(model)

# Visualizations
plot_graphs()