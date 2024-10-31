import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
data = pd.read_csv('tree_data.csv')

# Define features and target based on intended columns
features = ['root_stone', 'root_grate', 'root_other', 'trunk_wire', 'trnk_light', 
            'trnk_other', 'brch_light', 'brch_shoe', 'brch_other', 'problems']
target = 'health'

# Drop rows with missing values in relevant columns
data = data.dropna(subset=features + [target])

# Convert categorical variables into dummy/indicator variables
data = pd.get_dummies(data, columns=features, drop_first=True)
data[target] = data[target].map({'Good': 2, 'Fair': 1, 'Poor': 0})

# Separate features and target
X = data.drop(columns=[target])

# Ensure all columns in X are numeric by filtering out any non-numeric columns
X = X.select_dtypes(include=[float, int])

# Confirm y is numeric
y = data[target]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# Train and evaluate model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=['Poor', 'Fair', 'Good']))
