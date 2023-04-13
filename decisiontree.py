from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import pandas as pd

# Load the data
data = pd.read_csv('customer_data.csv')

# Split the data into training and testing sets
X = data.drop(['conversation'], axis=1)
y = data['conversation']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Decision Tree classifier and fit it to the training data
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Make predictions on the testing data and calculate accuracy
y_pred = clf.predict(X_test)
accuracy = clf.score(X_test, y_test)
print(f"Accuracy: {accuracy:.2f}")

# Make predictions on a new sample customer
sample_customer = [[90000, 40, 1]]
prediction = clf.predict(sample_customer)
if prediction == 1:
    print("This customer is likely to have a conversation.")
else:
    print("This customer is unlikely to have a conversation.")
