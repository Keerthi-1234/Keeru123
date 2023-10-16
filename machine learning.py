Python 3.12.0 (tags/v3.12.0:0fb18b0, Oct  2 2023, 13:03:39) [MSC v.1935 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
>>> # Import necessary libraries
... from sklearn.model_selection import train_test_split
... from sklearn.neighbors import KNeighborsClassifier
... from sklearn.metrics import accuracy_score
... from sklearn.datasets import load_iris
... 
... # Load the Iris dataset
... iris = load_iris()
... X, y = iris.data, iris.target
... 
... # Split the dataset into training and testing sets
... X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
... 
... # Initialize the K-Nearest Neighbors classifier
... knn_classifier = KNeighborsClassifier(n_neighbors=3)
... 
... # Train the classifier on the training data
... knn_classifier.fit(X_train, y_train)
... 
... # Make predictions on the test data
... y_pred = knn_classifier.predict(X_test)
... 
... # Evaluate the model accuracy
... accuracy = accuracy_score(y_test, y_pred)
... print(f'Model Accuracy: {accuracy * 100:.2f}%')
