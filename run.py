# #DEPRECATED: This file is just a reference.

# # Load useful libraries
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import cross_val_score
# from sklearn.metrics import classification_report
# import pandas as pd
# import shap
# import matplotlib.pyplot as plt
# import numpy as np

# import seaborn as sns


# # shap method to get dataset
# # X = Independent Variables
# # y = Dataset Variable
# X, y = shap.datasets.adult(n_points=1000)
# y = pd.Series(y.astype(int))  

# # Split dataset into training and testing sets
# # - training (X_train, y_train)
# # - testing (X_test, y_test)
# #
# # - test_size = Proportion of data used for testing
# # - Stratify=y ensures class distribution remains balanced
# # - random_state=13 ensures consistent split across runs
# X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3,
#                                                 	stratify =y,
#                                                 	random_state = 13)

# # Build the model
# rf_clf = RandomForestClassifier(max_features=2, n_estimators =100, bootstrap = True)

# rf_clf.fit(X_train, y_train)

# # Make prediction on the testing data
# y_pred = rf_clf.predict(X_test)

# # Classification Report
# print(classification_report(y_pred, y_test))

# # load JS visualization code to notebook
# shap.initjs()

# explainer = shap.TreeExplainer(rf_clf)

# # Compute SHAP values
# shap_values = explainer.shap_values(X_test)

# shap_values_class_1 = shap_values[:,:,1]  # Select only the values for class 1 

# # Summary plot
# shap.summary_plot(shap_values_class_1, X_test)

# # Dependency graph
# index = "Age"
# shap.dependence_plot(index, shap_values_class_1, X_test, interaction_index=index)


# shap_df = pd.DataFrame(shap_values_class_1, columns=X_test.columns)

# # Display the first few rows
# #print(shap_df.head())

# shap_importance = shap_df.abs().mean().sort_values(ascending=False)

# # Display sorted feature importance
# print(shap_importance)

# importances = rf_clf.feature_importances_
# feature_importances = pd.DataFrame({
#     'Feature': X.columns,
#     'Importance': importances
# })
# feature_importances = feature_importances.sort_values(by='Importance', ascending=False)

# print(feature_importances)
# plt.figure(figsize=(10, 6))
# sns.barplot(x='Importance', y='Feature', data=feature_importances)
# plt.title('Feature Importances - Random Forest')
# plt.show()


