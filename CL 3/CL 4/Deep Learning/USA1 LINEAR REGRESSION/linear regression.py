#!/usr/bin/env python
# coding: utf-8

# In[13]:


#Real estate agents want help to predict the house price for regions in the USA. He gave you the 
#dataset to work on and you decided to use the Linear Regression Model. Create a model that will help 
#him to estimate what the house would sell for.


# In[14]:


import pandas as pd

# Load the dataset
url = 'https://raw.githubusercontent.com/huzaifsayed/Linear-Regression-Model-for-House-Price-Prediction/master/USA_Housing.csv'
df = pd.read_csv(url)
# Show first few rows
df.head()


# In[15]:


# Step 2: Exploratory Data Analysis (EDA)

print(df.info())
print(df.describe())


# In[16]:


import seaborn as sns
import matplotlib.pyplot as plt

sns.pairplot(df)
plt.show()


# In[17]:


# Step 3: Data Preprocessing
# Drop 'Address' since it's non-numeric
X = df.drop(['Price', 'Address'], axis=1)
y = df['Price']


# In[18]:


# Step 4: Train-Test Split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[19]:



# Step 5: Train the Linear Regression Model

from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)


# In[20]:


# Step 6: Predict & Evaluate

from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

y_pred = model.predict(X_test)

print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R2 Score:", r2_score(y_test, y_pred))


# In[21]:


print(y_pred)


# In[22]:


# Select a sample from the test set  
index = 0  # Change index as needed
sample_input = X_test.iloc[[index]]  # Input features
actual_price = y_test.iloc[index]    # Ground truth label

# Predict using the trained model
predicted_price = model.predict(sample_input)[0]

# Print the results
print(f"Actual Price       : ${actual_price:,.2f}")
print(f"Predicted Price    : ${predicted_price:,.2f}")
print(f"Absolute Error     : ${abs(predicted_price - actual_price):,.2f}")
print(f"Percentage Error   : {abs(predicted_price - actual_price) / actual_price * 100:.2f}%")


# In[23]:


# Predict on the entire test set
y_pred = model.predict(X_test)

# Convert to NumPy arrays if not already
actual = y_test.values
predicted = y_pred

# Calculate absolute errors
absolute_errors = abs(predicted - actual)

# Calculate percentage errors
percentage_errors = (absolute_errors / actual) * 100

# Calculate mean absolute error and mean percentage error
mean_absolute_error = np.mean(absolute_errors)
mean_percentage_error = np.mean(percentage_errors)

# Print metrics
print(f"Mean Absolute Error       : ${mean_absolute_error:,.2f}")
print(f"Mean Percentage Error     : {mean_percentage_error:.2f}%")
print(f"R2 Score (overall fit)    : {r2_score(actual, predicted):.4f}")


# In[24]:


import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
plt.scatter(actual, predicted, alpha=0.6)
plt.plot([actual.min(), actual.max()], [actual.min(), actual.max()], 'r--')
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs. Predicted House Prices")
plt.grid(True)
plt.show()


# In[ ]:





# In[ ]:




