#!/usr/bin/env python
# coding: utf-8

# # Exploratory Data Analysis in Pandas

# ## Data Pre Processing

# ### Install and Import package
# #### If don't have the package already, please install first
# #### how to install: !pip install pandas
# #### how to import: Import package

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm


# #### In this section we will explore the dataset
# 
# #### Read the data and look at the first 5 rows using the head method. There are two datasets, including coffee and coffe cost dataset

# In[2]:


## Read dataset1
coffee = pd.read_csv("coffee.csv")
coffee.head()


# In[3]:


## Read dataset2
coffee_cost = pd.read_csv("coffee_cost.csv")
coffee_cost.head()


# In[4]:


## Merge two datasets together using smae column "recordID"
coffee_merged = coffee.merge(coffee_cost, how = "inner", on="recordID")
coffee_merged


# ### Data Summary

# In[5]:


coffee_merged.shape


# #### By using .shape, we can tell that there are 3000 rows and 15 columns
# 
# #### Method describe shows the main statistical characteristics of the dataset for each numerical feature (for this dataset only contain int64): the existing values number, mean, standard deviation, range, min & max, 0.25, 0.5 and 0.75 quartiles

# In[6]:


coffee_merged.describe()


# #### The dataset don't have any missing values, so there is no need to fill the gap.
# #### The dataset contains 1 object and 14 int64 features

# In[7]:


coffee_merged.info()


# ### Handling Missing Data
# 
# #### Double check if there are any missing data

# In[8]:


coffee_merged.isna().sum()


# ### Data Visualization
# 
# #### Since this dataset contains multiple numerical data, we could take a look about distribution by using histogram

# In[9]:


## Distribution of numeric columns
numeric_cols = ['Profit', 'Margin', 'Sales', 'Inventory', 'Budget.Profit', 'Budget.COGS', 'Budget.Margin', 'Budget.Sales']
for col in numeric_cols:
    plt.hist(coffee_merged[col], bins=20)
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.title(f'Histogram of {col}')
    plt.show()


# #### Based on the histogram, we can tell that profit, margin and inventory is similar to normal distribution. 
# #### While others features are all skewed distribution.
# 
# ### Find out if there is any correlation between features using pairplot and corr()

# In[10]:


## Pairplot to visualize relationships between numeric variables
sns.pairplot(coffee_merged[numeric_cols])
plt.show()


# #### Using heatmap to see sepcific correlation number between features
# #### Based on the result, we could tell that most feature are correlated except inventory since those features are related to profit/margin/sales.

# In[11]:


correlation_matrix = coffee_merged[numeric_cols].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()


# ## Exploratoy Analysis
# 
# ### Total Sales by month

# In[12]:


# Convert the DateTableau column to a datetime format
coffee_merged["DateTableau"] = pd.to_datetime(coffee_merged["DateTableau"])

# Extract the month from the DateTableau column and create a new column
coffee_merged["Month"] = coffee_merged["DateTableau"].dt.month

# Group the data by Month and Sales to calculate the total sales
monthly_product_count = coffee_merged[["Sales", "Month"]].groupby(["Month"]).sum()
monthly_product_count


# In[13]:


# Plot a bar chart to compare total sales by month
monthly_product_count.plot(kind='bar', stacked=True)
plt.xlabel('Month')
plt.ylabel('Count')
plt.title('Total Sales by Month')
plt.show()


# ### The top 5 product (by total sales) sales by month

# In[14]:


# Group the data b Productid, Month and calculate the sum of sales and profit
product_monthly_summary = coffee_merged.groupby(['ProductId', 'Month']).agg({
    'Sales': 'sum',
    'Profit': 'sum'
}).reset_index()
product_monthly_summary


# In[15]:


# Calculate the total sales for each product
product_total = coffee_merged.groupby("ProductId")["Sales"].sum().reset_index()
product_total


# In[16]:


# Sort the products by total sales in descending order and get the top 5 products
top_5 = product_total.nlargest(5, "Sales")
top_5


# In[17]:


# Filter the product_monthly_summary for the top 5 products
top5_summary = product_monthly_summary[product_monthly_summary["ProductId"].isin(top_5["ProductId"])]
top5_summary


# In[18]:


# Create a pivot table to prepare data for plotting
pivot_table = top5_summary.pivot(index="Month", columns = "ProductId", values="Sales")
pivot_table


# In[19]:


# Plot a line chart to compare the monthly sales of the top 5 products
pivot_table.plot(kind='line', marker='o')
plt.xlabel('Month')
plt.ylabel('Sales')
plt.title('Monthly Sales of Top 5 Products')
plt.show()


# ### Profit Margin Analysis:
# ### Calculate the average profit margin for each product and visualize it.
# ### Identify products with the highest and lowest profit margins.

# In[20]:


# Calculate the average profit margin for each product
average_profit_margin = coffee_merged.groupby('ProductId')['Margin'].mean().reset_index()

# Sort the products by average profit margin in descending order
average_profit_margin_sorted = average_profit_margin.sort_values(by='Margin', ascending=False)

# Identify products with the highest and lowest profit margins
highest_profit_margin_product = average_profit_margin_sorted.iloc[0]
lowest_profit_margin_product = average_profit_margin_sorted.iloc[-1]

# Print the products with the highest and lowest profit margins
print(f"Highest Profit Margin Product (Product ID {highest_profit_margin_product['ProductId']}): {highest_profit_margin_product['Margin']:.2f}")
print(f"Lowest Profit Margin Product (Product ID {lowest_profit_margin_product['ProductId']}): {lowest_profit_margin_product['Margin']:.2f}")

# Visualize the average profit margin for each product
plt.figure(figsize=(10, 6))
plt.bar(average_profit_margin_sorted['ProductId'], average_profit_margin_sorted['Margin'])
plt.xlabel('Product ID')
plt.ylabel('Average Profit Margin')
plt.title('Average Profit Margin by Product')
plt.xticks(rotation=45)
plt.show()


# ### Inventory Trends:
# 
# ### Analyze the trend in inventory over time. Are there any noticeable patterns or fluctuations?
# ### Identify products with consistently high or low inventory levels.

# In[21]:


# Convert the DateTableau column to a datetime format
coffee_merged["DateTableau"] = pd.to_datetime(coffee_merged["DateTableau"])

# Group the data by ProductId and DataTableau, and calculate the mean inventory
inventory_trend = coffee_merged.groupby(["ProductId", "DateTableau"])["Inventory"].mean().reset_index()

# Identify products with consistently high or low inventory levels
product_inventory_summary = inventory_trend.groupby('ProductId')['Inventory'].agg(['mean', 'std']).reset_index()

# Set a threshold to determine what is considered high or low inventory
threshold = 1000

# Identify products with consistently high or low inventory levels
high_inventory_products = product_inventory_summary[product_inventory_summary['mean'] > threshold].sort_values("mean",ascending=False)
low_inventory_products = product_inventory_summary[product_inventory_summary['mean'] <= threshold].sort_values("mean")

# Visualize the inventory trend for a specific product (you can choose one from the high or low inventory products)
product_id_to_visualize = high_inventory_products["ProductId"].iloc[0]
product_inventory_data  = inventory_trend[inventory_trend['ProductId'] == product_id_to_visualize]


# #### List out the highest mean inventory product and see the trend

# In[22]:


# Plot the inventory trend for the selected product
plt.figure(figsize=(12, 6))
plt.plot(product_inventory_data['DateTableau'], product_inventory_data['Inventory'], marker='o', linestyle='-')
plt.xlabel('Date')
plt.ylabel('Inventory')
plt.title(f'Inventory Trend for Product ID {product_id_to_visualize}')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()


# #### List out the lowest mean inventory product and see the trend

# In[23]:


# Visualize the inventory trend for a specific product (you can choose one from the high or low inventory products)
product_id_to_visualize = low_inventory_products["ProductId"].iloc[0]
product_inventory_data  = inventory_trend[inventory_trend['ProductId'] == product_id_to_visualize]

# Plot the inventory trend for the selected product
plt.figure(figsize=(12, 6))
plt.plot(product_inventory_data['DateTableau'], product_inventory_data['Inventory'], marker='o', linestyle='-')
plt.xlabel('Date')
plt.ylabel('Inventory')
plt.title(f'Inventory Trend for Product ID {product_id_to_visualize}')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()


# ### Budget vs. Actual Analysis:
# 
# ### Calculate and visualize the variance between budgeted and actual sales, profit, and margin.
# ### Identify products where actual performance significantly deviates from the budget.

# In[24]:


# Calculate the variances for sales, profit, and margin
coffee_merged['Sales_Variance'] = coffee_merged['Sales'] - coffee_merged['Budget.Sales']
coffee_merged['Profit_Variance'] = coffee_merged['Profit'] - coffee_merged['Budget.Profit']
coffee_merged['Margin_Variance'] = coffee_merged['Margin'] - coffee_merged['Budget.Margin']

# Define a threshold for significant deviation
threshold = 10  # You can adjust this threshold based on your data

# Identify products where actual performance significantly deviates from the budget
significant_deviation_products = coffee_merged[
    (abs(coffee_merged['Sales_Variance']) > threshold) |
    (abs(coffee_merged['Profit_Variance']) > threshold) |
    (abs(coffee_merged['Margin_Variance']) > threshold)
]


# In[25]:


# Create three subplots for sales, profit, and margin variances
fig, axes = plt.subplots(3, 1, figsize=(10, 12))
fig.subplots_adjust(hspace=0.5)

# Sales Variance subplot
axes[0].bar(coffee_merged['ProductId'], coffee_merged['Sales_Variance'], color='blue', label='Sales Variance')
axes[0].set_ylabel('Sales Variance')
axes[0].set_title('Sales Variance by Product')

# Profit Variance subplot
axes[1].bar(coffee_merged['ProductId'], coffee_merged['Profit_Variance'], color='green', label='Profit Variance')
axes[1].set_ylabel('Profit Variance')
axes[1].set_title('Profit Variance by Product')

# Margin Variance subplot
axes[2].bar(coffee_merged['ProductId'], coffee_merged['Margin_Variance'], color='orange', label='Margin Variance')
axes[2].set_ylabel('Margin Variance')
axes[2].set_title('Margin Variance by Product')

plt.xlabel('Product ID')
plt.show()


# ### Marketing Impact:
# 
# ### Analyze the correlation between marketing expenses and sales/profit for different products.
# ### Determine which products show the strongest correlation.

# In[26]:


# Calculate the correlation matrix between marketing expenses, sales, and profit for different products
correlation_matrix = coffee_merged.groupby('ProductId')[['Marketing', 'Sales', 'Profit']].corr().unstack()['Marketing']

# Get the products with the strongest correlation between marketing expenses and sales/profit
strongest_correlation_product_sales = correlation_matrix['Sales'].idxmax()
strongest_correlation_product_profit = correlation_matrix['Profit'].idxmax()


# In[27]:


# Print the product with the strongest correlation
print("Product with the Strongest Correlation Between Marketing Expenses and Sales:", strongest_correlation_product_sales)
print("Product with the Strongest Correlation Between Marketing Expenses and Profit:", strongest_correlation_product_profit)


# In[28]:


# Visualize the correlation between marketing expenses and sales for the product with the strongest correlation (Sales)
product_data_sales = coffee_merged[coffee_merged['ProductId'] == strongest_correlation_product_sales]

plt.figure(figsize=(10, 6))
sns.scatterplot(data=product_data_sales, x='Marketing', y='Sales')
plt.xlabel('Marketing Expenses')
plt.ylabel('Sales')
plt.title(f'Correlation Between Marketing Expenses and Sales for Product ID {strongest_correlation_product_sales}')
plt.show()

