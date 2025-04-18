#!/usr/bin/env python
# coding: utf-8

# # DATA ANALYSIS FOR SUPPLY CHAIN

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import datetime as dt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis,QuadraticDiscriminantAnalysis
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing
from sklearn import model_selection
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import roc_auc_score,r2_score,mean_absolute_error,mean_squared_error,accuracy_score,classification_report,confusion_matrix
from sklearn.model_selection import train_test_split,cross_val_score, cross_val_predict
from sklearn import svm,metrics,tree,preprocessing,linear_model
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier, GradientBoostingRegressor
from sklearn.metrics import accuracy_score,mean_squared_error,recall_score,confusion_matrix,f1_score,roc_curve, auc
from plotly.offline import iplot, init_notebook_mode
import pickle
import warnings
warnings.filterwarnings("ignore") 
import datetime as dt
from datetime import datetime
import plotly.express as px


# In[2]:


SC=pd.read_csv("Expanded_Pharma_Eye_Drops_Dataset 2.csv", encoding='latin1')


# In[4]:


SC.shape


# In[6]:


SC.dtypes


# In[7]:


SC.head(10)


# In[8]:


SC.info()


# In[9]:


SC.describe()


# ## To see which product needs maximum shelf space

# In[22]:


# Correct aggregation of both columns
summary = SC.groupby('Product')['Shelf_Space', 'Doctor_Promotions','Competitor_Price'].agg(['mean', 'std'])

# Properly sort by the mean of Shelf_Space using MultiIndex column reference
summary_sorted = summary.sort_values(by=('Shelf_Space', 'mean'), ascending=False)

# Show result
print(summary_sorted)


# In[23]:


np.sum(SC.isna())


# # 1. Sales Patterns

# ## Which product has the highest and most stable demand?

# In[32]:


product_demand_stats = SC.groupby('Product')['Units_Sold'].agg(['mean', 'std']).reset_index()
product_demand_stats.columns = ['Product', 'Average_Daily_Sales', 'Sales_Variability']


# In[33]:


product_demand_stats['Stability_Score'] = product_demand_stats['Average_Daily_Sales'] / product_demand_stats['Sales_Variability']
product_demand_stats.sort_values(by='Stability_Score', ascending=False)



# ### We can see most popular product is AntiAllergyZ with 305 units per day. AntiAllergyZ is not only popular but also really stable(stability score = 2.5). We can see our sales variablility is around 121 which tells us the avearge sales per day would be between  184 and 426 units
# 

# ## How does demand vary by region?

# In[38]:


product_demand_stats = SC.groupby('Region')['Units_Sold'].agg(['mean', 'std']).reset_index()
product_demand_stats.columns = ['Region', 'Average_Daily_Sales', 'Sales_Variability']
product_demand_stats.sort_values(by='Average_Daily_Sales',ascending=False)


# ### We can see South has the highest average sales per day with 306 units sold each day. The actual sales could be anywhere between 184 and 428 units
# 

# ## Are there any visible trends, seasonality, or irregular spikes?

# In[47]:


import pandas as pd
from datetime import datetime

# Filter for the last 3 years from the most recent date in data
end_date = SC['Date'].max()
start_date = end_date - pd.DateOffset(years=3)

SC_3yrs = SC[(SC['Date'] >= start_date) & (SC['Date'] <= end_date)]


# In[48]:


# Add a month column for grouping
SC_3yrs['Month'] = SC_3yrs['Date'].dt.to_period('M')

# Group by Month & Product
monthly_sales = SC_3yrs.groupby(['Month', 'Product'])['Units_Sold'].sum().reset_index()

# Convert Month back to datetime for plotting
monthly_sales['Month'] = monthly_sales['Month'].dt.to_timestamp()



# In[49]:


import matplotlib.pyplot as plt

products = monthly_sales['Product'].unique()

for product in products:
    df = monthly_sales[monthly_sales['Product'] == product]
    
    plt.figure(figsize=(12, 5))
    plt.plot(df['Month'], df['Units_Sold'], marker='o')
    plt.title(f'Monthly Sales – {product} (Last 3 Years)')
    plt.xlabel('Month')
    plt.ylabel('Units Sold')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# ### Sales of AntiAllergyZ have steadily increased from mid 2022 to 2025, rising from under 1,000 to over 8,000 units per month. However, several sharp spikes followed by sudden drops are visible. One clear example is the noticeable surge in early 2025.
# 
# ### These irregular swings may be driven by promotions, doctor campaigns, supply chain lags, or seasonal allergy events. Cross-referencing with Marketing Spend and Discount data can help explain these peaks and guide future planning.

# # 2. Marketing Effectiveness

# In[58]:


from sklearn.preprocessing import MinMaxScaler

# Scale all variables to 0–1 range
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(monthly_metrics[['Units_Sold', 'Marketing_Spend', 'Discount']])

# Convert back to DataFrame
scaled_df = pd.DataFrame(scaled_data, columns=['Units_Sold', 'Marketing_Spend', 'Discount'])
scaled_df['Month'] = monthly_metrics['Month']


# In[57]:


plt.figure(figsize=(14, 6))

plt.plot(scaled_df['Month'], scaled_df['Units_Sold'], label='Units Sold (scaled)', marker='o')
plt.plot(scaled_df['Month'], scaled_df['Marketing_Spend'], label='Marketing Spend (scaled)', linestyle='--', marker='s')
plt.plot(scaled_df['Month'], scaled_df['Discount'], label='Discount Rate (scaled)', linestyle=':', marker='^')

plt.title('AntiAllergyZ – Scaled Monthly Sales, Marketing, and Discount')
plt.xlabel('Month')
plt.ylabel('Scaled Value (0–1)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# ### Analysis of scaled monthly metrics shows a strong relationship between marketing spend and sales spikes for AntiAllergyZ. While discounting does appear to influence demand occasionally, its correlation is weaker and less consistent. Coordinated campaigns (where both discounting and marketing spike together) lead to the highest sales peaks, suggesting synergy between these levers.

# ## Is there a noticeable effect of online ads on units sold?

# In[59]:


# Filter for AntiAllergyZ in last 3 years
anti_df = SC[
    (SC['Product'] == 'AntiAllergyZ') &
    (SC['Date'] >= SC['Date'].max() - pd.DateOffset(years=3))
].copy()

# Add month column
anti_df['Month'] = anti_df['Date'].dt.to_period('M')

# Group by month
monthly_ads = anti_df.groupby('Month').agg({
    'Units_Sold': 'sum',
    'Online_Ads_Clicks': 'mean'  # assuming daily ad clicks → avg per month
}).reset_index()

monthly_ads['Month'] = monthly_ads['Month'].dt.to_timestamp()


# In[60]:


import matplotlib.pyplot as plt

fig, ax1 = plt.subplots(figsize=(14, 6))

# Units Sold
ax1.plot(monthly_ads['Month'], monthly_ads['Units_Sold'], label='Units Sold', color='blue', marker='o')
ax1.set_ylabel('Units Sold', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')
ax1.set_xlabel('Month')

# Online Ads on second y-axis
ax2 = ax1.twinx()
ax2.plot(monthly_ads['Month'], monthly_ads['Online_Ads_Clicks'], label='Online Ads Clicks', color='orange', linestyle='--', marker='s')
ax2.set_ylabel('Avg Monthly Online Ads Clicks', color='orange')
ax2.tick_params(axis='y', labelcolor='orange')

# Legends
fig.legend(loc='upper left')
plt.title('AntiAllergyZ – Units Sold vs Online Ads Clicks (Monthly)')
plt.grid(True)
plt.tight_layout()
plt.show()


# In[61]:


correlation = monthly_ads['Units_Sold'].corr(monthly_ads['Online_Ads_Clicks'])
print(f"Correlation between Online Ads and Units Sold: {correlation:.2f}")


# In[62]:


import seaborn as sns

sns.regplot(x='Online_Ads_Clicks', y='Units_Sold', data=monthly_ads)
plt.title('Scatter Plot: Online Ads vs Units Sold')
plt.show()


# ### The correlation between Online Ads Clicks and Units Sold for AntiAllergyZ is weak (r = 0.17), suggesting that online advertising has limited direct influence on driving sales for this product. This aligns with the nature of the product, AntiAllergyZ is likely consumed as needed, meaning purchases are driven more by seasonal or health related demand rather than promotional awareness.

# ## Marketing Drivers of Demand: Regression Analysis for AntiAllergyZ (Last 3 Years)

# In[65]:


import pandas as pd
import statsmodels.api as sm

# Step 1: Filter for AntiAllergyZ (last 3 years)
anti_df = SC[
    (SC['Product'] == 'AntiAllergyZ') &
    (SC['Date'] >= SC['Date'].max() - pd.DateOffset(years=3))
].copy()

# Add monthly period
anti_df['Month'] = anti_df['Date'].dt.to_period('M')

# Step 2: Group monthly data
monthly_data = anti_df.groupby('Month').agg({
    'Units_Sold': 'sum',
    'Marketing_Spend': 'mean',
    'Discount': 'mean',
    'Online_Ads_Clicks': 'mean',
    'Doctor_Promotions': 'mean'
}).reset_index()

# Convert Month to timestamp
monthly_data['Month'] = monthly_data['Month'].dt.to_timestamp()

# Step 3: Define features (X) and target (y)
X = monthly_data[['Marketing_Spend', 'Discount', 'Online_Ads_Clicks', 'Doctor_Promotions']]
y = monthly_data['Units_Sold']

# Add intercept term
X = sm.add_constant(X)

# Step 4: Fit the regression model
model = sm.OLS(y, X).fit()

# Step 5: Display summary
print(model.summary())


# ### None of the tested marketing variables,  including Doctor Promotions, Marketing Spend, Discount, or Online Ads,  show a statistically significant relationship with monthly sales of AntiAllergyZ. This supports the idea that the product's demand is driven more by real-world need (like seasonal allergies) rather than promotional efforts.

# ## Regression Analysis: Evaluating the Impact of Lagged Marketing Variables on AntiAllergyZ Sales

# In[66]:


anti_df['Month'] = anti_df['Date'].dt.to_period('M')
monthly_data = anti_df.groupby('Month').agg({
    'Units_Sold': 'sum',
    'Marketing_Spend': 'mean',
    'Discount': 'mean',
    'Online_Ads_Clicks': 'mean',
    'Doctor_Promotions': 'mean'
}).reset_index()

# Convert to Timestamp
monthly_data['Month'] = monthly_data['Month'].dt.to_timestamp()


# In[67]:


# Create a lag of 1 period for Doctor_Promotions
monthly_data['Doctor_Promotions_Lag1'] = monthly_data['Doctor_Promotions'].shift(1)


# In[68]:


import statsmodels.api as sm

# Drop first row (it will have NaN due to shift)
reg_data = monthly_data.dropna(subset=['Doctor_Promotions_Lag1'])

# Define X and y
X = reg_data[['Marketing_Spend', 'Discount', 'Online_Ads_Clicks', 'Doctor_Promotions_Lag1']]
y = reg_data['Units_Sold']
X = sm.add_constant(X)

# Fit model
model = sm.OLS(y, X).fit()
print(model.summary())


# ### Adding a 1-month lag for Doctor Promotions did not yield a statistically significant relationship with Units Sold (p = 0.46). This suggests that promotional efforts by doctors do not have a delayed impact on sales volume for AntiAllergyZ, at least not in a linear or immediate monthly pattern.

# # 3. External Influences

# In[71]:


# Filter for AntiAllergyZ
anti_df = SC[
    (SC['Product'] == 'AntiAllergyZ') &
    (SC['Date'] >= SC['Date'].max() - pd.DateOffset(years=3))
].copy()

# Add Month column
anti_df['Month'] = anti_df['Date'].dt.to_period('M')

# Group monthly metrics including weather
monthly_weather = anti_df.groupby('Month').agg({
    'Units_Sold': 'sum',
    'Temperature': 'mean',
    'Humidity': 'mean'
}).reset_index()

# Convert to timestamp for plotting
monthly_weather['Month'] = monthly_weather['Month'].dt.to_timestamp()


# In[72]:


corr_temp = monthly_weather['Units_Sold'].corr(monthly_weather['Temperature'])
corr_hum = monthly_weather['Units_Sold'].corr(monthly_weather['Humidity'])

print(f"Correlation with Temperature: {corr_temp:.2f}")
print(f"Correlation with Humidity: {corr_hum:.2f}")


# In[73]:


import seaborn as sns
import matplotlib.pyplot as plt

# Temp vs Units Sold
sns.regplot(x='Temperature', y='Units_Sold', data=monthly_weather)
plt.title('Temperature vs Units Sold')
plt.show()

# Humidity vs Units Sold
sns.regplot(x='Humidity', y='Units_Sold', data=monthly_weather)
plt.title('Humidity vs Units Sold')
plt.show()


# ### Temperature shows virtually no correlation with monthly sales of AntiAllergyZ, while humidity demonstrates a weak positive relationship (r = 0.17). This suggests that general weather patterns such as heat or moisture are not strong drivers of demand for the product.
# 
# ### Temperature does not appear to trigger purchasing behavior, and although humidity may have a slight influence, the relationship is too weak to draw confident conclusions. It's more likely that sales are influenced by allergy-specific environmental conditions (such as pollen levels), seasonal cycles (e.g., spring and fall allergy peaks), and real-time consumer need, rather than broad climate variables like temperature or humidity

# ## Are sales influenced by competitor pricing?

# In[74]:


# Filter for AntiAllergyZ
anti_df = SC[
    (SC['Product'] == 'AntiAllergyZ') &
    (SC['Date'] >= SC['Date'].max() - pd.DateOffset(years=3))
].copy()

# Add month column
anti_df['Month'] = anti_df['Date'].dt.to_period('M')

# Group by month and include competitor pricing
monthly_comp = anti_df.groupby('Month').agg({
    'Units_Sold': 'sum',
    'Competitor_Price': 'mean'
}).reset_index()

monthly_comp['Month'] = monthly_comp['Month'].dt.to_timestamp()


# In[75]:


corr_competitor = monthly_comp['Units_Sold'].corr(monthly_comp['Competitor_Price'])
print(f"Correlation with Competitor Price: {corr_competitor:.2f}")


# In[76]:


import seaborn as sns
import matplotlib.pyplot as plt

sns.regplot(x='Competitor_Price', y='Units_Sold', data=monthly_comp)
plt.title('Competitor Price vs Units Sold')
plt.xlabel('Average Competitor Price')
plt.ylabel('Units Sold')
plt.grid(True)
plt.tight_layout()
plt.show()


# ### There is a moderate negative correlation (r = -0.35) between competitor pricing and our units sold for AntiAllergyZ. However, this relationship may be misleading due to underlying seasonal dynamics. It’s likely that both we and competing brands adjust pricing in response to peak allergy seasons, which means we're responding to shared demand patterns rather than directly impacting one another. Additionally, higher competitor prices may reflect factors such as premium positioning, inventory constraints, or increased demand that draws consumers away from our offering. In some instances, market-wide price increases can also lead to an overall decline in demand. As a result, this correlation appears to be more coincidental than causal and should be interpreted with caution.

# # 4.	Operational Insights

# In[77]:


shelf_utilization = SC.groupby('Product').agg({
    'Units_Sold': 'sum',
    'Shelf_Space': 'mean'  # or 'sum', depending on your logic
}).reset_index()

# Calculate efficiency: sales per square foot
shelf_utilization['Units_Per_SqFt'] = shelf_utilization['Units_Sold'] / shelf_utilization['Shelf_Space']
shelf_utilization = shelf_utilization.sort_values(by='Units_Per_SqFt', ascending=False)
shelf_utilization


# ### Based on the table, AntiAllergyZ achieves the highest Units Per SqFt at approximately 142,022, compared to LubricantX at about 138,269 and RednessReliefQ at around 128,169. This means that for each square foot of shelf space allocated, AntiAllergyZ generates more sales than the other products. In other words, its shelf utilization is the best among the three, indicating it is the most efficient in turning shelf space into sales.

# # Forecasting for 3 months 

# In[78]:


import pandas as pd

# Ensure Date is in datetime format
SC['Date'] = pd.to_datetime(SC['Date'])

# Filter for one product-region pair (AntiAllergyZ - North)
df = SC[
    (SC['Product'] == 'AntiAllergyZ') &
    (SC['Region'] == 'North')
].copy()

# Group daily data to monthly totals
df['Month'] = df['Date'].dt.to_period('M')
monthly_sales = df.groupby('Month')['Units_Sold'].sum().reset_index()
monthly_sales['Month'] = monthly_sales['Month'].dt.to_timestamp()
monthly_sales.set_index('Month', inplace=True)

# Final time series
ts = monthly_sales['Units_Sold']


# In[79]:


import matplotlib.pyplot as plt

ts.plot(title='Monthly Sales – AntiAllergyZ (North)', figsize=(12,5))
plt.ylabel('Units Sold')
plt.grid(True)
plt.show()


# In[80]:


from statsmodels.tsa.stattools import adfuller

adf_result = adfuller(ts)
print(f"ADF Statistic: {adf_result[0]}")
print(f"p-value: {adf_result[1]}")


# ### This means our time series is not stationary, the p-value is much greater than 0.05, so we cannot reject the null hypothesis of a unit root.
# 

# In[82]:


ts_diff = ts.diff().dropna()


# In[83]:


from statsmodels.tsa.stattools import adfuller
adf_result_diff = adfuller(ts_diff)
print(f"ADF Statistic after differencing: {adf_result_diff[0]}")
print(f"p-value after differencing: {adf_result_diff[1]}")


# ### The p-value < 0.05 means we can reject the null hypothesis,  no unit root,  so our series is ready for ARIMA or SARIMA modeling.

# ## Fitting SARIMA Model

# In[86]:


from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt

# Plot ACF and PACF for the differenced series
plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
plot_acf(ts.diff().dropna(), lags=20, ax=plt.gca())
plt.title('ACF')

plt.subplot(1, 2, 2)
plot_pacf(ts.diff().dropna(), lags=20, ax=plt.gca())
plt.title('PACF')
plt.tight_layout()
plt.show()


# In[87]:


from statsmodels.tsa.statespace.sarimax import SARIMAX

model = SARIMAX(ts, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
results = model.fit()
print(results.summary())


# ### The SARIMA(1,1,1)(1,1,1,12) model fits reasonably well for AntiAllergyZ’s monthly sales in the North region. The residuals show no autocorrelation and are normally distributed, which supports the model’s validity. The strongest contributing factor is the short-term moving average (MA) component, suggesting that recent past errors have a strong influence on current sales forecasts. However, the seasonal components and autoregressive terms are not statistically significant in this model. If needed, we can simplify or tune the model further.

# In[89]:


# Forecast next 3 months
forecast = results.get_forecast(steps=3)

# Predicted values
forecast_mean = forecast.predicted_mean
conf_int = forecast.conf_int()


# In[91]:


# Forecast the next 3 months
forecast = results.get_forecast(steps=3)

# Extract predicted values and confidence intervals
forecast_mean = forecast.predicted_mean
conf_int = forecast.conf_int()

# === Plot actual + forecast ===
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 5))

# Plot historical data
plt.plot(ts, label='Observed', color='blue')

# Plot forecasted data
plt.plot(forecast_mean, label='Forecast (Next 3 Months)', color='orange')
plt.fill_between(conf_int.index, conf_int.iloc[:, 0], conf_int.iloc[:, 1],
                 color='orange', alpha=0.3, label='95% Confidence Interval')

# Formatting
plt.title('AntiAllergyZ (North) – 3-Month Sales Forecast')
plt.xlabel('Month')
plt.ylabel('Units Sold')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# === Print Forecasted Values ===
print("Forecasted Units Sold (Next 3 Months):")
print(forecast_mean.round(0))


# ### Based on historical sales trends and seasonal patterns, AntiAllergyZ is projected to sell approximately 1,930 units in July, 1,989 in August, and 1,738 in September 2025 in the North region. These forecasts account for seasonality and recent demand behavior. The model shows high confidence with relatively narrow prediction intervals, suggesting a stable short-term outlook

# In[ ]:




