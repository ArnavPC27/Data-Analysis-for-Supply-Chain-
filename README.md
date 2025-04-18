# Data-Analysis-for-Supply-Chain-

# AntiAllergyZ Demand Forecasting Project

This repository contains a complete end-to-end demand forecasting project for the pharmaceutical product **AntiAllergyZ**, focusing on monthly sales data from 2020 to 2025.

##  Project Overview

The goal of this project is to understand historical sales patterns and build a reliable time series forecasting model using SARIMA. The analysis focuses on one product-region pair: **AntiAllergyZ – North**.

##  Key Components

- **Data Preparation**: Daily sales, promotions, competitor pricing, and weather variables were aggregated to monthly format.
- **Trend & Seasonality Analysis**: Clear upward trend with visible seasonal spikes, particularly during allergy-prone months.
- **Stationarity Testing**: ADF test confirmed the need for first differencing to stabilize the series.
- **Modeling**: A SARIMA(1,1,1)(1,1,1,12) model was fitted and evaluated.
- **Forecasting**: 3-month sales forecast was generated with confidence intervals.
- **Drivers of Demand**: Exploratory analysis on the impact of Marketing Spend, Discounts, Doctor Promotions, Online Ads, Temperature, and Humidity.

##  Insights

- **Sales have grown significantly** from 2022 to 2025, with peaks exceeding 8,000 units per month.
- ![Image](https://github.com/user-attachments/assets/d37379ac-8f15-41f7-b7b8-9fcf76a9774f)
- **Sharp sales spikes** were observed, likely triggered by promotions, doctor campaigns, or seasonal allergy cycles.
- **Online Ads and Weather** (Temperature, Humidity) showed weak correlation with sales.
- **Competitor pricing** had a moderate negative correlation, possibly due to shared seasonality rather than direct influence.
- **Marketing Spend and Discounts** are stronger contributors to demand, though results vary by timing and region.

##  Forecast Summary

The SARIMA model predicted stable demand over the next 3 months, with the following sales estimates:

- **July 2025**: ~1,930 units  
- **August 2025**: ~1,989 units  
- **September 2025**: ~1,738 units  

##  Files Included

- `notebook.ipynb` – Full modeling workflow with visualizations
- `data.csv` – Cleaned monthly dataset
- `forecast_plot.png` – 3-month forecast visualization
- `README.md` – Project summary and interpretation

##  How to Use

1. Clone the repository  
2. Install required Python libraries  
3. Run the notebook step by step to reproduce the analysis and forecast  
4. Adjust model parameters or time ranges as needed

---

Feel free to fork this project or reach out with questions or suggestions!

