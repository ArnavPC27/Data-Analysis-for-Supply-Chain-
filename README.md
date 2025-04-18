# 💊 Demand Forecasting for AntiAllergyZ

This project presents an end-to-end analysis and time series forecasting model for **AntiAllergyZ**, a pharmaceutical eye-drop product. We investigate sales trends, identify demand drivers, and build a SARIMA model to forecast future demand with confidence.

---

## 📦 Objective

To analyze historical sales data of AntiAllergyZ in the North region, explore potential drivers of demand such as marketing spend, discounts, doctor promotions, and weather, and build a reliable model to forecast future sales.

---

## 📈 1. Sales Trends & Seasonality

Sales have grown steadily from 2022 to 2025, rising from under 1,000 to over 8,000 units per month. Clear seasonal spikes are present, with sharp peaks followed by dips, often aligning with known allergy periods.

![Image](https://github.com/user-attachments/assets/d37379ac-8f15-41f7-b7b8-9fcf76a9774f)

---

## 🧠 2. Investigating Demand Drivers

We explored the impact of:
- Marketing Spend
- Discounts
- Online Ads
- Doctor Promotions (including 1-month lag)
- Temperature & Humidity
- Competitor Pricing

### 📊 Marketing Spend vs Sales
![Image](https://github.com/user-attachments/assets/7cc0c7c1-51d5-40a7-8f38-9a8bfed7136d)

### 📊 Discount vs Sales
![Discount vs Sales](discount_vs_sales.png)

**Key Insight**: Marketing spend and discounts occasionally align with spikes in sales, but most variables showed weak or statistically insignificant relationships.

---

## 🧪 3. Regression Analysis

We ran a multiple linear regression using monthly aggregated data over 3 years. Doctor promotions (including with a 1-month lag) and online ads showed no significant predictive power. The model explained only 17% of the variation in sales.

```text
R-squared: 0.17
Adjusted R-squared: 0.065
P-values: None < 0.05
Conclusion: No strong statistical relationship found

# 💊 AntiAllergyZ Demand Forecasting

This project presents an end-to-end demand forecasting solution for **AntiAllergyZ**, a pharmaceutical eye-drop product. The analysis includes trend exploration, regression testing of promotional drivers, and a SARIMA time series model to forecast future demand.

---

## 📉 4. SARIMA Forecasting

Using **SARIMA(1,1,1)(1,1,1,12)**, we built a seasonal time series model to forecast the next 3 months of sales.

![Sales Forecast](sales_forecast.png)

### Forecasted Sales:
- **July 2025**: ~1,930 units  
- **August 2025**: ~1,989 units  
- **September 2025**: ~1,738 units  

---

## 📎 5. Model Diagnostics

### 📉 Residuals from SARIMA  
Residuals appear randomly distributed with no autocorrelation.

![SARIMA Residuals](sarima_residuals.png)

### 🌀 ACF & PACF Plots  
These plots helped validate lag choices and confirm seasonality in the SARIMA model.

![ACF and PACF](acf_pacf.png)

---

## 📌 Final Takeaways

- Sales are primarily influenced by **seasonal cycles** and **real-time need**.
- **Marketing**, **discounts**, and **doctor promotions** showed inconsistent influence.
- **Online ads**, **weather**, and **competitor pricing** had minimal or no correlation.
- The SARIMA model offers a **stable short-term forecast** and effectively captures the product’s cyclical nature.

---

## 🛠️ Tools & Technologies

- **Python**: `pandas`, `matplotlib`, `statsmodels`
- **Modeling**: SARIMA time series forecasting
- **Other techniques**: Data Aggregation, Regression Analysis, Residual Diagnostics

---

## 📂 File Structure

