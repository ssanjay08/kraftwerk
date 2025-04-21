# **Assignment 20.1 - Initial Report on Gold Price & Exploratory Data Analysis**

# **Project Overview**
This repository contains the initial report and exploratory data analysis (EDA) for my capstone project on gold price prediction. Gold prices are considered a safe deposit, which deeply interests me. I wanted to explore a comprehensive dataset on gold prices and use it to build a predictive model that considers economic factors influencing gold prices.

# **Problem Statement:**
 Develop a machine learning model to predict gold prices by analyzing the impact of various economic indicators.
* Data Needed:
* Economic indicators (e.g., inflation rates, interest rates, GDP)
* Historical gold prices
# **Techniques:**
* Multiple Linear Regression
* Random Forest
* Gradient Boosting
* Lasso Regression

# **Source of Data:**
Yahoo Finance, Kitco, Kaggle. I ended up using a dataset from Kaggle as it contains historical gold prices, financial information for some market indices, commodities, economic indicators, and forex rates. This dataset is suitable for my gold price analysis. Source dataset is available Kaggle link [Gold Price Regression](www.kaggle.com/datasets/franciscogcc/financial-data/data).

# **Key Features**
**Market Indices:**

* S&P 500: Includes opening, closing, high, low prices, and volume.
* NASDAQ:Includes opening, closing, high, low prices, and volume.

# **Economic Indicators:**

* Interest Rates (us_rates_%): Reflects the prevailing interest rates.
* Consumer Price Index (CPI): Measures inflation.

#**Forex Rates:**

* USD/CHF: Exchange rate between US Dollar and Swiss Franc.
* EUR/USD: Exchange rate between Euro and US Dollar.

# **Commodities:**

* Silver: Includes opening, closing, high, low prices, and volume.
* Oil: Includes opening, closing, high, low prices, and volume.
* Platinum: Includes opening, closing, high, low prices, and volume.
* Palladium: Includes opening, closing, high, low prices, and volume.

# **Gold Prices:**
* Gold: Includes opening, closing, high, low prices, and volume.


```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split , GridSearchCV, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

```

Read data set


```python
# Load the dataset
from google.colab import drive
drive.mount('/content/drive')
file_path = '/content/drive/My Drive/Assignment_20.1/financial_regression.csv'

#create Dataframe
data = pd.read_csv(file_path)


```

    Drive already mounted at /content/drive; to attempt to forcibly remount, call drive.mount("/content/drive", force_remount=True).
    


```python
#Preview Head
data.head()

```





  <div id="df-072ffbd8-ed26-47da-a4eb-eeacff8a0943" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>date</th>
      <th>sp500 open</th>
      <th>sp500 high</th>
      <th>sp500 low</th>
      <th>sp500 close</th>
      <th>sp500 volume</th>
      <th>sp500 high-low</th>
      <th>nasdaq open</th>
      <th>nasdaq high</th>
      <th>nasdaq low</th>
      <th>...</th>
      <th>palladium high</th>
      <th>palladium low</th>
      <th>palladium close</th>
      <th>palladium volume</th>
      <th>palladium high-low</th>
      <th>gold open</th>
      <th>gold high</th>
      <th>gold low</th>
      <th>gold close</th>
      <th>gold volume</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2010-01-14</td>
      <td>114.49</td>
      <td>115.14</td>
      <td>114.42</td>
      <td>114.93</td>
      <td>115646960.0</td>
      <td>0.72</td>
      <td>46.26</td>
      <td>46.520</td>
      <td>46.22</td>
      <td>...</td>
      <td>45.02</td>
      <td>43.86</td>
      <td>44.84</td>
      <td>364528.0</td>
      <td>1.16</td>
      <td>111.51</td>
      <td>112.37</td>
      <td>110.79</td>
      <td>112.03</td>
      <td>18305238.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2010-01-15</td>
      <td>114.73</td>
      <td>114.84</td>
      <td>113.20</td>
      <td>113.64</td>
      <td>212252769.0</td>
      <td>1.64</td>
      <td>46.46</td>
      <td>46.550</td>
      <td>45.65</td>
      <td>...</td>
      <td>45.76</td>
      <td>44.40</td>
      <td>45.76</td>
      <td>442210.0</td>
      <td>1.36</td>
      <td>111.35</td>
      <td>112.01</td>
      <td>110.38</td>
      <td>110.86</td>
      <td>18000724.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2010-01-18</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2010-01-19</td>
      <td>113.62</td>
      <td>115.13</td>
      <td>113.59</td>
      <td>115.06</td>
      <td>138671890.0</td>
      <td>1.54</td>
      <td>45.96</td>
      <td>46.640</td>
      <td>45.95</td>
      <td>...</td>
      <td>47.08</td>
      <td>45.70</td>
      <td>46.94</td>
      <td>629150.0</td>
      <td>1.38</td>
      <td>110.95</td>
      <td>111.75</td>
      <td>110.83</td>
      <td>111.52</td>
      <td>10467927.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2010-01-20</td>
      <td>114.28</td>
      <td>114.45</td>
      <td>112.98</td>
      <td>113.89</td>
      <td>216330645.0</td>
      <td>1.47</td>
      <td>46.27</td>
      <td>46.604</td>
      <td>45.43</td>
      <td>...</td>
      <td>47.31</td>
      <td>45.17</td>
      <td>47.05</td>
      <td>643198.0</td>
      <td>2.14</td>
      <td>109.97</td>
      <td>110.05</td>
      <td>108.46</td>
      <td>108.94</td>
      <td>17534231.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 47 columns</p>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-072ffbd8-ed26-47da-a4eb-eeacff8a0943')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-072ffbd8-ed26-47da-a4eb-eeacff8a0943 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-072ffbd8-ed26-47da-a4eb-eeacff8a0943');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-16ad3e14-a381-4c70-918e-93b46379f31c">
  <button class="colab-df-quickchart" onclick="quickchart('df-16ad3e14-a381-4c70-918e-93b46379f31c')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-16ad3e14-a381-4c70-918e-93b46379f31c button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

    </div>
  </div>





```python
# Check data types
print(data.dtypes)
```

    date                   object
    sp500 open            float64
    sp500 high            float64
    sp500 low             float64
    sp500 close           float64
    sp500 volume          float64
    sp500 high-low        float64
    nasdaq open           float64
    nasdaq high           float64
    nasdaq low            float64
    nasdaq close          float64
    nasdaq volume         float64
    nasdaq high-low       float64
    us_rates_%            float64
    CPI                   float64
    usd_chf               float64
    eur_usd               float64
    GDP                   float64
    silver open           float64
    silver high           float64
    silver low            float64
    silver close          float64
    silver volume         float64
    silver high-low       float64
    oil open              float64
    oil high              float64
    oil low               float64
    oil close             float64
    oil volume            float64
    oil high-low          float64
    platinum open         float64
    platinum high         float64
    platinum low          float64
    platinum close        float64
    platinum volume       float64
    platinum high-low     float64
    palladium open        float64
    palladium high        float64
    palladium low         float64
    palladium close       float64
    palladium volume      float64
    palladium high-low    float64
    gold open             float64
    gold high             float64
    gold low              float64
    gold close            float64
    gold volume           float64
    dtype: object
    


```python
data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 3904 entries, 0 to 3903
    Data columns (total 47 columns):
     #   Column              Non-Null Count  Dtype  
    ---  ------              --------------  -----  
     0   date                3904 non-null   object 
     1   sp500 open          3719 non-null   float64
     2   sp500 high          3719 non-null   float64
     3   sp500 low           3719 non-null   float64
     4   sp500 close         3719 non-null   float64
     5   sp500 volume        3719 non-null   float64
     6   sp500 high-low      3719 non-null   float64
     7   nasdaq open         3719 non-null   float64
     8   nasdaq high         3719 non-null   float64
     9   nasdaq low          3719 non-null   float64
     10  nasdaq close        3719 non-null   float64
     11  nasdaq volume       3719 non-null   float64
     12  nasdaq high-low     3719 non-null   float64
     13  us_rates_%          176 non-null    float64
     14  CPI                 176 non-null    float64
     15  usd_chf             3694 non-null   float64
     16  eur_usd             3694 non-null   float64
     17  GDP                 57 non-null     float64
     18  silver open         3719 non-null   float64
     19  silver high         3719 non-null   float64
     20  silver low          3719 non-null   float64
     21  silver close        3719 non-null   float64
     22  silver volume       3719 non-null   float64
     23  silver high-low     3719 non-null   float64
     24  oil open            3719 non-null   float64
     25  oil high            3719 non-null   float64
     26  oil low             3719 non-null   float64
     27  oil close           3719 non-null   float64
     28  oil volume          3719 non-null   float64
     29  oil high-low        3719 non-null   float64
     30  platinum open       3719 non-null   float64
     31  platinum high       3719 non-null   float64
     32  platinum low        3719 non-null   float64
     33  platinum close      3719 non-null   float64
     34  platinum volume     3719 non-null   float64
     35  platinum high-low   3719 non-null   float64
     36  palladium open      3719 non-null   float64
     37  palladium high      3719 non-null   float64
     38  palladium low       3719 non-null   float64
     39  palladium close     3719 non-null   float64
     40  palladium volume    3719 non-null   float64
     41  palladium high-low  3719 non-null   float64
     42  gold open           3719 non-null   float64
     43  gold high           3719 non-null   float64
     44  gold low            3719 non-null   float64
     45  gold close          3719 non-null   float64
     46  gold volume         3719 non-null   float64
    dtypes: float64(46), object(1)
    memory usage: 1.4+ MB
    

# **Data Preparation**
**Handling Missing Values:**
* Dropped rows with missing values to ensure a clean dataset.

**Feature Engineering:**
* Selected relevant financial indicators as features.
Chose the gold closing price as the target variable.

**Data Transformation:**
* Standardized the features using StandardScaler.
* Split the data into training and testing sets.


```python
# Drop rows with missing values
data_cleaned = data.dropna()

# Select features and target variable
features = data_cleaned[['sp500 open', 'sp500 high', 'sp500 low', 'sp500 close', 'sp500 volume', 'sp500 high-low',
                         'nasdaq open', 'nasdaq high', 'nasdaq low', 'nasdaq close', 'nasdaq volume', 'nasdaq high-low',
                         'us_rates_%', 'CPI', 'usd_chf', 'eur_usd', 'GDP', 'silver open', 'silver high', 'silver low',
                         'silver close', 'silver volume', 'silver high-low', 'oil open', 'oil high', 'oil low', 'oil close',
                         'oil volume', 'oil high-low', 'platinum open', 'platinum high', 'platinum low', 'platinum close',
                         'platinum volume', 'platinum high-low', 'palladium open', 'palladium high', 'palladium low',
                         'palladium close', 'palladium volume']]
target = data_cleaned['gold close']
```

**Pairplot of Selected Features**
* Pairplot of selected features to visualize the relationships between them and the gold closing price.


```python

# Select relevant features for the plot
selected_features = ['sp500 close', 'nasdaq close', 'us_rates_%', 'CPI', 'usd_chf', 'eur_usd',
                     'silver close', 'oil close', 'platinum close', 'palladium close', 'gold close']

# Create a pairplot to show the relationships between Market Indices, Economic Indicators, Forex Rates, Commodities, and Gold Prices
sns.pairplot(data_cleaned[selected_features])
plt.suptitle('Relationships between Market Indices, Economic Indicators, Forex Rates, Commodities, and Gold Prices', y=1.02)
plt.show()

```


    
![png](output_12_0.png)
    





```python

# Standardize the features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)



# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=0.2, random_state=42)


# Initialize models

# Initialize models
models = {
  "Linear Regression": LinearRegression(),
  "Ridge Regression": Ridge(),
  "Lasso Regression": Lasso(max_iter=10000),
  "Random Forest Regression": RandomForestRegressor(),
  "Gradient Boosting Regression": GradientBoostingRegressor()
}


# Define hyperparameters for Grid Search
param_grid = {
   "Ridge Regression": {"alpha": [0.1, 1.0, 10.0]},
   "Lasso Regression": {"alpha": [0.01, 0.1, 1.0]},
   "Random Forest Regression": {"n_estimators": [100, 200], "max_depth": [None, 10, 20]},
   "Gradient Boosting Regression": {"n_estimators": [100, 200], "learning_rate": [0.01, 0.1]}
}


```


```python
# Train and evaluate models

# Perform Grid Search and cross-validation

# Train and evaluate models
results = {}
predictions = {}
for name, model in models.items():
  model.fit(X_train, y_train)
  y_pred = model.predict(X_test)
  mse = mean_squared_error(y_test, y_pred)
  results[name] = {"Mean Squared Error": mse, "R-squared": r2}
  predictions[name] = y_pred



# Print results
for name, result in results.items():
    print(f"{name}:")
    print(f"  Mean Squared Error: {result['Mean Squared Error']}")
    print(f"  R-squared: {result['R-squared']}")


```

    Linear Regression:
      Mean Squared Error: 29.19048919786371
      R-squared: 0.8862496222603973
    Ridge Regression:
      Mean Squared Error: 32.1929833969403
      R-squared: 0.8862496222603973
    Lasso Regression:
      Mean Squared Error: 58.39627000705779
      R-squared: 0.8862496222603973
    Random Forest Regression:
      Mean Squared Error: 175.1257800162508
      R-squared: 0.8862496222603973
    Gradient Boosting Regression:
      Mean Squared Error: 89.0388320307855
      R-squared: 0.8862496222603973
    


```python
# Print results
for name, result in results.items():
    print(f"{name}:")
    print(f"  Mean Squared Error: {result['Mean Squared Error']}")
    print(f"  R-squared: {result['R-squared']}")
```

    Linear Regression:
      Mean Squared Error: 29.19048919786371
      R-squared: 0.8862496222603973
    Ridge Regression:
      Mean Squared Error: 32.1929833969403
      R-squared: 0.8862496222603973
    Lasso Regression:
      Mean Squared Error: 58.39627000705779
      R-squared: 0.8862496222603973
    Random Forest Regression:
      Mean Squared Error: 175.1257800162508
      R-squared: 0.8862496222603973
    Gradient Boosting Regression:
      Mean Squared Error: 89.0388320307855
      R-squared: 0.8862496222603973
    

# **Key Findings**
**Positive Correlation:**
* Silver and platinum prices have strong positive correlations with gold prices.
* Oil prices show moderate positive correlations with gold prices.

**Negative Correlation:**
* Interest rates have a negative correlation with gold prices.

**Stock Market Indicators:**
* Stock market indices show weaker correlations with gold prices.


```python

# Plot predictions vs actual values for each model
plt.figure(figsize=(15, 10))
for i, (name, y_pred) in enumerate(predictions.items(), 1):
  plt.subplot(3, 2, i)
  plt.scatter(y_test, y_pred, alpha=0.5)
  plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r')
  plt.xlabel('Actual Gold Prices')
  plt.ylabel('Predicted Gold Prices')
  plt.title(f'{name} Predictions vs Actuals')
plt.tight_layout()
plt.show()

```


    
![png](output_18_0.png)
    


**Recommendations**

**Monitor Precious Metal Prices:**

**Silver and Platinum:**
* Since silver and platinum prices have strong positive correlations with gold prices, it is recommended to monitor these precious metals closely. They tend to move in tandem with gold prices.
Keep an Eye on Interest Rates:

**Interest Rates:**
* Higher interest rates may lead to lower gold prices. It is important to keep an eye on interest rate changes as they can significantly impact gold prices.
Consider the Impact of Oil Prices:

**Oil Prices:**
* Oil prices show moderate positive correlations with gold prices. During periods of economic uncertainty, consider the influence of oil prices on gold prices.

**Stock Market Indicators:**
Stock Market: While stock market indicators are important, they may not be the primary drivers of gold prices. Focus more on precious metals and interest rates.
