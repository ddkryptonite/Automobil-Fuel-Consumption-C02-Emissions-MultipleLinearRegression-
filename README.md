# Fuel Consumption and CO2 Emissions Analysis

This repository contains code to analyze the relationship between engine size, fuel consumption, and CO2 emissions of automobiles using a dataset from an online source. The analysis includes data downloading, preprocessing, exploratory data analysis, and implementing multiple linear regression to predict CO2 emissions.

## Table of Contents
- [Dataset](#dataset)
- [Dependencies](#dependencies)
- [Usage](#usage)
- [Code Overview](#code-overview)
  - [Downloading CSV File](#downloading-csv-file)
  - [Reading the Data](#reading-the-data)
  - [Selecting Features](#selecting-features)
  - [Plotting Emission Values](#plotting-emission-values)
  - [Training and Testing Data](#training-and-testing-data)
  - [Multiple Linear Regression](#multiple-linear-regression)
  - [Prediction](#prediction)
  - [Practice](#practice)
- [Results](#results)

## Dataset
The dataset used in this project is the FuelConsumptionCo2 dataset, which includes various features such as engine size, number of cylinders, fuel consumption in city and highway, and CO2 emissions.

## Dependencies
- pandas
- numpy
- requests
- matplotlib
- scikit-learn
- seaborn

## Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/repo-name.git
   ```
##  Code Overview 
2. Downloading CSV File
The dataset is downloaded from a hosted server using the requests library.

## Reading the Data
3. The CSV file is read into a pandas DataFrame for further analysis.

## Selecting Features
4. Select relevant features for regression analysis.
   Selecting Features
   ```
      cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
      cdf.head(9)
   ```
## Plotting Emission Values
5. Plotting CO2 emissions against engine size to observe the linear relationship.
   ```
   plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS, color='blue')
   plt.xlabel("Engine size")
   plt.ylabel("Emission")
   plt.show()
   ```
## Training and Testing Data
6. Splitting the data into training and testing sets using an 80-20 split.
   ```
   msk = np.random.rand(len(df)) < 0.8
   train = cdf[msk]
   test = cdf[~msk]
   ```
## Multiple Linear Regression
7. Implementing multiple linear regression using sklearn's LinearRegression.
   ```
   from sklearn import linear_model
   regr = linear_model.LinearRegression()
   x = np.asanyarray(train[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
   y = np.asanyarray(train[['CO2EMISSIONS']])
   regr.fit(x, y)
   print('Coefficients: ', regr.coef_)
   ```
## Prediction
8. Making predictions on the test set and evaluating the model's performance.
   ```
   y_hat = regr.predict(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
   x = np.asanyarray(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
   y = np.asanyarray(test[['CO2EMISSIONS']])
   print("Mean Squared Error (MSE): %.2f" % np.mean((y_hat - y) ** 2))
   print('Variance score: %.2f' % regr.score(x, y))
   ```
## Practice
9. Additional practice with a different set of features
   ```
   regr = linear_model.LinearRegression()
   x = np.asanyarray(train[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY']])
   y = np.asanyarray(train[['CO2EMISSIONS']])
   regr.fit(x, y)
   print('Coefficients: ', regr.coef_)
   y_ = regr.predict(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY']])
   x = np.asanyarray(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY']])
   y = np.asanyarray(test[['CO2EMISSIONS']])
   print("Residual sum of squares: %.2f" % np.mean((y_ - y) ** 2))
   print('Variance score: %.2f' % regr.score(x, y))
   ```
## Results
Initial Model:
- Mean Squared Error (MSE): 461.06
- Variance score: 0.89
  
Practice Model:
- Residual sum of squares: 467.27
- Variance score: 0.89
  
These results indicate that the multiple linear regression model is a good fit for predicting CO2 emissions based on engine size and fuel consumption features.

