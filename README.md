# Australia-Rain-Prediction
This project predicts whether it will rain tomorrow in Australia using machine learning models.
The dataset is taken from [Kaggle :Rain in Australia](https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package/data)
This project applies **machine learning** models to forecast whether it will rain the next day **("RainTomorrow")** based on historical weather data

##Dataset
**Rows**: ~145,000 observations  
**Features**: 23 weather attributes 
**Target**: 'RainTomorrow' (Yes/No)

##Methodology
1. **Data Preprocessing**
   - Handled missing values using **Iterative Imputer.**
   - Encoded categorical variables **(One-Hot)**  
   - Standardized numerical features with **StandardScaler**

2. **Class Balancing**
   - Addressed imbalance using **KMeans clustering** on the majority class('No')
   - Different values of 'k' (2, 3, 4, 5, 7, 9, 100, 200, 250).  
   - From each cluster, representative samples were selected to balance the dataset.
     
3. **Model Training**
   - Trained and compared multiple machine learning models:  
     - Logistic Regression 
     - Random Forest 
     - Decision Tree   
     - Gradient Boosting  
     - Bagging with Decision Tree  
     - XGBoost  
     - LightGBM  
     - CatBoost  

4. **Model Evaluation**
   - Plotted **ROC-AUC curves** for all models to compare classification performance.  
   - Evaluated each model using:  
     - Accuracy  
     - Precision  
     - Recall  
     - F1-Score  
     - Support (class distribution)  


