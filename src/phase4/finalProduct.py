import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score
from datetime import datetime
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline



## Requirements.txt is basically a file that is used to get all the version of libraries used in the final phase.
## To run: streamlit run finalProduct.py


## loading our default Data 
data = pd.read_csv("../dataset/NYPD_Arrest_Data__Year_to_Date_cleaned.csv")

## setting the page config 
## Streamlit page configuration
st.set_page_config(page_title="NYPD Crime Prediction System",page_icon="🚨",layout="wide",initial_sidebar_state="auto")
st.title('🚨 NYPD Crime Prediction System')
st.markdown("""This system allows analyzing New York Police Department (NYPD) arrest data and predicting whether a crime is violent or a 'felony' offense based on various factors.""")
## Set menu, i divided it in 5 different tabs: 
## First tab: the user can see the dashboard, it's a summary of the data
## Second tab: the user can predict if a crime is violent or a felony
## Third tab: the user can train a model with the data
## Fourth tab: the user can see the performance of the models
firstTab, secondTab, thirdTab, fourthTab = st.tabs(['Dashboard', 'Crime Prediction', 'Model Training', 'Performance'])

with firstTab:
    if data is not None:
        st.header("Analisys of Arrest Data")
        st.subheader("Dataset After Clening")
        st.dataframe(data.head())
        st.subheader("Graph Analysis from EDA")
        col1, secondCol = st.columns(2)
        with col1:
            ## I got these two plots from our first phase, i only added some commands to make it more readable in the page indentation 
            if 'Arrest_Borough' in data.columns:
                st.subheader("Arrest by Borough")
                borough_counts = data['Arrest_Borough'].value_counts()
                fig, ax = plt.subplots(figsize=(10, 6))
                borough_counts.plot(kind='bar', ax=ax)
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
            else:
                st.warning("The column 'Arrest_Borough' is not present in the dataset.")
        
        with secondCol:
            if 'Offense_Category_Code' in data.columns:
                st.subheader("Arrest by Offense Category")
                offense_counts = data['Offense_Category_Code'].value_counts()
                fig, ax = plt.subplots(figsize=(10, 6))
                offense_counts.plot(kind='bar', ax=ax)
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
            else:
                st.warning("The column 'Offense_Category_Code' is not present in the dataset.")

## Literally copy and past from phase two, same functions 
def cleanAndPrepareFeatures(file):
    
    if isinstance(file, pd.DataFrame):
        arrest = file
    else:
        arrest = pd.read_csv(file)
    
    ## We have to transform some columns in int (basically a mapping process since the ML algorithm needs numerical values)
    arrest['Arrest_Date'] = pd.to_datetime(arrest['Arrest_Date'])
    arrest['Arrest_Month'] = arrest['Arrest_Date'].dt.month
    arrest['Arrest_Day'] = arrest['Arrest_Date'].dt.day
    arrest['Arrest_Year'] = arrest['Arrest_Date'].dt.year
    
    ## I created a new column that is 1 if the day is a weekend and 0 otherwise
    arrest['Is_Weekend'] = arrest['Arrest_Day_of_Week'].isin(['Saturday', 'Sunday']).astype(int)
    
    ## I selected some crimes that are basically considered as violent crimes (based on internet research)
    ## and i created a new column (for the same reason as above) that is 1 if the crime is a violent crime and 0 otherwise    
    violentCrimes = ['RAPE 1', 'RAPE 2', 'RAPE 3', 'SODOMY 1', 'STRANGULATION 1ST', 'SEXUAL ABUSE']
    arrest['Is_Violent_Crime'] = arrest['Offense_Description'].isin(violentCrimes).astype(int)
    
    ## Same stuff for the felony column
    arrest['Is_felony'] = (arrest['Offense_Category_Code'] == 'F').astype(int)
    
    ## Prepare the features for the model but we have some problems: 
    ## 1. We have to separate the categorical features and the numerical ones
    ##  (like the borough, etc...) because we need numbers 
    ## 2. I double checked the dataset and I saw that there are some missing values (remove them)
    
    ##1. 
    categoricalFeatures = ['Perpetrator_Race', 'Perpetrator_Sex', 'Perpetrator_Age_Group', 'Arrest_Borough', 'Arrest_Day_of_Week']
    numericalFeatures = ['Latitude', 'Longitude', 'Arrest_Month', 'Is_Weekend']
    totalFeatures = categoricalFeatures + numericalFeatures
    
    ##2.
    for i in totalFeatures:
        if i in arrest.columns:
            arrest = arrest.dropna(subset=[i])
            
    ## Now we have to get all the input features and the target variables
    ## X is the input features and y is the target (what we want to predict)
    X = arrest[totalFeatures]
    y = arrest['Is_Violent_Crime']
    y2 = arrest['Is_felony']
    
    ## I tried before with preprocessing.LabelEncoder() but it didn't work this because: 
    ## the LabelEncoder() creating some order relation between the values and so give me some errors when I tried to process data 
    ## I found OneHotEncoder(), creates new columns for each category and assigns 1 or 0 to the columns
    
    catEncoding = OneHotEncoder()
    numScaling = StandardScaler()
    
    ## This is essential, otherwise we have to write a lot more code
    preprocessor = ColumnTransformer(transformers=[
        ('cat', catEncoding, categoricalFeatures),
        ('num', numScaling, numericalFeatures)
    ])
    
    return X, y, y2, preprocessor, totalFeatures

def algorihtmsApplication(X, y, preprocessor, model_type):    
    ## Split the dataset in training and test set
    ## 0.3 means that we split the set 70% for training and 30% for testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    ## We have to create a pipeline to apply the model
    ## PIPELINE is basically a way in which we can preprocess the data and apply the model in one shot
    
    ## WE have to apply smote because the dataset is unbalanced and we have to balance it, 
    ## otherwise the model will give us a 0 precision and recall
    
    smote = SMOTE(random_state=43)
    
    if model_type == 'Logistic Regression':
        logisticRegressionPipe = ImbPipeline(steps=[
            ('preprocessor', preprocessor),
            ('sampling', smote),
            ('classifier', LogisticRegression(max_iter=1000, random_state=43, class_weight='balanced'))
        ])
        
        ## We have to train the model
        logisticRegressionPipe.fit(X_train, y_train)
        
        ## We have to predict the values
        y_pred = logisticRegressionPipe.predict(X_test)
        
        logisticRegAccuracy = accuracy_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)

        metrics = {
            'accuracy': logisticRegAccuracy,
            'recall': recall,
            'f1': f1,
            'precision': precision
        }
        
        return logisticRegressionPipe, metrics
        
    elif model_type == 'KNN':
        ## i set n_neighbors = 5 
        knnPipe = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', KNeighborsClassifier(n_neighbors=5))])
        
        knnPipe.fit(X_train, y_train)
        
        y_pred = knnPipe.predict(X_test)
        
        knnAccuracy = accuracy_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)

        metrics = {
            'accuracy': knnAccuracy,
            'recall': recall,
            'f1': f1,
            'precision': precision
        }
        
        return knnPipe, metrics
        
    elif model_type == 'Naive Bayes':
        naiveBayesPipe = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', GaussianNB())])
        naiveBayesPipe.fit(X_train, y_train)
        y_pred = naiveBayesPipe.predict(X_test)
        naiveBayesPipeAccuracy = accuracy_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)

        metrics = {
            'accuracy': naiveBayesPipeAccuracy,
            'recall': recall,
            'f1': f1,
            'precision': precision
        }
        
        return naiveBayesPipe, metrics
        
    elif model_type == 'SVM':
        ## using the same pipeline as before but with the SVC, same problem
        SVMPipe = ImbPipeline(steps=[
            ('preprocessor', preprocessor),
            ('sampling', smote),
            ('classifier', SVC(random_state=43, probability=True))
        ])
        SVMPipe.fit(X_train, y_train)
        y_pred = SVMPipe.predict(X_test)
        SVMAccuracy = accuracy_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)

        metrics = {
            'accuracy': SVMAccuracy,
            'recall': recall,
            'f1': f1,
            'precision': precision
        }
        
        return SVMPipe, metrics
        
    elif model_type == 'Random Forest':
        ## Random forest creates decision trees, each trained on a random subset of the data
        ## The result is the combination of the trees (each tree votes to classify data)
        randomForestPipe = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', RandomForestClassifier(random_state=43))])
        randomForestPipe.fit(X_train, y_train)
        y_pred = randomForestPipe.predict(X_test)
        randomForestAccuracy = accuracy_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)

        metrics = {
            'accuracy': randomForestAccuracy,
            'recall': recall,
            'f1': f1,
            'precision': precision
        }
        
        return randomForestPipe, metrics
        
    else:  # Decision Tree
        decisionTreePipe = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', DecisionTreeClassifier(random_state=43))])
        decisionTreePipe.fit(X_train, y_train)
        y_pred = decisionTreePipe.predict(X_test)
        decisionTreeAccuracy = accuracy_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)

        metrics = {
            'accuracy': decisionTreeAccuracy,
            'recall': recall,
            'f1': f1,
            'precision': precision
        }
        
        return decisionTreePipe, metrics

with secondTab:
    st.header("Crime Prediction")
    
    if data is not None:
        model_type = st.selectbox(
            'Select the model type',
            ['Logistic Regression', 'KNN', 'Naive Bayes', 'SVM', 'Random Forest', 'Decision Tree']
        )
        prediction_target = st.radio(
            "What do you want to predict?",
            ('Violent Crime', 'Felony')
        )
        
        # Parameters for prediction
        st.subheader("Choose your parameters")
        firstCol, secondCol, thirdCol = st.columns(3)
        with firstCol:
            borough = st.selectbox('Borough', ['MANHATTAN', 'BROOKLYN', 'BRONX', 'QUEENS', 'STATEN ISLAND'])
            age_group = st.selectbox('Age', ['<18', '18-24', '25-44', '45-64', '65+'])
        
        with secondCol:
            gender = st.selectbox('Sex', ['M', 'F'])
            race = st.selectbox('Race', ['BLACK', 'WHITE', 'HISPANIC', 'ASIAN', 'NATIVE AMERICAN'])
        
        with thirdCol:
            day_of_week = st.selectbox('Weekday', ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
            is_weekend = 1 if day_of_week in ['Saturday', 'Sunday'] else 0
            month = st.slider('Month', 1, 12, 6)
            
        ## Our latitude and longitude are set to the default values of New York City, this is the same discussed in the EDA phase 
        ## since the latitude and longitude can SLIGHTLY vary since even if we consider different boroughs, the values are not that different
        st.subheader("Geographic Coordinates")
        firstCol, secondCol = st.columns(2)
        with firstCol:
            latitude = st.number_input('Latitude', value=40.7128, format="%.4f")
        with secondCol:
            longitude = st.number_input('Longitude', value=-74.0060, format="%.4f")
            
        ## Prediction start 
        if st.button('Generate prediction'):
            ## Create input_data for prediction
            input_data = pd.DataFrame({
                'Perpetrator_Race': [race],
                'Perpetrator_Sex': [gender],
                'Perpetrator_Age_Group': [age_group],
                'Arrest_Borough': [borough],
                'Arrest_Day_of_Week': [day_of_week],
                'Latitude': [latitude],
                'Longitude': [longitude],
                'Arrest_Month': [month],
                'Is_Weekend': [is_weekend]
            })
            ## we save the model in a pickle file, so we can load it later
            ## the name of the file is based on the model type and the target
            model_file = f"{model_type.replace(' ', '-')}_{prediction_target.replace(' ', '-')}.pkl"
            
            ## we can do a check if our model exixts or not
            ## if it exists we load it, otherwise we train it (at the beginning we have to train it since no model exists)
            try:
                # Check if the model already exists
                with open(model_file, 'rb') as f:
                    model_exists = True
            except FileNotFoundError:
                model_exists = False
            if model_exists:
                with open(model_file, 'rb') as f:
                    model = pickle.load(f)
                probability = model.predict_proba(input_data)[0][1]
            else:
                st.info("Training model😎")
                try:
                    ## this is the same main of phase 2 file except for the save 
                    X, y_violent, y_felony, preprocessor, features = cleanAndPrepareFeatures(data)
                    target_y = y_violent if prediction_target == 'Violent Crime' else y_felony
                    model, metrics = algorihtmsApplication(X, target_y, preprocessor, model_type)
                    with open(model_file, 'wb') as f:
                        pickle.dump(model, f)
                    probability = model.predict_proba(input_data)[0][1]
                except Exception as e:
                    st.error(f"Error during model training: {e}")
                    st.stop()
            
            ## Get result 
            st.subheader("Prediction Result")
            ## I created a gauge chart to show the probability of the prediction
            ## I set the probability to 0.5 since it's the default value for the threshold
            #CSE 487 Data Intensive Computing 
            probability_percentage = probability * 100
            col1, secondCol = st.columns([1, 2])
            with col1:
                fig, ax = plt.subplots(figsize=(4, 4))
                ax.add_patch(plt.Circle((0.5, 0), 0.4, color='lightgrey', zorder=0))
                ax.add_patch(plt.Rectangle((0.1, 0), 0.8, -0.1, color='white', zorder=1))
                theta = np.linspace(0, np.pi, 100)
                x_low = 0.5 + 0.4 * np.cos(theta)
                y_low = 0 + 0.4 * np.sin(theta)
                ax.plot(x_low, y_low, color='green', linewidth=10, zorder=2)
                x_medium = 0.5 + 0.4 * np.cos(theta[33:67])
                y_medium = 0 + 0.4 * np.sin(theta[33:67])
                ax.plot(x_medium, y_medium, color='orange', linewidth=10, zorder=3)
                x_high = 0.5 + 0.4 * np.cos(theta[67:])
                y_high = 0 + 0.4 * np.sin(theta[67:])
                ax.plot(x_high, y_high, color='red', linewidth=10, zorder=4)
                needle_angle = np.pi * probability
                needle_x = 0.5 + 0.35 * np.cos(needle_angle)
                needle_y = 0 + 0.35 * np.sin(needle_angle)
                ax.plot([0.5, needle_x], [0, needle_y], color='black', linewidth=2, zorder=5)
                ax.add_patch(plt.Circle((0.5, 0), 0.05, color='black', zorder=6))
                ax.text(0.15, -0.15, "0%", fontsize=12)
                ax.text(0.5, 0.45, f"{probability_percentage:.1f}%", fontsize=14, ha='center')
                ax.text(0.85, -0.15, "100%", fontsize=12)
                ax.set_xlim(0, 1)
                ax.set_ylim(-0.5, 0.5)
                ax.axis('off')
                ax.set_aspect('equal')
                st.pyplot(fig)
            ## Explains the result
            with secondCol:
                prediction_text = "Violent Crime" if prediction_target == "Violent Crime" else "Felony"
                
                st.markdown(f"""
                ### Interpretation
                
                The probability that this crime is a **{prediction_text}** is **{probability_percentage:.1f}%**.
                
                - **Low probability** (0-33%): Unlikely to be a {prediction_text.lower()}
                - **Medium probability** (34-66%): Moderate chance of being a {prediction_text.lower()}
                - **High probability** (67-100%): High likelihood of being a {prediction_text.lower()}
                """)
                
                # Add an explanation of the factors influencing the prediction
                st.markdown("""
                ### Factors Influencing the Prediction
                The main factors considered in this prediction include:
                - Demographic characteristics (age, race, gender)
                - Geographic location (borough, coordinates)
                - Time (day of the week, month, weekend)
                """)
    
    else:
        st.warning("No data available for model training. Please upload a CSV file or verify that the default dataset is accessible.")

## This interface is used to train and display the performance of the models, basically 
## Department of Computer Science and Engineering
## we show the same results as before 
with thirdTab:
    st.header("Model Training")
    if data is not None:
        ## new model training 
        ## we must be able to select the model and the target to predict       
        st.subheader("Train a new model")
        col1, secondCol = st.columns(2)
        with col1:
            train_model_type = st.selectbox(
                'Select model type',
                ['Logistic Regression', 'KNN', 'Naive Bayes', 'SVM', 'Random Forest', 'Decision Tree'],
                key='train_model_type'
            )
        with secondCol:
            train_target = st.radio(
                "Select target to predict",
                ('Violent Crime', 'Felony'),
                key='train_target'
            )
        

        
        ##Start training 
        if st.button("Train Model"):
            ## Bar to show the progress of the training
            st.info("Training in progress👍")
            progress_bar = st.progress(0)

            try:
                X, y_violent, y_felony, preprocessor, features = cleanAndPrepareFeatures(data)
                target_y = y_violent if train_target == 'Violent Crime' else y_felony
                
                ## basically same code of before (phase 2)
                for i in range(0, 90):
                    progress_bar.progress(i + 1)
                model, metrics = algorihtmsApplication(X, target_y, preprocessor, train_model_type)
                progress_bar.progress(100)
                model_filename = f"{train_model_type.replace(' ', '_')}_{train_target.replace(' ', '_')}.pkl"
                with open(model_filename, 'wb') as f:
                    pickle.dump(model, f)
                st.success(f"Model successfully trained and saved as {model_filename}")
                
                ## Our Metrics
                st.subheader("Performance Metrics")
                col1, secondCol, thirdCol, col4 = st.columns(4)
                col1.metric("Accuracy", f"{metrics['accuracy']:.2f}")
                secondCol.metric("Precision", f"{metrics['precision']:.2f}")
                thirdCol.metric("Recall", f"{metrics['recall']:.2f}")
                col4.metric("F1 Score", f"{metrics['f1']:.2f}")
            
            except Exception as e:
                st.error(f"An error occurred during model training: {e}")
    else:
        st.warning("No data available for model training. Please upload a CSV file or verify that the default dataset is accessible.")


with fourthTab:
    st.header("Model Performance")
    if data is not None:
        ## I can select the model and the target to predict between my models 
        st.subheader("Compare different models")
        tagetToCompare = st.radio(
            "Select target to compare",
            ('Violent Crime', 'Felony'),
            key='compare_target'
        )
        modelsToCompare = st.multiselect(
            "Select models to compare",
            ['Logistic Regression', 'KNN', 'Naive Bayes', 'SVM', 'Random Forest', 'Decision Tree'],
            default=['Logistic Regression', 'Random Forest', 'Decision Tree']
        )
        
        if st.button("Compare Models"):
            if len(modelsToCompare) >0:
                ## the idea is to: get the data, clean it, prepare it and then train the models
                ## then we save the models in a pickle file
                ## then we load the models and compare them
                
                X, y_violent, y_felony, preprocessor, features = cleanAndPrepareFeatures(data)
                target_y = y_violent if tagetToCompare == 'Violent Crime' else y_felony
                comparison_results = []
                
                ## For each model, trained it using the algorihtmsApplication function (phase 2)
                for i, model_name in enumerate(modelsToCompare):
                    st.info(f"Training model: {model_name}...")
                    model, metrics = algorihtmsApplication(X, target_y, preprocessor, model_name)
                    if model is not None:
                        comparison_results.append({
                            'Model': model_name,
                            'Accuracy': metrics['accuracy'],
                            'Precision': metrics['precision'],
                            'Recall': metrics['recall'],
                            'F1 Score': metrics['f1']
                        })
                
                ## Table to show the results
                ## acudeobi
                if comparison_results:
                    results_df = pd.DataFrame(comparison_results)
                    st.dataframe(results_df)
                    fig, ax = plt.subplots(figsize=(10, 8))
                    bar_width = 0.2
                    x = np.arange(len(comparison_results))
                    ax.bar(x - bar_width*1.5, results_df['Accuracy'], bar_width, label='Accuracy')
                    ax.bar(x - bar_width/2, results_df['Precision'], bar_width, label='Precision')
                    ax.bar(x + bar_width/2, results_df['Recall'], bar_width, label='Recall')
                    ax.bar(x + bar_width*1.5, results_df['F1 Score'], bar_width, label='F1 Score')
                    ax.set_ylabel('Score')
                    ax.set_title(f'Model metrics comparison for {tagetToCompare}')
                    ax.set_xticks(x)
                    ax.set_xticklabels(results_df['Model'])
                    ax.legend()
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    ## Get the best model for each metric
                    st.subheader("Best models by metric")
                    best_accuracy = results_df.loc[results_df['Accuracy'].idxmax()]
                    best_precision = results_df.loc[results_df['Precision'].idxmax()]
                    best_recall = results_df.loc[results_df['Recall'].idxmax()]
                    best_f1 = results_df.loc[results_df['F1 Score'].idxmax()]
                    
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Best Accuracy", f"{best_accuracy['Accuracy']:.2f}", 
                                f"{best_accuracy['Model']}")
                    col2.metric("Best Precision", f"{best_precision['Precision']:.2f}", 
                                f"{best_precision['Model']}")
                    col3.metric("Best Recall", f"{best_recall['Recall']:.2f}", 
                                f"{best_recall['Model']}")
                    col4.metric("Best F1 Score", f"{best_f1['F1 Score']:.2f}", 
                                f"{best_f1['Model']}")