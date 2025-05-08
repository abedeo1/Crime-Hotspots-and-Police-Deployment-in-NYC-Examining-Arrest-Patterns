# Crime-Hotspots-and-Police-Deployment-in-NYC-Examining-Arrest-Patterns

## Overview
This data science project investigates arrest patterns in New York City using the NYPD Arrests dataset from 2024. By analyzing over 260,000 arrest records, we aim to identify crime hotspots, assess the influence of demographic factors on arrest trends, and provide data-driven insights that can help optimize police resource deployment. The findings aim to enhance public safety, help law enforcement allocate resources more effectively, and assist both residents and tourists in making informed decisions about safety across the city.


## Dataset 
Link to data: ![NYC NYPD Arrest Data (Year-to-Date)](https://catalog.data.gov/dataset/nypd-arrest-data-year-to-date)
(Picture gotten from geeksforgeeks)

Key Fields:
- Date and time of arrest
- Type of offense
- Arrest precinct
- Perpetrator’s gender, age, and race
- Geolocation (latitude/longitude)


## Key results

- Top 5 high-crime precincts identified with clear temporal patterns
- Clusters of crime found near Times Square, major subway hubs, and stadiums
- Race and gender disparities observed across offense categories


## Project Pipeline

This project follows the the data science workflow below:

![Data Pipeline](https://media.geeksforgeeks.org/wp-content/uploads/20201124094427/finalpipeline2.jpg)



### Phase 1: Problem Definition & EDA

- Identified key public safety issues in NYC
- Collected and cleaned structured NYPD arrest data
- Conducted exploratory data analysis (EDA) following John Tukey’s methodology to understand patterns in arrests


### Phase 2: Modeling & Visualization

- Applied classification, clustering, and statistical models to uncover behavioral patterns
- Compared multiple algorithms and visualized outcomes using Seaborn and Matplotlib


### Phase 3: Distributed Processing & Spark ML

- Scaled preprocessing using PySpark with RDD operations and data segmentation
- Built and evaluated machine learning models using Spark ML (DataFrame API), focusing on performance metrics like accuracy, F1 Score, and execution time
- Analyzed job performance with Spark DAG visualizations


### Phase 4: Data Product

- Built a Streamlit web app to explore arrest data in real-time, predict crime severity, perform analysis (e.g., crime distribution by borough and offense category), and access project documentation
- Enabled users to train machine learning models, evaluate performance, and visualize metrics like accuracy, precision, recall, and F1 score



