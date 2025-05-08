from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, lit, to_timestamp, concat, date_format

def renameColumsn(row):    
    return [row[i] for i in range(len(row))]

def LoadAndRenameColumn(spark, file_path):
    
    ## Load the CSV file
    arrest = spark.read.csv(file_path, header=True, inferSchema=True)
    
    ## Defining the column mapping (same as in the original cleaning code)
    column_mapping = {
        'ARREST_KEY': 'Arrest_ID',
        'ARREST_DATE': 'Arrest_Date',
        'PD_CD': 'Police_Department_Code',
        'PD_DESC': 'Offense_Description',
        'KY_CD': 'Offense_Key_Code',
        'OFNS_DESC': 'Offense_Detailed_Description',
        'LAW_CODE': 'Law_Code',
        'LAW_CAT_CD': 'Offense_Category_Code',
        'ARREST_BORO': 'Arrest_Borough',
        'ARREST_PRECINCT': 'Arrest_Precinct',
        'JURISDICTION_CODE': 'Jurisdiction_Code',
        'AGE_GROUP': 'Perpetrator_Age_Group',
        'PERP_SEX': 'Perpetrator_Sex',
        'PERP_RACE': 'Perpetrator_Race',
        'X_COORD_CD': 'X_Coordinate',
        'Y_COORD_CD': 'Y_Coordinate',
        'Latitude': 'Latitude',
        'Longitude': 'Longitude',
        'New Georeferenced Column': 'Georeferenced_Location'
    }
    
    ## converting  dataframe to RDD to compute it in a distributed way
    rdd = arrest.rdd
    
    ## Get the header (column names)
    columnName = arrest.columns
    
    ## Create a mapping from original column index to new column name
    ## this mapping is to rename the columns in the RDD.
    ## Creating a dictionary where the key is the index of the column and the value is the new name of the column.
    ## using this dictionary to rename the columns in the RDD.
    newIndexName = {}   
    for old_name, new_name in column_mapping.items():
        if old_name in columnName:
            index = columnName.index(old_name)
            newIndexName[index] = new_name
    
    # applying the renameColumsn function to each row of the RDD thanks to the map function
    newRDDWithColumns = rdd.map(renameColumsn)
    newColumnName = [column_mapping.get(name, name) for name in columnName]
    
    ## https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/api/pyspark.sql.SparkSession.createDataFrame.html
    renamedArrest = spark.createDataFrame(newRDDWithColumns, newColumnName)
    # Drop the columns that were dropped in the original cleaning code
    renamedArrest = renamedArrest.drop("Arrest_ID", "Georeferenced_Location")
    
    
    return renamedArrest


def ConvertDateFormat(arrest_nypd):
    
    
    convertDateToTs = arrest_nypd.withColumn("Arrest_Date", to_timestamp(col("Arrest_Date"), "MM/dd/yyyy"))
    
    # adding the day of the week
    dataOfTheWeek = convertDateToTs.withColumn("Arrest_Day_of_Week", date_format(col("Arrest_Date"), "EEEE"))
    return dataOfTheWeek


def MissingValueHandling(arrest_nypd):
    # Same as before, convert the dataframe to RDD and get the columns
    rdd = arrest_nypd.rdd
    columns = arrest_nypd.columns
    indexOfOffenseCode = columns.index("Offense_Category_Code")
    
    # Same procdure of original cleaning code, checking for null values and removing them (rows with null values)
    filtered_rdd = rdd.filter(lambda row: row[indexOfOffenseCode] is not None and row[indexOfOffenseCode] != "(null)")
    
    # Create a new DataFrame from the filtering RDD
    cleaned_arrest_nypd = arrest_nypd.sparkSession.createDataFrame(filtered_rdd, arrest_nypd.schema)
    
    return cleaned_arrest_nypd


def LocationFeatures(arrest_nypd):
    ## I found on https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/api/pyspark.sql.functions.concat.html that its possible to directly 
    ## concatenate columns using the concat function. 
    location_df = arrest_nypd.withColumn("Location", concat(col("Latitude"), lit(", "), col("Longitude")))
    # Drop them as in the original cleaning code
    location_df = location_df.drop("X_Coordinate", "Y_Coordinate")
    
    
    return location_df

def ConsolidateRace(arrest_nypd):
    
    
    ## convert in RDD, get the columns and the index of the column that I want to change and map the values.
    rdd = arrest_nypd.rdd
    
    # Get column indices
    columns = arrest_nypd.columns
    race_idx = columns.index("Perpetrator_Race")
    
    # Define race mapping
    raceMapping = {'BLACK HISPANIC': 'HISPANIC','WHITE HISPANIC': 'HISPANIC','ASIAN / PACIFIC ISLANDER': 'ASIAN','AMERICAN INDIAN/ALASKAN NATIVE': 'NATIVE AMERICAN'}
    

    ## get the index of the column to change and map the values
    rddMap = rdd.map(lambda row: [raceMapping.get(row[race_idx], row[race_idx]) if i == race_idx else row[i] for i in range(len(row))])
    
    ## Create DataFrame as before and filter the rows with UNKNOWN values
    cleanedData = arrest_nypd.sparkSession.createDataFrame(rddMap, arrest_nypd.schema)
    cleanedData = cleanedData.filter(col("Perpetrator_Race") != "UNKNOWN")
    
    return cleanedData

def ConsolidateOffense(arrest_nypd):
    
    ## on the documentation of Spark I found that I can use the when/otherwise
    ## functions to apply a mapping logic. The idea is to map the values of the Offense_Detailed_Description column to a new value.
    ## If the value is not in the mapping, assign the value "OTHER".
    offenseMap = arrest_nypd.withColumn(
        "Offense_Detailed_Description",
        when(col("Offense_Detailed_Description") == "PETIT LARCENY", "THEFT")
        .when(col("Offense_Detailed_Description") == "GRAND LARCENY", "THEFT")
        .when(col("Offense_Detailed_Description") == "ASSAULT 3 & RELATED OFFENSES", "ASSAULT")
        .when(col("Offense_Detailed_Description") == "FELONY ASSAULT", "ASSAULT")
        .when(col("Offense_Detailed_Description") == "DANGEROUS DRUGS", "DRUGS")
        .when(col("Offense_Detailed_Description") == "MARIJUANA, POSSESSION 4 & 5", "DRUGS")
        .otherwise(when(col("Offense_Detailed_Description").isin(
            ["THEFT", "ASSAULT", "DRUGS"]), col("Offense_Detailed_Description"))
                 .otherwise("OTHER"))
    )
    
    return offenseMap


def BoroughExpansion(arrest_nypd):
    ## mapping the values of the borough column
    rdd = arrest_nypd.rdd
    columns = arrest_nypd.columns
    borIndex = columns.index("Arrest_Borough")
    
    # Define borough mapping
    borMap = {
        'M': 'MANHATTAN',
        'B': 'BRONX',
        'K': 'BROOKLYN',
        'Q': 'QUEENS',
        'S': 'STATEN ISLAND'
    }
    
    mappedBor = rdd.map(lambda row: [borMap.get(row[borIndex], row[borIndex]) if i == borIndex else row[i] for i in range(len(row))])
    
    borData = arrest_nypd.sparkSession.createDataFrame(mappedBor, arrest_nypd.schema)
    
    
    return borData

    
if __name__ == "__main__":
    
    ## Initialize Spark session
    ## applying the same operations as in the original cleaning code, but using pyPark. 
    
    spark = SparkSession.builder.appName("NYPD Arrest").config("spark.executor.memory", "4g").config("spark.driver.memory", "4g").getOrCreate()
    arrest_nypd = "../dataset/NYPD_Arrest_Data__Year_to_Date_.csv"

    # Load and rename columns
    #  Convert date format
    # Handle missing values
    # location features
    #  Consolidate race categories
    #  Consolidate offense categories
    #  Expand borough names
    
    arrest_nypd = LoadAndRenameColumn(spark, arrest_nypd)
    arrest_nypd = ConvertDateFormat(arrest_nypd)
    arrest_nypd = MissingValueHandling(arrest_nypd)
    arrest_nypd = LocationFeatures(arrest_nypd)
    arrest_nypd = ConsolidateRace(arrest_nypd)
    arrest_nypd = ConsolidateOffense(arrest_nypd)
    arrest_nypd = BoroughExpansion(arrest_nypd)
    arrest_nypd.write.csv("../dataset/NYPD_Arrest_Data_Processed_Spark.csv", header=True, mode="overwrite")
    
    print("START!")
    
    # Stop Spark session
    spark.stop()
