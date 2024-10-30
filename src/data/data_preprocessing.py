# importing necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import zscore
import yaml
import os



def load_config(config_path):
    """
    Load the configuration file.
    :param config_path: str - The path to the configuration file.
    :return: dict - The loaded configuration.
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def load_data(file_path, encoding = 'ISO-8859-1'):
    """
    Load a csv file with a specified encoding.
    :param file_path: str - The path to the CSV file.
    :param encoding: str - The encoding of the CSV file.
    :return: pd.DataFrame - The loaded DataFrame.
    """
    return pd.read_csv(file_path, encoding = encoding)


# function to encode customer sgments using a mapping technique
def encode_customer_segment(data):
    # defining the mapping
    segment_mapping = {
        'Consumer' : 1,
        'Corporate' : 2,
        'Home Office' : 3
    }
    # applying the mapping to the customer segment column
    data['Customer Segment'] = data['Customer Segment'].map(segment_mapping)
    
    return data

# function to clean data
def clean_data(data):
    # dropping columns with excessive missing values
    data = data.drop(columns = ['Customer Lname', 'Order Zipcode', 'Product Description'])

    # filling missing values for customer zipcode with mean
    data['Customer Zipcode'] = data['Customer Zipcode'].fillna(data['Customer Zipcode'].mean())

    # encoding customer segment
    data = encode_customer_segment(data)

    return data


# Function to derive date-related features
def derive_date_features(data):
    # Ensure the Order Date is in datetime format
    data['order date (DateOrders)'] = pd.to_datetime(data['order date (DateOrders)'], errors='coerce')
    
    # Extracting date features
    data['Order Day'] = data['order date (DateOrders)'].dt.day
    data['Order Month'] = data['order date (DateOrders)'].dt.month
    data['Order Year'] = data['order date (DateOrders)'].dt.year
    data['Order Day of Week'] = data['order date (DateOrders)'].dt.dayofweek
    data['Order Week of Year'] = data['order date (DateOrders)'].dt.isocalendar().week
    data['Is Weekend'] = data['Order Day of Week'].apply(lambda x: 1 if x >= 5 else 0)
    
    return data

# Function for feature engineering
def feature_engineering(data):
    # Selecting relevant features for demand forecasting
    selected_features = [
        'Order Item Discount', 
        'Order Item Discount Rate', 
        'Order Item Product Price', 
        'Order Item Total', 
        'Product Price', 
        'Order Item Quantity',
        'Order Day', 
        'Order Month', 
        'Order Year', 
        'Order Day of Week', 
        'Order Week of Year', 
        'Is Weekend',
        'Customer Segment',  # Now encoded as 1, 2, 3
        'Category Id'        # Used as is
    ]
    
    # Create a new dataframe with selected features
    data = data[selected_features]
    
    return data

# Function to remove outliers using z-score for specific features
def remove_outliers_zscore(data, features, threshold=3):
    # Calculate z-scores for specified features
    z_scores = data[features].apply(zscore)
    
    # Filter out data points with z-score above the threshold
    data = data[(np.abs(z_scores) < threshold).all(axis=1)]
    
    return data

# Function to remove duplicate rows
def remove_duplicates(data):
    return data.drop_duplicates()

# Main function to run all preprocessing steps
def main():
    #Load configuration
    config = load_config('config\config.yaml')

    # Load the datasets
    supply_chain_data_path = os.path.join(config['data']['raw_path'],'DataCoSupplyChainDataset.csv' )

    data = load_data(supply_chain_data_path)
    
    # Clean the data
    data = clean_data(data)
    
    # Derive date-related features
    data = derive_date_features(data)
    
    # Perform feature engineering
    data = feature_engineering(data)

     # Remove duplicate rows
    data = remove_duplicates(data)
    
    # Specify the features for outlier removal
    outlier_features = ['Order Item Discount', 'Order Item Product Price', 'Order Item Total', 'Product Price']
    
    # Remove outliers for specified features using z-score
    processed_data = remove_outliers_zscore(data, outlier_features)
    
    # Save the cleaned and processed data
    processed_data.to_csv('data/processed/processed_data.csv', index=False)
    
    # Display the final cleaned data
    print(processed_data.head())

if __name__ == "__main__":
    main()

