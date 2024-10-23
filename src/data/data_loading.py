import pandas as pd
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
    Load a csv file with a secified encoding.
    :param file_path: str - The path to the CSV file.
    :param encoding: str - The encoding of the CSV file.
    :return: pd.DataFrame - The loaded DataFrame.
    """
    return pd.read_csv(file_path, encoding = encoding)


def inspect_data(data):
    """
    Inspect the first few rows, summary statistics, and missing values of a DataFrame.
    :param data: pd.DataFrame - The dataframe to inspect.
    :return: dict - A dictionary containing inspection results.
    """
    inspection = {
        "first_few_rows": data.head(),
        "summary_statistics": data.describe(),
        "missing_values": data.isnull().sum()

    }

    return inspection

if __name__ == "__main__":
    #Load configuration
    config = load_config('config\config.yaml')

    # Load the datasets
    supply_chain_data_path = os.path.join(config['data']['raw_path'],'DataCoSupplyChainDataset.csv' )
    description_data_path = os.path.join(config['data']['raw_path'], 'DescriptionDataCoSupplyChain.csv')


    supply_chain_data = load_data(supply_chain_data_path)
    description_data = load_data(description_data_path)

    # inspect the datasets

    print("Supply Chain Data:")
    supply_chain_inspection = inspect_data(supply_chain_data)
    for key, value in supply_chain_inspection.items():
        print(f"\n{key}:\n{value}")


    print("\nDescription Data:")
    description_inspection = inspect_data(description_data)
    for key,value in description_inspection.items():
        print(f"\n{key}:\n{value}")