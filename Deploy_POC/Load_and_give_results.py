import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import numpy as np
import io
from typing import Union, Dict
from pathlib import Path

def load_and_transform_csv(file_path: Union[str, Path], original_sep: str = ',', new_sep: str = ';') -> pd.DataFrame:
    """
    Opens a CSV file, replaces column separators, and loads the content into a Pandas DataFrame.
    
    Parameters:
    - file_path (str or Path): The path to the CSV file to open.
    - original_sep (str): The original column separator to be replaced. Defaults to ','.
    - new_sep (str): The new column separator. Defaults to ';'.
    
    Returns:
    pd.DataFrame: A DataFrame containing the loaded data.
    
    Example:
    >>> df = load_and_transform_csv('my_file.csv')
    """
    
    with open(file_path, 'r', encoding='utf-8') as file:
        file_content = file.read()
    
    modified_content = file_content.replace(original_sep, new_sep)
    buffer = io.StringIO(modified_content)
    df = pd.read_csv(buffer, sep=new_sep)
    
    return df


def clean_text_column(df: pd.DataFrame, column_name: str, manual_mapping: Dict[str, str] , nan_placeholder: str = 'unknown') -> pd.DataFrame:
    """
    Clean a text column by converting to lowercase and handling NaN values. Mappe value if they correspond to the dict. 
    
    Parameters:
        df (pd.DataFrame): The DataFrame containing the column to clean.
        column_name (str): The name of the column to clean.
        nan_placeholder (str): Placeholder for NaN values. Default is 'unknown'.
        
    Returns:
        pd.Series: The cleaned column.
    """
    cleaned_column = df[column_name].str.lower()
    cleaned_column.fillna(nan_placeholder, inplace=True)
    cleaned_column = cleaned_column.replace(manual_mapping)
    return cleaned_column




def load_and_preprocess_data(Input_file : str) -> pd.DataFrame:
    """
    Load and preprocess the data using the functions defined above.
    
    Parameters:
        file_path (Union[str, Path]): The path to the CSV file to open.
        color_column (str): The name of the color column to reduce.
        text_column (str): The name of the text column to clean.
        manual_mapping (Dict[str, str]): A mapping to apply to the text column.
        general_colors (List[str]): List of colors to retain.
        
    Returns:
        pd.DataFrame: The preprocessed DataFrame.
    """

    df=load_and_transform_csv(Input_file)
    df['incorrect_fedas_code'] = df['incorrect_fedas_code'].astype(str)
    
    df['net_weight'].replace(0, np.nan, inplace=True)
    df['raw_weight'].replace(0, np.nan, inplace=True)
    
    df['average_weight'] = df[['net_weight', 'raw_weight']].mean(axis=1)
    # Define a list of general colors to retain
    general_colors = ["BLACK", "WHITE", "NOIR", "BLANC", "BLEU", "ROUGE", "GREEN", "YELLOW", "RED", "BLUE", "GREY", "GRAY", "BROWN", "PINK"]

    # Group specific shades and less frequent colors under 'Other' category
    df['color_label_reduced'] = df['color_label'].where(df['color_label'].str.upper().isin(general_colors), 'Other')
    
    
    
    columns_to_clean = [
    'brand', 
    'article_main_category', 
    'article_type',
    'article_detail',
    'country_of_origin', 'incorrect_fedas_code', 'color_label','accurate_gender'
    ]

    manual_mapping = {
    "homme": "man",
    "men": "man",
    "femme": "woman",
    "women": "woman",
    "loisirs": "leisure",
    "loisir": "leisure",
    "095a  black": "black",
    "bla black": "black",
    "0019 black": "black",
    # Add more mappings as identified from the dataset.
}

    # Clean the specified columns
    for column in columns_to_clean:
        df[f'{column}_reduced'] = clean_text_column(df, column, manual_mapping)  
    return df  



def make_prediction(df, outputPath : str):
    """
    Make a prediction using the model.
    
    Parameters:
    preprocessed_input (DataFrame): The preprocessed input data.
    model (Model): The model to be used for making the prediction.
    
    Returns:
    tuple: The prediction and the probabilities.
    """
    model = joblib.load('rf_model.pkl')

    features = [
        'brand_reduced', 
        'article_main_category_reduced', 
        'article_type_reduced',
        'article_detail_reduced',
        'country_of_origin_reduced',
        'color_label_reduced',
        'incorrect_fedas_code_reduced',
        'size', 'accurate_gender'
    ]

    # Encodage des étiquettes pour les variables catégorielles réduites
    label_encoders = {}
    for feature in features:
        le = LabelEncoder()
        df[feature] = le.fit_transform(df[feature].astype(str))
        label_encoders[feature] = le

    preprocessed_input=df[features]
    prediction = model.predict(preprocessed_input)
    probabilities = model.predict_proba(preprocessed_input)
    class_names = model.classes_
    predicted_probas_df = pd.DataFrame(probabilities, columns=class_names)
    actual_and_predicted = pd.DataFrame({
        'predicted': prediction
    }).reset_index()
    final_df = pd.concat([actual_and_predicted, predicted_probas_df], axis=1)
    final_df.to_csv(outputPath,index=False)
    
    return prediction, probabilities