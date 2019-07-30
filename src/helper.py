import requests
from time import sleep
import datetime
import pandas as pd   
from pandas.io.json import json_normalize  
from pandas.api.types import CategoricalDtype 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import warnings

warnings.simplefilter('ignore')

def get_api_info_division(URL):
    """ 
    API information conversion to Pandas DataFrame.
        - Similar Function to 'get_api_info', but for that requires division information.
        
        1.Receives information from API using requests.
        2.Take entries column from response.json
        3.Normalize the list of dictionaries
        4.Add column total_games and winrate

    Parameters: 
    URL (str): target url for the request (APIKEY included). 
    
    
    Returns: 
    Pandas Dataframe with total_games and winrate column
  
    """
    sleep(1)  # sleep functions is used to prevent exceeding request limit. 
    
    response=requests.get(URL)  # requesting information using API KEY.
    df=json_normalize(response.json()) #Received data format is .json, so it requires the normalization
    
    return df