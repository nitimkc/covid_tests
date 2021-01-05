import pandas as pd 
import numpy as np
from pathlib import Path

PATH = Path(r'C:\Users\niti.mishra\Documents\2_TDMDAL\projects\covid_tests\covid_tests\data') 

data = pd.read_csv(Path.joinpath(PATH, 'IMOH Data with Prev.csv'))
data['Gender'] = data['Gender'].replace('NULL', np.nan) 
data = data[data['Ave_Pos_Past7d'].notna()]
X_test = data[data['Validation']=='Test']

X_test.to_csv(Path.joinpath(PATH, 'X_test.csv')) # replace file
data.to_csv(Path.joinpath(PATH, 'IMOH Data with Prev.csv')) # replace file
