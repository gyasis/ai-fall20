import numpy as np
import pandas as pd


data = pd.read_csv('ai-fall20/datasets/predict-future-sales/sales_train.csv'
con
# Manipulate Train Data
from dateutil.parser import parse
data.date = data.date.astype('str')
data.date = data.date.apply(lambda x: parse(x))
data.date = data.date.apply(lambda x: pd.to_datetime(x,
    format='%Y-%m-%dT',
    errors='coerce'))
data['month'] = data.date.dt.month
data = data.groupby(['date_block_num','item_id','shop_id'])['item_cnt_day'].sum()
data = data.reset_index()

# Mainputlate Test Data
test = pd.read_csv('ai-fall20/datasets/predict-future-sales/test.csv')
test = test.assign(date_block_num = 34)
test  = test.assign(month = 11)

#WriteOut to csv
data.to_csv("Train_mod.csv", index=False)
test.to_csv("Test_mod.csv", index=False)
pr