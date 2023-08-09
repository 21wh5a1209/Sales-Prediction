import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_scorep

# Data preprocessing
def preprocess(df):
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df.drop(columns=['date'], inplace=True)
    country_mapping={'Finland':0,'Norway':1,'Sweden':2}
    df['country'] = df['country'].map(country_mapping)
    store_mapping={'KaggleMart':0,'KaggleRama':1}
    df['store'] = df['store'].map(store_mapping)
    product_mapping={'Kaggle Mug':0,'Kaggle Hat':1,'Kaggle Sticker':2}
    df['product'] = df['product'].map(product_mapping) 
        
# Read in the training and testing data
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
sample_submission = pd.read_csv('sample_submission.csv')

# Data preprocessing for train data and test data
preprocess(train_data)
preprocess(test_data)

# Split the training data into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(train_data.drop(columns=['num_sold','row_id']), train_data['num_sold'], test_size=0.2, random_state=42)

# Train a random forest regressor
rf = RandomForestRegressor(bootstrap=True,n_estimators=1000, random_state=42)
rf.fit(X_train, y_train)

# Make predictions on the validation set and calculate the ROC AUC score
y_val_pred = rf.predict(X_val)
print("done")    

def sales(date,country,store,product):
    df=pd.DataFrame.from_dict({'date':[date], 'country':[country], 'store':[store], 'product':[product]})
    print(df)
    preprocess(df)
    print(df)
    result=rf.predict(df)
    return result


from tkinter import *
from tkcalendar import DateEntry
from ttkbootstrap.constants import *
import ttkbootstrap as tb
from datetime import date
from ttkbootstrap.dialogs import Querybox


root = tb.Window(themename="superhero")
root.title("Sales Prediction")
root.geometry('600x500')
 
date= tb.Frame(root)
date.pack(pady=15,padx=10,fill="x")
tb.Label(date,text="Date",font=("Arial",20,"bold"),bootstyle="warning").pack(side="left",padx=15)
de=tb.DateEntry(date,dateformat='%Y-%m-%d')
de.pack(side="left",padx=55)


var_country = StringVar()
country=tb.Frame(root)
country.pack(pady=15,padx=10,fill="x")
tb.Label(country,text="Country",font=("Arial",20,"bold"),bootstyle="warning").pack(side="left",padx=15)
tb.Radiobutton(country,text="Finland",bootstyle="danger",variable=var_country,value="Finland").pack(side="left",padx=10,anchor="w")
tb.Radiobutton(country,text="Norway",bootstyle="danger",variable=var_country,value="Norway").pack(side="left",padx=10,anchor="w")
tb.Radiobutton(country,text="Sweden",bootstyle="danger",variable=var_country,value="Sweden").pack(side="left",padx=10,anchor="w")

var_store = StringVar()
store=tb.Frame(root)
store.pack(pady=15,padx=5,fill="x")
tb.Label(store,text="Store",font=("Arial",20,"bold"),bootstyle="warning").pack(side="left",padx=15)
tb.Radiobutton(store,text="KaggleMart",bootstyle="danger",variable=var_store,value="KaggleMart").pack(side="left",padx=42,anchor="w")
tb.Radiobutton(store,text="KaggleRama",bootstyle="danger",variable=var_store,value="KaggleRama").pack(side="left",padx=5,anchor="w")


var_product = StringVar()
product=tb.Frame(root)
product.pack(pady=15,padx=10,fill="x")
tb.Label(product,text="Product",font=("Arial",20,"bold"),bootstyle="warning").pack(side="left",padx=15)
tb.Radiobutton(product,text="Kaggle Mug",bootstyle="danger",variable=var_product,value="Kaggle Mug").pack(side="left",padx=10,anchor="w")
tb.Radiobutton(product,text="Kaggle Hat",bootstyle="danger",variable=var_product,value="Kaggle Hat").pack(side="left",padx=10,anchor="w")
tb.Radiobutton(product,text="Kaggle Sticker",bootstyle="danger",variable=var_product,value="Kaggle Sticker").pack(side="left",padx=10,anchor="w")

# submit_button = tb.Button(text="Submit")  

# Define button click function
def button_click():
    selected_date = de.entry.get()
    country = var_country.get()
    print(selected_date)
    product = var_product.get()
    store = var_store.get()
    output_value = sales(selected_date,country,store,product)
    result_window = tb.Toplevel(root)
    result_window.title("Result")
    result_window.geometry('200x100')
    result_label1 = tb.Label(result_window, text="The Sales Are :")
    result_label1.pack()
    result_label = tb.Label(result_window, text=output_value)
    result_label.pack()

# Bind button click event to button
submit = tb.Frame(root)
submit.pack(pady=50,padx=10,fill="x")
submit_button=tb.Button(submit,text="submit",bootstyle="SUCCESS",command=button_click).pack(padx=15)
 

     
root.mainloop()