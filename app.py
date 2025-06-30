import streamlit as st 
import numpy as np 
import tensorflow as tf 
from sklearn.preprocessing import StandardScaler , LabelEncoder , OneHotEncoder 
import pandas as pd 
import pickle 
from  tensorflow.keras.models import load_model 



# loading the trained model 

model = load_model('model.h5')




# loading the encoded and scaler 
with open('onehot_encoder_geo.pkl','rb') as file:
    label_encoder_geo = pickle.load(file) 
 

with open('label_gender_encoder.pkl','rb') as file:
    label_encoder_gender = pickle.load(file)


with open('scaler.pkl' , 'rb') as file :
    scaler = pickle.load(file)
    

# streamlit app 

st.title('Customer Churn Predication')


# user input 

geography = st.selectbox('Geograpy',label_encoder_geo.categories_[0])
gender = st.selectbox('Gender',label_encoder_gender.classes_)
age = st.slider('Age',18,92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimeted Salary')
tenure = st.slider('Tenure',0,10)
num_of_prod = st.slider('Number Of Products',1,4)
has_cred_card = st.selectbox('Has Credit Card' , [0,1])
is_active_number = st.selectbox('Is Actice Member',[0,1])


input_data = pd.DataFrame({
    'CreditScore' : [credit_score] , 
    'Gender' : [label_encoder_gender.transform([gender])[0]],
    'Age' : [age],
    'Tenure' : [tenure] , 
    'Balance' : [balance] , 
    'NumOfProducts' : [num_of_prod] , 
    'HasCrCard' : [has_cred_card] , 
    'IsActiveMember' : [is_active_number] , 
    'EstimatedSalary' : [estimated_salary] 
})


# one hot encode geography 

geo_encoded = label_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded,columns=label_encoder_geo.get_feature_names_out(['Geography']))

input_data = pd.concat([input_data.reset_index(drop=True),geo_encoded_df],axis=1)  



# scale the input  data 
input_data_scaled = scaler.transform(input_data)

# predication 
predication = model.predict(input_data_scaled)

predication_prob = predication[0][0]

st.write(predication_prob)
if predication_prob > 0.5 :
    st.write("This customer is likely to churn ")
else:
    st.write("This customer will not churn at any cost")

