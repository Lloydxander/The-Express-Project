import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import pickle

model = pickle.load(open('ExpressoModel.pkl', 'rb'))
st.markdown("<h1 style = 'text-align: center; color: 3D0C11'>EXPRESS PROJECT </h1> ", unsafe_allow_html = True)
st.markdown("<h6 style = 'top_margin: 0rem; text-align: center; color: #FFB4B4'>Built by Lloydxander</h6>", unsafe_allow_html = True)
st.image('pngwing.com (2).png', width = 400)


st.subheader('Project Brief')

st.markdown("<p style = 'top_margin: 0rem; text-align: justify; color: #FFB4B4'> In the dynamic and ever-evolving landscape of entrepreneurship, startups represent the vanguard of innovation and economic growth. The inception of a new venture is often accompanied by great enthusiasm and ambition, as entrepreneurs strive to transform their groundbreaking ideas into successful businesses. However, one of the central challenges faced by startups is the uncertainty surrounding their financial sustainability and profitability. This uncertainty is exacerbated by a myriad of factors,<br> ranging from market volatility and competition to operational costs and customer acquisition.</p>", unsafe_allow_html = True)

st.markdown("<br><br>", unsafe_allow_html = True)

username = st.text_input('Enter your name')
if st.button('submit name'):
    st.success(f"Welcome {username}. Pls use according to usage guidelines")

data = pd.read_csv('Expresso_churn_dataset.csv')
heat = plt.figure(figsize = (14, 7))
sel_col = ['TENURE','REGULARITY', 'CHURN']
data = data[sel_col]

from sklearn.preprocessing import LabelEncoder, StandardScaler
def transformer(dataframe):
    lb = LabelEncoder()
    scaler = StandardScaler()
    
    for i in dataframe.columns:  # --------------------------------------------- Iterate through the dataframe columns
        if i in dataframe.select_dtypes(include = ['object', 'category',]).columns: #-- Select all categorical columns
            dataframe[i] = lb.fit_transform(dataframe[i]) # -------------------- Label encode selected categorical columns
    return dataframe

transformer(data)

sns.heatmap(data.corr(), annot = True, cmap = 'BuPu')

st.write(heat)

st.write(data.sample(10))


st.sidebar.image('pngwing.com (4).png', caption= f'Welcome {username}')

input_type = st.sidebar.selectbox('Select Your preffered Input type', ['Slider Input', 'Number Input'])

if input_type == "Slider Input":
    tenure = st.sidebar.slider("TENURE", data['TENURE'].min(), data['TENURE'].max())
    regularity = st.sidebar.slider("REGULARITY", data['REGULARITY'].min(), data['REGULARITY'].max())
else:
    tenure = st.sidebar.number_input("TENURE", data['TENURE'].min(), data['TENURE'].max())
    regularity = st.sidebar.number_input("REGULARITY", data['REGULARITY'].min(), data['REGULARITY'].max())
    
input_variable = pd.DataFrame([{"TENURE":tenure, "REGULARITY": regularity}])
st.write(input_variable)

pred_result, interpret = st.tabs(["Prediction Tab", "Interpretation Tab"])
with pred_result:
    if st.button('PREDICT'):

        st.markdown("<br>", unsafe_allow_html= True)
        prediction = model.predict(input_variable)
        st.write("Predicted Profit is :", prediction)
    else:
        st.write('Pls press the predict button for prediction')

with interpret:
    st.subheader('Model Interpretation')
    #st.write(f"CHURN = {model.intercept_.round(2)} + {model.coef_[0].round(2)} TENURE + {model.coef_[1].round(2)} REGULARITY")

    #st.markdown("<br>", unsafe_allow_html= True)

    #st.markdown(f"- The expected Profit for a startup is {model.intercept_}")

    #st.markdown(f"- For every additional 1 dollar spent on R&D Spend, the expected profit is expected to increase by ${model.coef_[0].round(2)}  ")

    #st.markdown(f"- For every additional 1 dollar spent on Administration Expense, the expected profit is expected to decrease by ${model.coef_[1].round(2)}  ")
