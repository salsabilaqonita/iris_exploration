import streamlit as st
import pickle
import time
import pandas as pd

st.set_page_config(page_title='ML Portfolio', page_icon='📊', layout='wide')

st.write('Welcome to the ML Portfolio App!')
st.write('This dashboard create by: Salsabila Qonita Kaltsum')

select_var=st.sidebar.selectbox("Want to open about?",("Home","Iris Species"))

def iris():
    st.write("""
    This app predicts the **Iris Species**
    
    Data obtained from the [iris dataset](https://www.kaggle.com/uciml/iris) by UCIML. 
    """)

    st.sidebar.header('User Input Features:')

    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
    if uploaded_file is not None:
        input_df = pd.read_csv(uploaded_file)
    else:
        def user_input_features():
            st.sidebar.header('Input Manual')
            SepalLengthCm = st.sidebar.slider('Sepal Length (cm)', min_value=4.3, value=6.5, max_value=10.0)
            SepalWidthCm = st.sidebar.slider('Sepal Width (cm)', min_value=2.0, value=3.3, max_value=5.0)
            PetalLengthCm = st.sidebar.slider('Petal Length (cm)',min_value= 1.0, value=4.5, max_value=9.0)
            PetalWidthCm = st.sidebar.slider('Petal Width (cm)',min_value= 0.1, value=1.4, max_value=5.0)
            data = {'SepalLengthCm': SepalLengthCm,
                    'SepalWidthCm': SepalWidthCm,
                    'PetalLengthCm': PetalLengthCm,
                    'PetalWidthCm': PetalWidthCm}
            features = pd.DataFrame(data, index=[0]) #transform to pandas
            return features
        input_df = user_input_features()
    # # img = Image.open("iris.JPG")

    st.image("https://www.easytogrowbulbs.com/cdn/shop/products/BeardedIrisColorfullMix_VIS-sqWeb_8a293612-7bc0-4a9f-89ac-917e820d0ccb.jpg?v=1664472481&width=1920")

    button_var = st.sidebar.button('Predict!')

    if button_var:
        df = input_df
        st.write(df)
        
        with open("generate_iris.pkl", 'rb') as file:  
            loaded_model = pickle.load(file) # load previous trained model
        prediction = loaded_model.predict(df) # do prediction according to input value
        result = ['Iris-setosa' if prediction == 0 else ('Iris-versicolor' if prediction == 1 else 'Iris-virginica')]
        
        st.subheader('Prediction: ')
        output = str(result[0])
        with st.spinner('Wait for it...'):
            time.sleep(4)
            st.success(f"Prediction of this app is {output}")

if select_var == "Iris Species":
    iris()


