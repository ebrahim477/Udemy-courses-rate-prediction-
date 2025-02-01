import streamlit as st
import pandas as pd
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

def load_resources():   
    le_title = joblib.load('../workspace/le_title.pkl')
    le_wishlist = joblib.load('../workspace/le_wishlist.pkl')
    scaler = joblib.load('../workspace/scaler.pkl')
    model = joblib.load('../workspace/rf_model.pkl')
    return le_title, le_wishlist, scaler, model


def main():
    st.title('Udemy Course Rating Prediction')
    le_title, le_wishlist, scaler, model = load_resources()
    input_data = {}
    available_titles = list(le_title.classes_)
    search_query = st.text_input('Search for a Course Title')
    filtered_titles = [title for title in available_titles if search_query.lower() in title.lower()]
    selected_title = st.selectbox(
        'Select Course Title',
        filtered_titles if search_query else available_titles,
    )
    
    if selected_title in available_titles:
        input_data['title'] = selected_title
    else:
        st.error("Invalid course title selected.")
        return

    input_data['num_subscribers'] = st.number_input('Number of Subscribers', min_value=0, value=100)
    input_data['num_reviews'] = st.number_input('Number of Reviews', min_value=0, value=10)
    input_data['num_published_practice_tests'] = st.number_input('Number of Practice Tests', min_value=0, value=10)
    input_data['price_detail__amount'] = st.number_input('Course Price', min_value=0.0, value=10.0, step=0.1)
    input_data['is_wishlisted'] = st.selectbox('Is Wishlisted', [False, True])

    input_df = pd.DataFrame([input_data])
    input_df['title'] = le_title.transform([input_df['title'][0]])
    input_df['is_wishlisted'] = int(input_df['is_wishlisted'][0])  
    
    numerical_cols = ['num_subscribers', 'num_reviews', 'price_detail__amount']
    input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])
    if st.button('Predict Rating'):
        input_df = input_df.reindex(columns=model.feature_names_in_, fill_value=0)
        prediction = model.predict(input_df)
        st.success(f'Predicted Course Rating: {prediction[0]:.2f}')


if __name__ == '__main__':
    main()