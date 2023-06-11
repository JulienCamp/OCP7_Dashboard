import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import requests
import shap
from streamlit_shap import st_shap
import joblib
import plotly.subplots as sp
import plotly.graph_objects as go


st.set_option('deprecation.showPyplotGlobalUse', False)

MODELS_PATH = "./models/"
DATA_PATH = "./data/"

@st.cache_data
def load_data(max_rows) :
    train_data = joblib.load(DATA_PATH + "train_data.joblib")
    ref_data = train_data.iloc[:max_rows,:]
    
    test_data = joblib.load(DATA_PATH + "test_data.joblib")
    X_data = test_data.iloc[:max_rows,:]
    
    my_model = joblib.load(MODELS_PATH + "best_model.joblib")
    my_cal_model = joblib.load(MODELS_PATH + "cal_best_model.joblib")
       
    return X_data, my_model, my_cal_model, ref_data
    

def display_shap(model, client_data, X) : 
    explainer = shap.TreeExplainer(model['classifier'])
    
    # Get the feature names (lost during scaling)
    column_transformer = model.named_steps['transformers']
    feature_names = column_transformer.get_feature_names_out()
    feature_names = [name.replace("scaling__", "").replace("one_hot__", "").replace("passthrough_numeric__", "") for name in feature_names]
    
    # create preprocessed array from the line of dataframe which corresponds to the client
    X_t_cust = model['transformers'].transform(client_data)
    X_t = model['transformers'].transform(X)
    # create preprocessed dataframe from preprocessed array
    X_t_cust_df = pd.DataFrame(X_t_cust, columns=feature_names)
    X_t_df = pd.DataFrame(X_t, columns=feature_names)
    chosen_instance = X_t_cust_df
    shap_values = explainer.shap_values(chosen_instance)
    
    st.header("Global and local model interpretation")
    
    st.write("**Summary Plot**")
    expander = st.expander("Summary Plot")
    expander.write("This chart shows a summary of the average impact of features on the final prediction (here for the prediction of class 1). It's the global interpretation of the model since it computes feature importance on the whole dataset")
    st_shap(shap.summary_plot(explainer.shap_values(X_t_df)[1],  features=X_t_df, feature_names=feature_names, max_display=10))

    st.write('**Force charts**')
    expander = st.expander("About force chart")
    expander.write("Here we can see which features are the most predominant for prediction and how much force they apply on it. \
             Bigger is the bar corresponding to the feature, bigger is the force applied by this feature. If force is applied \
             to the left, it's a negative force (decrese the score for this prediction)\
             If applied to the right, it's a positive force (increase the score for this prediction)")
    
    st.write("Predictions for Class 0")
    st_shap(shap.force_plot(explainer.expected_value[0], shap_values[0], chosen_instance))
    st.write("Prediction for Class 1")
    st_shap(shap.force_plot(explainer.expected_value[1], shap_values[1], chosen_instance))
   
    return shap_values, feature_names

def display_hists(client_data, important_features,df,test_df,y) :
    
    expander = st.expander("About histograms")
    expander.write("The purpose to this histograms is to see the distributions of accepted clients on 6 most important features used to predict the client's class \
                   and show where the client is situated in theses distributions")
    index_accepted = y.loc[y == 0].index
    accepted_df = df.loc[index_accepted]
    
    fig = sp.make_subplots(rows=3, cols=2, subplot_titles=important_features)
    
    for i, feature in enumerate(important_features):
        row = i // 2 + 1
        col = i % 2 + 1
        client_value = client_data[feature].values
        
        fig.add_trace(
            go.Histogram(
                x=accepted_df[feature],
                name='Accepted clients',
                marker_color='rgba(0, 0.5, 0, 1)'
            ),
            row=row,
            col=col
        )
        
        fig.add_trace(
            go.Bar(
                x=[client_value[0]],
                y=(200,0),  #arbitrary offset to make the bar visible
                name='Client',
                marker_color='red'
            ),
            row=row,
            col=col
        )
        
    fig.update_layout(
        showlegend=False,
        height=800,
        width=1000,
        title_text="Histograms of Important Features"
    )
    
    st.plotly_chart(fig)

def transform_data(train_data, test_data, client_data, model) :
    column_transformer = model.named_steps['transformers']
    feature_names = column_transformer.get_feature_names_out()
    feature_names = [name.replace("scaling__", "").replace("one_hot__", "").replace("passthrough_numeric__", "") for name in feature_names]

    train_data_t = model['transformers'].transform(train_data)
    train_data_t = pd.DataFrame(train_data_t, columns=feature_names)

    test_data_t = model['transformers'].transform(test_data)
    test_data_t = pd.DataFrame(test_data_t, columns=feature_names)

    client_data_t = model['transformers'].transform(client_data)
    client_data_t = pd.DataFrame(test_data_t, columns=feature_names)

    return train_data_t, test_data_t, client_data_t
    
def get_prediction(client_id, selected_model, credit_amount) :
    json_data = {'client_id': client_id,
                 'model_type': selected_model,
                 'credit_amount': credit_amount}
    
    # Send a POST request to Flask API
    response = requests.post('https://ocp7flaskapi.herokuapp.com/api/prediction', json=json_data)
    
    # Check if the request was successful
    if response.status_code == 200:
        prediction_result = int(response.json()['prediction'])
        confidence_result = float(response.json()['confidence'])
        st.success(f'Prediction Result: {prediction_result}')
        if prediction_result == 0 :
            st.write(f":blue[Model predicted 0 for this client with a confidence of {round(confidence_result*100,2)}%]")
        else :
            st.write(f":red[[Model predicted 1 for this client with a confidence of {round(confidence_result*100,2)}%]")
    else:
        st.error('Prediction request failed with statut code :', response.status_code)

    return prediction_result, confidence_result 


def main() : 

    X, my_model, my_cal_model, ref_data= load_data(10000)
    st.title("Scoring application for OpenClassrooms's Data Science Project 7")
   

    selected_id = st.selectbox("Select a customer ID", X["SK_ID_CURR"])
    st.write("Customer selected:", selected_id)
    if selected_id is not None:
        model_choice = st.radio(
            "Which model do you want to use?",
            ('Non calibrated model', 'Calibrated model'))
        
       
        client_index = X.loc[X["SK_ID_CURR"] == selected_id].index
        client_data = X.loc[[client_index[0]]]
        st.write("Client's data : ")
        st.write(client_data)

        default_val = float(client_data['AMT_CREDIT'].values)
        credit_amount = st.number_input('Crédit amount:', value=default_val)
        client_data['AMT_CREDIT'] = credit_amount
        
        if model_choice == 'Non calibrated model':
            chosen_model = "default"
            model = my_model
        else:
            chosen_model = "calibrated"
            model = my_cal_model

        if st.button('Predict'):
            get_prediction(selected_id,chosen_model,credit_amount)

            #apply the model transformers on the data
            train_data_t, test_data_t, client_data_t = transform_data(ref_data, X, client_data, model)

            #Shap display
            if model_choice == 'Non calibrated model':
                shap_values, features = display_shap(model, X.loc[[client_index[0]]], X)
            else :
                shap_values, features = display_shap(model.calibrated_classifiers_[0].estimator, X.loc[[client_index[0]]], X)
            
            #Important features display
            vals= np.abs(shap_values).mean(0)
            feature_importance = pd.DataFrame(list(zip(features, sum(vals))), columns=['feature_name','feature_importance_vals'])
            feature_importance.sort_values(by=['feature_importance_vals'], ascending=False,inplace=True)
            important_features = feature_importance['feature_name'].head(6).to_list()
            st.write(feature_importance)

            #display histograms
            y = ref_data['TARGET']

            display_hists(client_data_t, important_features,train_data_t, test_data_t,y.reset_index(drop=True))

      


if __name__ == '__main__':
    main()