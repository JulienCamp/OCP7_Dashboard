import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import requests
import shap
from streamlit_shap import st_shap
import joblib


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
    

def display_shap(model, client_data) : 
    explainer = shap.TreeExplainer(model['classifier'])
    

    # Get the feature names (lost during scaling)
    column_transformer = model.named_steps['transformers']
    feature_names = column_transformer.get_feature_names_out()
    
    # create preprocessed array from the line of dataframe which corresponds to the client
    X_t = model['transformers'].transform(client_data)

    # create preprocessed dataframe from preprocessed array
    X_t_df = pd.DataFrame(X_t, columns=feature_names)
    chosen_instance = X_t_df
    shap_values = explainer.shap_values(chosen_instance)
    st.write("Diagramme des forces de la classe 0")
    st_shap(shap.force_plot(explainer.expected_value[0], shap_values[0], chosen_instance))
    st.write("Diagramme des forces de la classe 1")
    st_shap(shap.force_plot(explainer.expected_value[1], shap_values[1], chosen_instance))
    st_shap(shap.summary_plot(shap_values, X_t_df, max_display=10))
    return shap_values

def display_hists(client_index, important_features,df,y) :
    # Function that plot hist of distribution of each important features for clients of class 0
    # and color in red the bar to which belongs the client
    index_accepted = y.loc[y==0].index
    accepted_df = df.loc[index_accepted]
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(13, 15))
    for i, feature in enumerate(important_features):
        row = i // 2
        col = i % 2
        client_value = df.loc[client_index, feature].values
        ax = sns.histplot(accepted_df, x=feature, ax=axes[row, col])
        xtlabels = ax.get_xticklabels()
        #st.write("client value:", client_value[0])
        if isinstance(client_value[0], str) :
            ax.tick_params(labelrotation=45)
            pos=find_index_by_string(xtlabels, client_value[0])
            ax.patches[pos].set_facecolor('red')
        else : 
            for patch in ax.patches:
                if patch.get_x() <= float(client_value[0]) <= patch.get_x() + patch.get_width():
                    patch.set_facecolor('red')
    
        axes[row, col].set_title(feature)
    plt.tight_layout()
    plt.show()
    st.pyplot()
     


def main() : 

    X, my_model, my_cal_model, ref_data= load_data(10000)
    selected_id = st.selectbox("Select an ID", X["SK_ID_CURR"])

    st.write("Id client sélectionné:", selected_id)
    if selected_id is not None:
        model_choice = st.radio(
            "Quel modèle souhaitez vous utiliser",
            ('Modèle non calibré', 'Modèle calibré'))
        
        default_val = float(to_predict['AMT_CREDIT'].values)
        credit_amount = st.number_input('Montant du prêt', value=default_val)
        to_predict['AMT_CREDIT'] = credit_amount

        client_index = X.loc[X["SK_ID_CURR"] == selected_id].index
        to_predict = X.loc[[client_index[0]]]
        st.write("Informations du client : ")
        st.write(to_predict)
        
        if model_choice == 'Modèle non calibré':
            chosen_model = "default"
            model = my_model
        else:
            chosen_model = "calibrated"
            model = my_cal_model

        json_data = {'client_id': selected_id,
                    'model_type': chosen_model,
                    'credit_amount': credit_amount }
        
        if st.button('Predict'):
            # Send a POST request to Flask API
            response = requests.post('https://ocp7flaskapi.herokuapp.com/api/prediction', json=json_data)
            # Check if the request was successful
            st.write(response.status_code)
            if response.status_code == 200:
                st.write("request successfull")
                prediction_result = int(response.json()['prediction'])
                confidence_result = float(response.json()['confidence'])
        #         # shap_values_list = response.json()['shap_values']
        #         # shap_values = [np.array(arr) for arr in shap_values_list]
        #         # display_shap(shap_values)
                st.success(f'Prediction Result: {prediction_result}')
                if prediction_result == 0 :
                    st.write(f":blue[Le modèle a prédit 0 pour ce client avec une confiance de {round(confidence_result*100,2)}%]")
                else :
                    st.write(f":red[Le modèle a prédit 1 pour ce client avec une confiance de {round(confidence_result*100,2)}%]")
            else:
                st.error('Prediction request failed')    
               
                
        #         # Predictions
        #     # if st.button("Display Shap") :
        #     features = X.columns

        #     #Shap display
        #     if model_choice == 'Modèle non calibré':
        #         shap_values = display_shap(model, X.loc[[client_index[0]]])
        #     else :
        #         shap_values = display_shap(model.calibrated_classifiers_[0].estimator, X.loc[[client_index[0]]])
            
        #     #Important features display
        #     vals= np.abs(shap_values).mean(0)
        #     feature_importance = pd.DataFrame(list(zip(features, sum(vals))), columns=['feature_name','feature_importance_vals'])
        #     feature_importance.sort_values(by=['feature_importance_vals'], ascending=False,inplace=True)
        #     st.write(feature_importance.head(10))
        #     important_features = feature_importance['feature_name'].head(6).to_list()
        #     #st.write(important_features)
        #     df = full_data[important_features]

    #display_hists(client_index, important_features,df,y)

if __name__ == '__main__':
    main()