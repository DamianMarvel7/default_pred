from flask import Flask, request, render_template
import numpy as np
import pickle
import joblib
import pandas as pd

# Load the saved machine learning model
model = pickle.load(open('model.pkl', 'rb'))
encoder = joblib.load("encoder.joblib") 

app = Flask(__name__)
def predict_default(df_input):
    X_cat=pd.DataFrame(encoder.transform(df_input[["SEX","MARRIAGE","EDUCATION"]]).toarray(),columns=encoder.get_feature_names_out())
    df_input=pd.concat([X_cat,df_input.drop(["SEX","MARRIAGE","EDUCATION"],axis=1)],axis=1)
    features=['SEX_male', 'MARRIAGE_single', 'EDUCATION_others',
       'EDUCATION_university', 'LIMIT_BAL', 'AGE', 'PAY_0', 'BILL_MEAN',
       'PAY_AMT_MEAN']   
    prediction = model.predict_proba(df_input[features])
    pred = (prediction>=0.4).astype('int')[:,1]
    return pred


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract the input features from the form data
    LIMIT_BAL = request.form['LIMIT_BAL']
    SEX = request.form['SEX']
    EDUCATION = request.form['EDUCATION']
    MARRIAGE = request.form['MARRIAGE']
    AGE = request.form['AGE']
    PAY_0 = request.form['PAY_0']
    PAY_AMT_MEAN = request.form['PAY_AMT_MEAN']
    BILL_MEAN = request.form['BILL_MEAN']

    # Convert the input features to the correct data type and format
    X_input = [LIMIT_BAL, SEX, EDUCATION, MARRIAGE, AGE, PAY_0, PAY_AMT_MEAN, BILL_MEAN]

    input=[]
    c = 0
    for x in X_input:
        if (c==1 or c==2 or c==3):
            input.append(x)
        else:
            input.append(float(x))
        c=c+1
    input_features = ['LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE', 'PAY_0', 'PAY_AMT_MEAN', 'BILL_MEAN']
    df_input = pd.DataFrame([input],columns=input_features)
    # Use the model to make a prediction
    prediction = predict_default(df_input)
    if(prediction==1):
        prediction="default"
    else:
        prediction="not default"
    # Return the prediction to the user
    return render_template('index.html', prediction_text='Credit card will be: {}'.format(prediction))

if __name__ == "__main__":
    app.run(host='0.0.0.0',port=8080)

