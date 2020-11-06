from flask import Flask, render_template, request
import pickle
import numpy as np
from pathlib import Path

RESULTS = Path(r'C:\Users\niti.mishra\Documents\2_TDMDAL\covid_tests\covid_tests\results')
# load the model
with open(Path.joinpath(RESULTS, "best_model.pkl"), 'rb') as f:
    model = pickle.load(f) 

# instantiate Flask
app = Flask('covid_predictor', template_folder='templates')

# use Python decorators to decorate a function to map URL to a function
# "/" URL is now mapped to "show_predict_covid_form" function
@app.route('/') 

# if user visits "/", then Flask will render "predictorform.html" on the web browser
#                          Flask will look for the html file in templates folder
# "render_tempplate" function renders the template and expects it to be stored 
# in the Templates folder on the same level as the "app.py" file
def show_predict_covid_form():
    return render_template('main.html')

# "/results" is now mapped to "results" function defined below (line 18 to 30)    
@app.route('/results', methods=['GET', 'POST'])

def results():
    form = request.form
    if request.method == 'GET':
        # return(render_template('predictorform.html'))
        show_predict_covid_form()

    if request.method == 'POST':

        # gather input from web form using request.Form, which is a dictionary object
        cough = request.form['cough']
        fever = request.form['fever']
        sorethroat = request.form['sorethroat']
        shortnessofbreath = request.form['shortnessofbreath']
        headache = request.form['headache']
        sixtiesplus = request.form['sixtiesplus']
        gender = request.form['gender']
        contact = request.form['contact']
        abroad = request.form['abroad']
        
        # convert input elements into list and dictionary
        vars = ['cough', 'fever', 'sorethroat', 'shortnessofbreath', 'headache', 'sixtiesplus', 'gender', 'contact', 'abroad']
        X = [cough, fever, sorethroat, shortnessofbreath, headache, sixtiesplus, gender, contact, abroad]
        X = [int(i) for i in X]
        input_vars = {vars[i]: X[i] for i in range(len(vars))} 

        # prepare X for sklearn model
        X = np.array(X)
        if len(X.shape) == 1:
            X = X.reshape(1,-1)

        # pass X to predict y
        y = model.predict_proba( X )[:,1]*100
        y = np.append( np.round(y, 2), '' ) 
        predicted_covid_prob = '% '.join(map(str, y))
        
        # pass input variables and "predicted_prob" to the "render_template" function
        # display the predicted value on the webpage by rendering the "resultsform.html" file
        return render_template('main.html', 
                                original_input=input_vars,
                                prediction_prob=predicted_covid_prob)

# app.run() will start running the app on the host “localhost” on the port number 9999
# "debug": - during development the Flask server can reload the code without restarting the app
#          - also outputs useful debugging information
# visiting http://localhost:9999/ will render the "predictorform.html" page.
app.run("localhost", "9999", debug=True)