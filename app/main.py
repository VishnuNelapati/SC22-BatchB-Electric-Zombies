# import requirements needed
from flask import Flask, render_template,request,redirect
from utils import get_base_url
import pickle
import pandas as pd
import numpy as np
import sklearn

# setup the webserver
# port may need to be changed if there are multiple flask servers running on same server
port = 12345
base_url = get_base_url(port)

# if the base url is not empty, then the server is running in development, and we need to specify the static folder so that the static files are served
if base_url == '/':
    app = Flask(__name__)
else:
    app = Flask(__name__, static_url_path=base_url + 'static')


# set up the routes and logic for the webserver
@app.route(f'{base_url}' , methods = ['POST','GET'])
def home():
    if request.method == "POST":
        filename = 'LogisticRegression.sav'
        loaded_model = pickle.load(open(filename, 'rb'))
        values = [i for i in request.form.values()]
        for i in range(len(values)):
            try:
                values[i] = int(values[i])
            except:
                pass

        ## Create a dataframe
        columns_list = ['gender', 'age', 'hypertension', 'heart_disease', 'ever_married','work_type', 'avg_glucose_level', 'bmi', 'smoking_status']
        dataframe = pd.DataFrame(data=values)
        dataframe = dataframe.T
        dataframe.columns = columns_list


        ## predict
#         print(loaded_model.predict_proba(dataframe))
#         print(type(loaded_model.predict_proba(dataframe)))
        results="results"
        chanceofStroke = round(loaded_model.predict_proba(dataframe)[0][1]*100,3)
        ModelPredictionText = "Our model predicted you might have a " + str(chanceofStroke) + "% chance of having a stroke"
        links = ["https://www.cdc.gov/stroke/prevention.htm","https://www.betterhealth.vic.gov.au/health/conditionsandtreatments/stroke-risk-factors-and-prevention","https://www.cdc.gov/stroke/risk_factors.htm","https://www.ninds.nih.gov/health-information/patient-caregiver-education/brain-basics-preventing-stroke","https://www.cdc.gov/stroke/prevention.htm","https://www.springvalleyhospital.com/services/stroke-services/warning-signs-risk-prevention","https://www.carrushealth.com/2020/04/10/recognize-these-early-warning-signs-of-a-stroke/","https://vnacare.org/community/health-tips/warning-signs-of-stroke","https://www.springvalleyhospital.com/services/stroke-services/warning-signs-risk-prevention","https://www.cdc.gov/stroke/treatments.htm","https://www.health.harvard.edu/womens-health/8-things-you-can-do-to-prevent-a-stroke"]
        linkstobereturned = []
        linkstexttobereturned = []
        links_text = ["Heart Stroke Prevention" , "Stroke Risk Factors and Preventions","CDC risk factors for strokes","Caregiver instructions for preventing a Stroke","Warning Signs for a Stroke","Do you recognize these early warning signs of a stroke","Health-Tips: Warning Signs of a stroke","Stroke Warning Signs and Prevention","Things can you do to prevent a stroke"]
        ## Display prediction on the website
        chanceofStroke = loaded_model.predict_proba(dataframe)[0][1]
        image1 = "http://www.secondscount.org/image.axd?id=79546346-4475-4e97-a0e7-4c1066e5ca61&t=635420557277400000"
        if chanceofStroke >=0.70:
            linkstobereturned = links[6:10]
            linkstexttobereturned = links_text[6:10]
            processed_text = "high chance of stroke"
            return render_template('index.html',processed_text=processed_text,image1=image1,results=results,links=links,linkstexttobereturned=linkstexttobereturned,linkstobereturned=linkstobereturned,ModelPredictionText=ModelPredictionText,trigger = True)
        elif chanceofStroke >=0.50:
            linkstobereturned = links[6:10]
            linkstexttobereturned = links_text[6:10]
            processed_text = "medium chance of stroke"
            return render_template('index.html',processed_text=processed_text,image1=image1,results=results,links=links,linkstexttobereturned=linkstexttobereturned,linkstobereturned=linkstobereturned,ModelPredictionText=ModelPredictionText,trigger = True)
        elif chanceofStroke >= 20:
            linkstobereturned = links[3:6]
            linkstexttobereturned = links_text[3:6]
            processed_text = "medium-low chance of stroke"
            return render_template('index.html',processed_text=processed_text,image1=image1,results=results,links=links,linkstexttobereturned=linkstexttobereturned,linkstobereturned=linkstobereturned,ModelPredictionText=ModelPredictionText,trigger = True)
        else:
            linkstobereturned = links[0:3]
            linkstexttobereturned = links_text[0:3]
            processed_text = "low chance of stroke"
            return render_template('index.html',processed_text=processed_text,image1=image1,results=results,links=links,linkstexttobereturned=linkstexttobereturned,linkstobereturned=linkstobereturned,ModelPredictionText=ModelPredictionText,trigger = True)


    return render_template('index.html',trigger = False)



@app.route(f'{base_url}/random_forest.html' , methods = ['POST','GET'])
def rf():
    return render_template('random_forest.html')

# set up the routes and logic for the webserver
@app.route(f'{base_url}/NeuralNetwork.html' , methods = ['POST','GET'])
def nn():
    return render_template('NeuralNetwork.html')


# set up the routes and logic for the webserver
@app.route(f'{base_url}/Logistic_Regression.html' , methods = ['POST','GET'])
def lr():
    return render_template('Logistic_Regression.html')

# set up the routes and logic for the webserver
@app.route(f'{base_url}/index.html' , methods = ['POST','GET'])
def index():
    return redirect(f'{base_url}')


if __name__ == '__main__':
    # IMPORTANT: change url to the site where you are editing this file.
    website_url = 'cocalc3.ai-camp.dev'

    print(f'Try to open\n\n    https://{website_url}' + base_url + '\n\n')
    app.run(host='0.0.0.0', port=port, debug=True)
