# import libraries
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

# create app and load the trained Model
app = Flask(__name__)

# Route to handle HOME
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle PREDICTED RESULT
@app.route('/',methods=['POST'])
def predict():
  
    inputs = [] # declaring input array
    
    return render_template('index.html', predicted_result = 'Results')

if __name__ == "__main__":
    app.run(debug=True)
