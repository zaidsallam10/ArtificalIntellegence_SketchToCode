from flask import Flask
from flask.json import jsonify

app = Flask(__name__)


# use (flask) work in the terminal to more info
# you've to set flask_app
# you've to set flask_env


@app.route('/')
def hello_world():
    return 'Hello world!\n'


@app.route('/employees')
def employees():
    return jsonify(['zaid', 'dali', 'mohd']);


@app.route('/numbers')
def numbers():
    return jsonify([1, 2, 3, 4, 5, 5, 6]);


app.run(debug=True)
