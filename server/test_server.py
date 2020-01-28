import argparse
# Requests HTTP lib
import requests
import datetime

# Flask server imports
from flask import Flask
from flask import request
from flask import Response
from flask import render_template

data = {1: ['Velocidad', 0],
        2: ['Velocidad', 0]}


app = Flask(__name__)


@app.route('/', methods=['POST', 'GET'])
def index():
    #global current_chats
    global data

    if request.method == 'POST':
        req = request.get_json(force=True)
        app.logger.info(req)

        data[req.get('id')] = [req.get('status'), req.get('value')]

    return 'hola'


# making list of pokemons



@app.route('/test')
def test():
    return render_template("index.html", value_1=data[1][1], status_1=data[1][0],
                           value_2=data[2][1], status_2=data[2][0])


if __name__ == '__main__':
    #running app
    app.run(use_reloader=True, host='0.0.0.0', debug=True)
