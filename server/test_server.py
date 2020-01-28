import argparse
# Requests HTTP lib
import requests
import datetime

# Flask server imports
from flask import Flask
from flask import request
from flask import Response
from flask import render_template

value = 1
status = 'Velocidad'


app = Flask(__name__)


@app.route('/', methods=['POST', 'GET'])
def index():
    #global current_chats
    global status, value

    if request.method == 'POST':
        req = request.get_json(force=True)
        app.logger.info(req)
        value = req.get('value')
        status = req.get('status')
    return 'hola'


# making list of pokemons



@app.route('/test')
def test():
    return render_template("index.html", value=value, status=status)


if __name__ == '__main__':
    #running app
    app.run(use_reloader=True, host='0.0.0.0', debug=True)
