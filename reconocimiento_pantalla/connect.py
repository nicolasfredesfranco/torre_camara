from rauth import OAuth2Service
import json
import datetime
import requests

map_status_state = {'Falla':  2,  # parada
                    'Velocidad': 1,  # operativa
                    'Detenido': 0  # inactiva
                    }


class Connect2Server:
    def __init__(self, client_id='torre-client', client_secret='torre-secret'):
        self.access_token = 'XXXXX'
        self.refresh_token = None
        self.expires_in = None
        self.client_id = client_id
        self.client_secret = client_secret

        self.service = OAuth2Service(
            name="foo",
            client_id=client_id,
            client_secret=client_secret,
            access_token_url="http://femto.duckdns.org:8081/torre/oauth/token",
            authorize_url="http://femto.duckdns.org:8081/torre/oauth/token",
            base_url="http://femto.duckdns.org:8080/torre",
        )

        self.url_to_send = 'http://femto.duckdns.org:8081/torre/api/v1/mediciones/'

        #self.get_access_token()

    def get_access_token(self):
        data = {'grant_type': 'password',
                "username": "demo",
                "password": "a665a45920422f9d417e4867efdc4fb8a04a1f3fff1fa07e998e86f7f7a27ae3"
                }
        req = self.service.get_raw_access_token('POST', data=data)
        data_req = req.json()
        
        self.access_token = data_req.get('access_token')
        self.refresh_token = data_req.get('refresh_token')
        self.expires_in = data_req.get('expires_in')

    def send(self, number, status, idMaquina=1):
        valorDetectado = str(number)
        idEstadoMaquina = map_status_state[status]
        now = datetime.datetime.now().isoformat()

        data = {'idMaquina': idMaquina,
                 'fecha': now,
                 'idEstadoMaquina': idEstadoMaquina,
                 'valorDetectado': valorDetectado
                }

        headers = {'content-type': 'application/json',
                   'Authorization': "bearer " + self.access_token}

        try:
            response = requests.post(self.url_to_send, headers=headers, json=data, timeout=1)
            if response.status_code == 201:
                return 0
            elif response.status_code == 401:
                print('resfrescar token')
                self.get_access_token()
                return 1
            else:
                print('error', response.status_code)
        except requests.exceptions.RequestException as e:
            return 2
    
        
    #def handle_error_connect():

    #    while True:
    #        print('ñe')
        
    #def refresh_token(self):
        #if not expired():
        #    return

        # OAuth 2.0 example
        #data = {'client_id': self.client_id,
        #        'client_secret': self.client_secret,
        #        'grant_type': 'refresh_token',
        #        'refresh_token': self.refresh_token}

        #return self.service.get_access_token(data=data)










