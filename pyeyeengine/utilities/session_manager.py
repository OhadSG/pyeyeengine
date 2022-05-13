import os
import string
import random

LENGTH = 10
BASE_PATH = os.path.dirname(os.path.realpath(__file__))
SESSION_ID_PATH = BASE_PATH + "/../session_id.txt"

def create_session():
    session_id = ''.join(random.choice(string.ascii_lowercase + string.digits) for i in range(LENGTH))
    with open(SESSION_ID_PATH, "w") as file:
        file.write(session_id)
        file.close()
    return "{}".format(session_id)

def get_session_id():
    with open(SESSION_ID_PATH, "r") as file:
        session_id = file.read()
        file.close()
    return "{}".format(session_id)