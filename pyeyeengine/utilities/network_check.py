import os
from urllib.request import urlopen

def can_access_internet():
    try:
        response = urlopen('https://www.google.com/', timeout=4)
        return True
    except:
        return False