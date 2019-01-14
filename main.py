#IMPORTS
from naoqi import ALProxy

#VARIABLES
IP = "192.168.43.226"
PORT = 9559

TTSAPI = "ALTextToSpeech"
ANIMTTS = "ALAnimatedSpeech"
ALMOOD = "ALMood" # Reads instantaneous emotion of persons and ambiance.



def getAPIByName(name):
    obj = ALProxy(name,IP,PORT)
    return obj


if __name__ == "__main__":
    #animated = getAPIByName(ANIMTTS)
    print("Hola")