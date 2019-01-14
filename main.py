from naoqi import ALProxy

TTSAPI = "ALTextToSpeech"
ANIMTTS = "ALAnimatedSpeech"
ALMOOD = "ALMood" # Reads instantaneous emotion of persons and ambiance.

class StoryTeller:
	def __init__(self):
		self.ip = "192.168.43.226"
		#self.ip = "127.0.0.1"
		self.port = 9559
		self.init_stories()

	def init_stories(self):
		self.story = \
			"Long ago, there lived a lion in a dense forest. One morning his \
			wife told him that his breath was bad and unpleasant. \
			The lion became embarrassed and angry upon hearing it. \
			He wanted to check this fact with others. So he called three \
			others outside his cave."

	def getAPIByName(self, name):
	    return ALProxy(name, self.ip, self.port)

	def main(self):
		tts = story_teller.getAPIByName(TTSAPI)
		tts.say(self.story)
    

if __name__ == "__main__":
	story_teller = StoryTeller()
	story_teller.main()