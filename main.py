from naoqi import ALProxy

TTSAPI = "ALTextToSpeech"
ANIMTTS = "ALAnimatedSpeech"
ALMOOD = "ALMood" # Reads instantaneous emotion of persons and ambiance.

class StoryTeller:
	def get_session(self, name):
	    return ALProxy(name, self.ip, self.port)

	def __init__(self):
		self.ip = "192.168.43.226"
		#self.ip = "127.0.0.1"
		self.port = 9559
		self.init_stories()
		self.tts = self.get_session(TTSAPI)
		self.memory = self.get_session("ALMemory")
		self.anim = self.get_session("ALAnimationPlayer")


	def init_stories(self):
		self.story = \
			"Long ago, there lived a lion in a dense forest. One morning his \
			wife told him \\mrk=0\\ that his breath was bad and unpleasant. \
			The lion became embarrassed and angry upon hearing it. \
			He wanted to check this fact with others. So he called three \
			others outside his cave."

# tts.say("\\bound=S\\ \\vol=80\\ \\vct=60\\ \\rspd=90\\ {}".format(text))
# http://doc.aldebaran.com/2-1/naoqi/audio/altexttospeech-tuto.html	

	def bookmark_handler(self,value):
		self.anim.run('animations/Stand/Emotions/Negative/Angry_3')

	def main(self):
		#tts = story_teller.getSession(TTSAPI)
		sub = self.memory.subscriber("ALTextToSpeech/CurrentBookMark")
		sub.signal.connect("bookmark_handler")
		self.tts.say(self.story)
    

if __name__ == "__main__":
	story_teller = StoryTeller()
	story_teller.main()