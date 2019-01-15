from naoqi import ALProxy
import naoqi 
from random import randint


TTSAPI = "ALTextToSpeech"
ANIMTTS = "ALAnimatedSpeech"
ALMOOD = "ALMood" # Reads instantaneous emotion of persons and ambiance.
ALANIM = "ALAnimationPlayer"

class StoryTeller:
	def get_session(self, name):
	    return ALProxy(name, self.ip, self.port)

	def __init__(self):
		self.ip = "169.254.61.137"
		#self.ip = "127.0.0.1"
		self.port = 9559
		self.init_stories()
		self.init_gestures()
		self.tts = self.get_session(TTSAPI)
		self.memory = self.get_session("ALMemory")
		self.anim = self.get_session("ALAnimationPlayer")
		#self.init_gesture_handler()


	def init_stories(self):
		self.story = \
			"\\bound=S\\ \\vol=80\\ \\vct=60\\ \\rspd=80\\ \
			Long ago, there lived a lion in a dense forest. One morning his \
			wife told him that his breath was bad and unpleasant.  \
			The lion became embarrassed and angry upon hearing it. \
			He wanted to check this fact with others. So he called three \
			others outside his cave."
		
		new_story = []
		for sentence in self.story.split("."):
			sentence_words = sentence.split(" ")
			middle = len(sentence_words) / 2
			new_sentence = " ".join(sentence_words[:middle]) + \
						   " \\mrk=0\\ " + \
						   " ".join(sentence_words[middle:])
			new_story += [new_sentence]
		self.story = ".".join(new_story)
		#self.story = " \\mrk=0\\ " + self.story.replace(".", ". \\mrk=0\\ ")
		print(self.story)

	def init_gestures(self):
		self.gestures_dict = {'0': ['animations/Stand/BodyTalk/Speaking/BodyTalk_11', 'animations/Stand/BodyTalk/Speaking/BodyTalk_17', 'animations/Stand/BodyTalk/Speaking/BodyTalk_18', 'animations/Stand/BodyTalk/Speaking/BodyTalk_4', 'animations/Stand/BodyTalk/Speaking/BodyTalk_5', 'animations/Stand/Gestures/Explain_1', 'animations/Stand/Gestures/Explain_4', 'animations/Stand/Gestures/Explain_8', 'animations/Stand/Gestures/Explain_9', 'animations/Stand/Self & others/NAO/Left_Neutral_SAO_01', 'animations/Stand/Self & others/NAO/Left_Neutral_SAO_02', 'animations/Stand/Self & others/NAO/Left_Strong_SAO_02', 'animations/Stand/Negation/NAO/Center_Strong_NEG_01', 'animations/Stand/Exclamation/NAO/Center_Neutral_EXC_01', 'animations/Stand/BodyTalk/BodyLanguage/NAO/Center_Slow_AFF_01', 'animations/Stand/BodyTalk/BodyLanguage/NAO/Center_Strong_AFF_01'], 
						 	  '+': ['animations/Stand/Emotions/Positive/Confident_1', 'animations/Stand/Emotions/Positive/Happy_4', 'animations/Stand/Emotions/Positive/Proud_2', 'animations/Stand/Self & others/NAO/Center_Neutral_SAO_03', 'animations/Stand/Exclamation/NAO/Left_Strong_EXC_03', 'animations/Stand/Exclamation/NAO/Right_Strong_EXC_04', 'animations/Stand/BodyTalk/BodyLanguage/NAO/Center_Slow_AFF_03', 'animations/Stand/Space & time/NAO/Right_Slow_SAT_01'], 
						 	  '-': ['animations/Stand/Emotions/Negative/Fearful_1', 'animations/Stand/Gestures/Desperate_2', 'animations/Stand/Gestures/Desperate_5', 'animations/Stand/Negation/NAO/Center_Strong_NEG_05', 'animations/Stand/BodyTalk/BodyLanguage/NAO/Center_Strong_AFF_05']}

	def init_gesture_handler(self):
		print("init handler")
		memory_service = self.memory.session().service("ALMemory")
		sub = memory_service.subscriber("ALTextToSpeech/CurrentBookMark")
		sub.signal.connect(self.bookmark_handler)

# tts.say("\\bound=S\\ \\vol=80\\ \\vct=60\\ \\rspd=90\\ {}".format(text))
# http://doc.aldebaran.com/2-1/naoqi/audio/altexttospeech-tuto.html	

	def bookmark_handler(self, value):
		num_to_sentiment = {0: "-", 1: "0", 2:"+"}
		sentiment = num_to_sentiment[randint(0, 2)]
		print(sentiment)
		gesture_index = randint(0, len(self.gestures_dict[sentiment]) - 1)
		gesture = self.gestures_dict[sentiment][gesture_index]
		print(gesture)
		self.anim.run(gesture)

	def main(self):
		memory_service = self.memory.session().service("ALMemory")
		sub = memory_service.subscriber("ALTextToSpeech/CurrentBookMark")
		sub.signal.connect(self.bookmark_handler)
		self.tts.say(self.story)
		
		
		

if __name__ == "__main__":
	story_teller = StoryTeller()
	story_teller.main()