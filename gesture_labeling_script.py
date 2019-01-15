from naoqi import ALProxy
import naoqi 


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
		self.tts = self.get_session(TTSAPI)
		self.memory = self.get_session("ALMemory")
		self.anim = self.get_session("ALAnimationPlayer")


	def init_stories(self):
		self.story = \
			"\\bound=S\\ \\vol=80\\ \\vct=60\\ \\rspd=90\\ \
			Long ago, there lived a lion in a dense forest. One morning his \
			wife told him \\mrk=0\\ that his breath was bad and unpleasant. \
			The lion became embarrassed and angry upon hearing it. \
			He wanted to check this fact with others. So he called three \
			others outside his cave."


# tts.say("\\bound=S\\ \\vol=80\\ \\vct=60\\ \\rspd=90\\ {}".format(text))
# http://doc.aldebaran.com/2-1/naoqi/audio/altexttospeech-tuto.html	

	def bookmark_handler(self, value):
		self.anim.run('animations/Stand/Emotions/Negative/Angry_1')

	def main(self):
		memory_service = self.memory.session().service("ALMemory")
		sub = memory_service.subscriber("ALTextToSpeech/CurrentBookMark")
		#sub.signal.connect(self.bookmark_handler)
		#self.tts.say(self.story)
		#annotated = {}
		gest = {"+":[],"0":[],"-":[]}
		
		gestures_list = ['animations/Stand/BodyTalk/Speaking/BodyTalk_11','animations/Stand/Negation/NAO/Right_Strong_NEG_01', 'animations/Stand/BodyTalk/Speaking/BodyTalk_14', 'animations/Stand/BodyTalk/Speaking/BodyTalk_17', 'animations/Stand/BodyTalk/Speaking/BodyTalk_18', 'animations/Stand/BodyTalk/Speaking/BodyTalk_19', 'animations/Stand/BodyTalk/Speaking/BodyTalk_2', 'animations/Stand/BodyTalk/Speaking/BodyTalk_4', 'animations/Stand/BodyTalk/Speaking/BodyTalk_5', 'animations/Stand/BodyTalk/Speaking/BodyTalk_6', 'animations/Stand/Emotions/Negative/Disappointed_1', 'animations/Stand/Emotions/Negative/Fearful_1', 'animations/Stand/Emotions/Negative/Frustrated_1', 'animations/Stand/Emotions/Neutral/AskForAttention_2', 'animations/Stand/Emotions/Positive/Confident_1', 'animations/Stand/Emotions/Positive/Excited_1', 'animations/Stand/Emotions/Positive/Happy_1', 'animations/Stand/Emotions/Positive/Happy_4', 'animations/Stand/Emotions/Positive/Proud_2', 'animations/Stand/Emotions/Positive/Sure_1', 'animations/Stand/Gestures/Desperate_1', 'animations/Stand/Gestures/Desperate_2', 'animations/Stand/Gestures/Desperate_5', 'animations/Stand/Gestures/Enthusiastic_1', 'animations/Stand/Gestures/Everything_1', 'animations/Stand/Gestures/Explain_1', 'animations/Stand/Gestures/Explain_10', 'animations/Stand/Gestures/Explain_3', 'animations/Stand/Gestures/Explain_4', 'animations/Stand/Gestures/Explain_5', 'animations/Stand/Gestures/Explain_6','animations/Stand/Gestures/Explain_8', 'animations/Stand/Gestures/Explain_9', 'animations/Stand/Gestures/Yes_3', 'animations/Stand/Gestures/You_3', 'animations/Stand/Self & others/NAO/Left_Neutral_SAO_01', 'animations/Stand/Self & others/NAO/Left_Neutral_SAO_06', 'animations/Stand/Self & others/NAO/Right_Neutral_SAO_02', 'animations/Stand/Self & others/NAO/Left_Neutral_SAO_02', 'animations/Stand/Self & others/NAO/Left_Strong_SAO_02', 'animations/Stand/Self & others/NAO/Center_Neutral_SAO_03', 'animations/Stand/Negation/NAO/Left_Strong_NEG_01', 'animations/Stand/Negation/NAO/Left_Strong_NEG_03', 'animations/Stand/Negation/NAO/Right_Strong_NEG_04', 'animations/Stand/Negation/NAO/Center_Neutral_NEG_02', 'animations/Stand/Negation/NAO/Center_Strong_NEG_01', 'animations/Stand/Negation/NAO/Center_Strong_NEG_05', 'animations/Stand/Exclamation/NAO/Left_Strong_EXC_03', 'animations/Stand/Exclamation/NAO/Center_Neutral_EXC_01', 'animations/Stand/Exclamation/NAO/Right_Strong_EXC_04', 'animations/Stand/BodyTalk/BodyLanguage/NAO/Center_Neutral_AFF_07', 'animations/Stand/BodyTalk/BodyLanguage/NAO/Center_Strong_AFF_06', 'animations/Stand/BodyTalk/BodyLanguage/NAO/Center_Slow_AFF_01', 'animations/Stand/BodyTalk/BodyLanguage/NAO/Center_Strong_AFF_05', 'animations/Stand/BodyTalk/BodyLanguage/NAO/Center_Neutral_AFF_01', 'animations/Stand/BodyTalk/BodyLanguage/NAO/Right_Slow_AFF_01', 'animations/Stand/BodyTalk/BodyLanguage/NAO/Center_Strong_AFF_01', 'animations/Stand/BodyTalk/BodyLanguage/NAO/Center_Slow_AFF_03', 'animations/Stand/Space & time/NAO/Left_Neutral_SAT_06', 'animations/Stand/Space & time/NAO/Left_Neutral_SAT_01', 'animations/Stand/Space & time/NAO/Right_Slow_SAT_01', 'animations/Stand/Space & time/NAO/Right_Strong_SAT_03', 'animations/Stand/Space & time/NAO/Left_Neutral_SAT_02']
		gestures_dict = {'0': ['animations/Stand/BodyTalk/Speaking/BodyTalk_11', 'animations/Stand/BodyTalk/Speaking/BodyTalk_17', 'animations/Stand/BodyTalk/Speaking/BodyTalk_18', 'animations/Stand/BodyTalk/Speaking/BodyTalk_4', 'animations/Stand/BodyTalk/Speaking/BodyTalk_5', 'animations/Stand/Gestures/Explain_1', 'animations/Stand/Gestures/Explain_4', 'animations/Stand/Gestures/Explain_8', 'animations/Stand/Gestures/Explain_9', 'animations/Stand/Self & others/NAO/Left_Neutral_SAO_01', 'animations/Stand/Self & others/NAO/Left_Neutral_SAO_02', 'animations/Stand/Self & others/NAO/Left_Strong_SAO_02', 'animations/Stand/Negation/NAO/Center_Strong_NEG_01', 'animations/Stand/Exclamation/NAO/Center_Neutral_EXC_01', 'animations/Stand/BodyTalk/BodyLanguage/NAO/Center_Slow_AFF_01', 'animations/Stand/BodyTalk/BodyLanguage/NAO/Center_Strong_AFF_01'], 
						 '+': ['animations/Stand/Emotions/Positive/Confident_1', 'animations/Stand/Emotions/Positive/Happy_4', 'animations/Stand/Emotions/Positive/Proud_2', 'animations/Stand/Self & others/NAO/Center_Neutral_SAO_03', 'animations/Stand/Exclamation/NAO/Left_Strong_EXC_03', 'animations/Stand/Exclamation/NAO/Right_Strong_EXC_04', 'animations/Stand/BodyTalk/BodyLanguage/NAO/Center_Slow_AFF_03', 'animations/Stand/Space & time/NAO/Right_Slow_SAT_01'], 
						 '-': ['animations/Stand/Emotions/Negative/Fearful_1', 'animations/Stand/Gestures/Desperate_2', 'animations/Stand/Gestures/Desperate_5', 'animations/Stand/Negation/NAO/Center_Strong_NEG_05', 'animations/Stand/BodyTalk/BodyLanguage/NAO/Center_Strong_AFF_05']}
		
		for g in gestures_list:	
			q = "r"		
			while q == "r":
				print("Do we want {}?".format(g))
				self.anim.run(g)
				q = raw_input()
			if q == "n":
				gestures_list.remove(g)	
			else:
				print("Sentiment?")
				answer = raw_input()
				gest[answer].append(g)
			
		print(gest)
    

if __name__ == "__main__":
	story_teller = StoryTeller()
	story_teller.main()