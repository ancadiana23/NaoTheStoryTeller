import json
import matplotlib.pyplot as plt
import naoqi 
import re

from naoqi import ALProxy
from random import randint
from stanfordcorenlp import StanfordCoreNLP


TTSAPI = "ALTextToSpeech"
ANIMTTS = "ALAnimatedSpeech"
ALMOOD = "ALMood" # Reads instantaneous emotion of persons and ambiance.
ALANIM = "ALAnimationPlayer"


class StoryTeller:
	def get_session(self, name):
	    return ALProxy(name, self.ip, self.port)

	def __init__(self):
		self.init_stories()
		self.init_gestures()
		#self.init_gesture_handler()
		self.init_sentiment_analysis_client()


	def init_robot_connection(self):
		self.ip = "169.254.63.184"
		self.port = 9559
		self.tts = self.get_session(TTSAPI)
		self.memory = self.get_session("ALMemory")
		self.anim = self.get_session("ALAnimationPlayer")
	
	def init_stories(self):
		'''
		self.story = \
			'\\bound=S\\ \\vol=80\\ \\vct=100\\ \\rspd=80\\ \
			Long ago, there lived a lion in a dense forest. One morning his \
			wife told him that his breath was bad and unpleasant.  \
			The lion became embarrassed and angry upon hearing it. \
			He wanted to check this fact with others. So he called three \
			others outside his cave. \
			First came the sheep. The Lion opening his mouth wide said, \
			"Sheep, tell me if my mouth smells bad?" \
			The sheep thought that the lion wanted an honest answer, so the sheep said, \
			"Yes, Friend. There seems to be something wrong with your breath". \
			This plain speak did not go well with the lion. He pounced on the sheep, \
			killing it.'
		'''
		self.story = \
			'\\bound=S\\ \\vol=80\\ \\vct=100\\ \\rspd=80\\ \
			The little mouse. \
			Once upon a time, there was a Baby Mouse and Mother Mouse. \
			They lived in a hole in the skirting board in a big, warm house \
			with lots of cheese to eat, where they wanted for nothing. \
			Then, one day, Mother Mouse decided to take Baby Mouse outside of their home. \
			Waiting outside for them was a huge ginger tomcat, licking its lips \
			and waiting to eat them both up. \
			"Mother, Mother! What should we do?" Cried Baby Mouse, clinging to his mother\'s tail. \
			Mother Mouse paused, staring up into the beady eyes of the hungry cat. \
			But she wasn\'t scared because she knew exactly how to deal with big, scary cats. \
			She opened her mouth and took in a deep breath. \
			"Woof! Woof! Bark bark bark!" She shouted, and the cat ran away as fast as he could.\
			"Wow, Mother! That was amazing!" Baby Mouse said to his mother, smiling happily.\
			"And that, my child, is why it is always best to have a second language." \
			Moral: It\'s always good to have a second language.'

		new_story = []
		index = 0
		self.sentences = re.split("\\.|\"", self.story) 
		for sentence in self.sentences:
			sentence_words = sentence.split(" ")
			middle = len(sentence_words) / 2
			new_sentence = " ".join(sentence_words[:middle]) + \
						   " \\mrk=" + str(index) + "\\ " + \
						   " ".join(sentence_words[middle:])
			index += 1
			new_story += [new_sentence]
		self.story = ".".join(new_story)
		#self.story = " \\mrk=0\\ " + self.story.replace(".", ". \\mrk=0\\ ")
		#print(self.story)
		for sen in self.sentences:
			print(sen)


	def init_gestures(self):
		self.gestures_dict = {'0': ['animations/Stand/BodyTalk/Speaking/BodyTalk_11', 'animations/Stand/BodyTalk/Speaking/BodyTalk_17', 'animations/Stand/BodyTalk/Speaking/BodyTalk_18', 'animations/Stand/BodyTalk/Speaking/BodyTalk_4', 'animations/Stand/BodyTalk/Speaking/BodyTalk_5', 'animations/Stand/Gestures/Explain_1', 'animations/Stand/Gestures/Explain_4', 'animations/Stand/Gestures/Explain_8', 'animations/Stand/Gestures/Explain_9', 'animations/Stand/Self & others/NAO/Left_Neutral_SAO_01', 'animations/Stand/Self & others/NAO/Left_Neutral_SAO_02', 'animations/Stand/Self & others/NAO/Left_Strong_SAO_02', 'animations/Stand/Negation/NAO/Center_Strong_NEG_01', 'animations/Stand/Exclamation/NAO/Center_Neutral_EXC_01', 'animations/Stand/BodyTalk/BodyLanguage/NAO/Center_Slow_AFF_01', 'animations/Stand/BodyTalk/BodyLanguage/NAO/Center_Strong_AFF_01'], 
						 	  '+': ['animations/Stand/Emotions/Positive/Confident_1', 'animations/Stand/Emotions/Positive/Happy_4', 'animations/Stand/Emotions/Positive/Proud_2', 'animations/Stand/Self & others/NAO/Center_Neutral_SAO_03', 'animations/Stand/Exclamation/NAO/Left_Strong_EXC_03', 'animations/Stand/Exclamation/NAO/Right_Strong_EXC_04', 'animations/Stand/BodyTalk/BodyLanguage/NAO/Center_Slow_AFF_03', 'animations/Stand/Space & time/NAO/Right_Slow_SAT_01'], 
						 	  '-': ['animations/Stand/Emotions/Negative/Fearful_1', 'animations/Stand/Gestures/Desperate_2', 'animations/Stand/Gestures/Desperate_5', 'animations/Stand/Negation/NAO/Center_Strong_NEG_05', 'animations/Stand/BodyTalk/BodyLanguage/NAO/Center_Strong_AFF_05']}
		self.gensture_to_sentiment = {'animations/Stand/Emotions/Negative/Fearful_1': -0.9, 'animations/Stand/BodyTalk/Speaking/BodyTalk_18': -0.3, 'animations/Stand/BodyTalk/BodyLanguage/NAO/Center_Slow_AFF_01': -0.1, 'animations/Stand/BodyTalk/BodyLanguage/NAO/Center_Slow_AFF_03': 0.5, 'animations/Stand/BodyTalk/Speaking/BodyTalk_11': 0.1, 'animations/Stand/BodyTalk/Speaking/BodyTalk_17': -0.1, 'animations/Stand/Emotions/Positive/Proud_2': 0.8, 'animations/Stand/Emotions/Positive/Confident_1': 0.8, 'animations/Stand/Exclamation/NAO/Center_Neutral_EXC_01': 0.3, 'animations/Stand/Exclamation/NAO/Right_Strong_EXC_04': 0.7, 'animations/Stand/BodyTalk/Speaking/BodyTalk_5': 0.2, 'animations/Stand/BodyTalk/Speaking/BodyTalk_4': 0.2, 'animations/Stand/Negation/NAO/Center_Strong_NEG_01': 0.5, 'animations/Stand/Self & others/NAO/Center_Neutral_SAO_03': 0.9, 'animations/Stand/Gestures/Explain_4': -0.1, 'animations/Stand/Gestures/Explain_1': 0.1, 'animations/Stand/Emotions/Positive/Happy_4': 1.0, 'animations/Stand/BodyTalk/BodyLanguage/NAO/Center_Strong_AFF_01': 0.0, 'animations/Stand/Gestures/Explain_9': 0.0, 'animations/Stand/Gestures/Explain_8': 0.0, 'animations/Stand/BodyTalk/BodyLanguage/NAO/Center_Strong_AFF_05': -0.75, 'animations/Stand/Negation/NAO/Center_Strong_NEG_05': -0.8, 'animations/Stand/Self & others/NAO/Left_Neutral_SAO_01': 0.3, 'animations/Stand/Self & others/NAO/Left_Neutral_SAO_02': 0.2, 'animations/Stand/Self & others/NAO/Left_Strong_SAO_02': 0.2, 'animations/Stand/Space & time/NAO/Right_Slow_SAT_01': 0.6, 'animations/Stand/Exclamation/NAO/Left_Strong_EXC_03': 0.7, 'animations/Stand/Gestures/Desperate_5': -0.7, 'animations/Stand/Gestures/Desperate_2': -0.5}


	def init_gesture_handler(self):
		print("init handler")
		self.memory_service = self.memory.session().service("ALMemory")
		self.sub = memory_service.subscriber("ALTextToSpeech/CurrentBookMark")
		self.sub.signal.connect(self.bookmark_handler)


	def init_sentiment_analysis_client(self):
		self.nlp = StanfordCoreNLP('http://localhost', port=9000, timeout=30000)
		self.props = {
            'annotators': 'sentiment',
            'pipelineLanguage': 'en',
            'outputFormat': 'json'
        }


	def quality_function(self, sentence, gesture):
		annotation = json.loads(self.nlp.annotate(sentence, properties=self.props))
		if len(annotation["sentences"]) == 0:
			print(sentence)
			return 0.0
		sentiment_distribution = annotation["sentences"][0]["sentimentDistribution"]
		normalizer = len(sentiment_distribution) / 2
		sentiment_distribution = [(i - normalizer) * sentiment_distribution[i] \
								  for i in range(len(sentiment_distribution))]
		print(sentiment_distribution)
		sentiment = sum(sentiment_distribution) / 2.0

		quality = self.gensture_to_sentiment[gesture] * sentiment
		
		print(sentiment, self.gensture_to_sentiment[gesture], quality)
		return quality

	def choose_gesture(self):
		num_to_sentiment = {0: "-", 1: "0", 2:"+"}
		sentiment = num_to_sentiment[randint(0, 2)]
		print(sentiment)
		gesture_index = randint(0, len(self.gestures_dict[sentiment]) - 1)
		gesture = self.gestures_dict[sentiment][gesture_index]
		print(gesture)
		return gesture


	def bookmark_handler(self, value):
		print("Value = ", value)
		gesture = self.choose_gesture()
		self.anim.run(gesture)
		quality = self.quality_function(self.sentences[int(value)], gesture)
		self.quality_sum += quality


	def main(self):
		self.init_robot_connection()
		self.quality_sum = 0.0
		memory_service = self.memory.session().service("ALMemory")
		self.sub = memory_service.subscriber("ALTextToSpeech/CurrentBookMark")
		self.sub.signal.connect(self.bookmark_handler)
		self.tts.say(self.story)
		
		self.quality = quality / len(self.sentences)
		print(self.quality_sum)


	def simulate(self):
		self.quality_sum = 0
		for sentence in self.sentences:
			gesture = self.choose_gesture()
			quality = self.quality_function(sentence, gesture)
			self.quality_sum += quality
		self.quality_sum = self.quality_sum / len(self.sentences)
		print(self.quality_sum)
		return self.quality_sum

def run_simulations(story_teller):
	simulation_range = range(10)
	simulations = [story_teller.simulate() for _ in simulation_range]
	print(simulations)
	plt.ylim((-1.0, 1.0))
	plt.plot(simulation_range, simulations)
	plt.show()

if __name__ == "__main__":
	story_teller = StoryTeller()
	#story_teller.main()
	run_simulations(story_teller)
	

