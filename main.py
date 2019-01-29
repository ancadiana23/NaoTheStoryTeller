import json
import matplotlib.pyplot as plt
import naoqi 
import numpy as np
import re

from ast import literal_eval as make_tuple
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
		#self.init_stories()
		#self.init_gestures()
		#self.init_gesture_handler()
		#self.init_sentiment_analysis_client()
		return

	def init_robot_connection(self):
		self.ip = "169.254.156.88"
		self.port = 9559
		self.tts = self.get_session(TTSAPI)
		self.memory = self.get_session("ALMemory")
		self.leds = self.get_session("ALLeds")
		self.anim = self.get_session("ALAnimationPlayer")
	

	def init_stories(self):
		self.story = \
			'\\bound=S\\ \\vol=80\\ \\vct=100\\ \\rspd=80\\' \
			'The little mouse. '\
			'Once upon a time, there was a Baby Mouse and Mother Mouse. '\
			'They lived in a hole in the skirting board in a big, warm house '\
			'with lots of cheese to eat, where they wanted for nothing. '\
			'Then, one day, Mother Mouse decided to take Baby Mouse outside of their home. '\
			'Waiting outside for them was a huge ginger tomcat, licking its lips '\
			'and waiting to eat them both up. '\
			'"Mother, Mother! What should we do?" Cried Baby Mouse, clinging to his mother\'s tail. '\
			'Mother Mouse paused, staring up into the beady eyes of the hungry cat. '\
			'But she wasn\'t scared because she knew exactly how to deal with big, scary cats. '\
			'She opened her mouth and took in a deep breath. '\
			'"Woof! Woof! Bark bark bark!" She shouted, and the cat ran away as fast as he could.'\
			'"Wow, Mother! That was amazing!" Baby Mouse said to his mother, smiling happily.'\
			'"And that, my child, is why it is always best to have a second language." '\
			'\n	Moral: It\'s always good to have a second language.'
		
		# Split the text into sentences using a regular expression that matches 
		# combinations of the delimiters '"' and '.'.
		# The combinations are contained in a capture group because they will be
		# used to reconstruct the original text after adding the bookmarks. 
		self.sentences = re.split("((?:[\\.\"]+\\s*)+)", self.story)

		# Filter out the empty sentences [optional].
		self.sentences = filter(None, self.sentences)

		# Add a bookmark to every delimiter, by changing every string at an odd
		# index in the list. 
		# The first sentence should not have any gesture and the indexes for
		# the senteces and gestures should match. Thus, the indexes for gestures 
		# will start from 1. 
		new_story = [self.sentences[i] if i%2 == 0 else \
					 self.sentences[i] + " \\mrk=" + str(i/2 + 1) + "\\ "  \
					 for i in range(len(self.sentences))]
		# Reconstruct the story.
		self.story = "".join(new_story)

		# Filter out the delimeters from the sentences list.
		self.sentences = self.sentences[0::2]
		#print(self.story)
		#print(self.sentences)
	

	def init_story(self):
		with open("aesopFables.json") as file:
			dataset = json.load(file)["stories"]
		self.story = dataset[44]
		self.sentences = self.story["story"]
		processed_story = [" \\mrk=" + str(i + 1) + "\\ " + str(self.sentences[i]) \
					 for i in range(len(self.sentences))]
		
		# Reconstruct the story.
		# "\\bound=S\\ \\vol=80\\ \\vct=100\\ \\rspd=85\\" + \
		self.story["story"] = "\\rspd=85\\" + \
							  "".join(processed_story)


	def init_policy(self):
		with open("best_policy.json") as file:
			policy = json.load(file)
		self.policy = {make_tuple(x): str(y) for (x, y) in policy.items()}


	def get_corenlp_sentiment(self, sentence):
		annotation = json.loads(self.nlp.annotate(sentence, \
								properties=self.props))
		sentiment_distribution = annotation["sentences"][0]["sentimentDistribution"]
		return sentiment_distribution
	
	def init_sentence_to_sentiment(self):
		self.sentences_sentiment_distribution = \
			[self.get_corenlp_sentiment(sentence) for sentence in self.sentences]


	def init_gesture_list(self):
		self.gesture_list = list(self.gesture_to_probability.keys())


	def init_gestures(self):
		self.gesture_to_probability = \
		{'animations/Stand/Emotions/Negative/Fearful_1': [0.718, 0.138, 0.048, 0.048, 0.048], 
		 'animations/Stand/BodyTalk/Speaking/BodyTalk_18': [0.25, 0.3, 0.25, 0.1, 0.1], 
		 'animations/Stand/BodyTalk/BodyLanguage/NAO/Center_Slow_AFF_01': [0.05, 0.2, 0.5, 0.2, 0.05], 
		 'animations/Stand/BodyTalk/BodyLanguage/NAO/Center_Slow_AFF_03': [0.05, 0.05, 0.2, 0.5, 0.2], 
		 'animations/Stand/BodyTalk/Speaking/BodyTalk_11': [0.05, 0.2, 0.5, 0.2, 0.05], 
		 'animations/Stand/BodyTalk/Speaking/BodyTalk_17': [0.03, 0.12, 0.7, 0.12, 0.03], 
		 'animations/Stand/Emotions/Positive/Proud_2': [0.048, 0.048, 0.048, 0.138, 0.718], 
		 'animations/Stand/Emotions/Positive/Confident_1': [0.08, 0.08, 0.08, 0.23, 0.53], 
		 'animations/Stand/Exclamation/NAO/Center_Neutral_EXC_01': [0.1, 0.1, 0.25, 0.3, 0.25], 
		 'animations/Stand/Exclamation/NAO/Right_Strong_EXC_04': [0.048, 0.048, 0.048, 0.138, 0.718], 
		 'animations/Stand/BodyTalk/Speaking/BodyTalk_5': [0.05, 0.05, 0.2, 0.5, 0.2], 
		 'animations/Stand/BodyTalk/Speaking/BodyTalk_4': [0.03, 0.12, 0.7, 0.12, 0.03], 
		 'animations/Stand/BodyTalk/BodyLanguage/NAO/Center_Strong_AFF_05': [0.53, 0.23, 0.08, 0.08, 0.08], 
		 'animations/Stand/Self & others/NAO/Center_Neutral_SAO_03': [0.1, 0.25, 0.3, 0.25, 0.1], 
		 'animations/Stand/Gestures/Explain_4': [0.1, 0.25, 0.3, 0.25, 0.1], 
		 'animations/Stand/Gestures/Explain_1': [0.1, 0.25, 0.3, 0.25, 0.1], 
		 'animations/Stand/Emotions/Positive/Happy_4': [0.048, 0.048, 0.048, 0.138, 0.718], 
		 'animations/Stand/BodyTalk/BodyLanguage/NAO/Center_Strong_AFF_01': [0.05, 0.2, 0.5, 0.2, 0.05], 
		 'animations/Stand/Gestures/Explain_9': [0.05, 0.2, 0.5, 0.2, 0.05], 
		 'animations/Stand/Gestures/Explain_8': [0.05, 0.2, 0.5, 0.2, 0.05], 
		 'animations/Stand/Negation/NAO/Center_Strong_NEG_01': [0.08, 0.08, 0.08, 0.23, 0.53], 
		 'animations/Stand/Negation/NAO/Center_Strong_NEG_05': [0.53, 0.23, 0.08, 0.08, 0.08], 
		 'animations/Stand/Self & others/NAO/Left_Neutral_SAO_01': [0.05, 0.2, 0.5, 0.2, 0.05], 
		 'animations/Stand/Self & others/NAO/Left_Neutral_SAO_02': [0.05, 0.05, 0.2, 0.5, 0.2], 
		 'animations/Stand/Self & others/NAO/Left_Strong_SAO_02': [0.05, 0.05, 0.2, 0.5, 0.2], 
		 'animations/Stand/Space & time/NAO/Right_Slow_SAT_01': [0.05, 0.05, 0.2, 0.5, 0.2], 
		 'animations/Stand/Exclamation/NAO/Left_Strong_EXC_03': [0.048, 0.048, 0.048, 0.138, 0.718], 
		 'animations/Stand/Gestures/Desperate_5': [0.53, 0.23, 0.08, 0.08, 0.08], 
		 'animations/Stand/Gestures/Desperate_2': [0.2, 0.5, 0.2, 0.05, 0.05],
		 'animations/Stand/Emotions/Positive/Excited_1': [0.048, 0.048, 0.048, 0.138, 0.718], 
		 'animations/Stand/Gestures/You_3': [0.05, 0.2, 0.5, 0.2, 0.05], 
		 'animations/Stand/Emotions/Negative/Disappointed_1': [0.718, 0.138, 0.048, 0.048, 0.048], 
		 'animations/Stand/Negation/NAO/Right_Strong_NEG_01': [0.2, 0.5, 0.2, 0.05, 0.05], 
		 'animations/Stand/Emotions/Negative/Frustrated_1': [0.53, 0.23, 0.08, 0.08, 0.08], 
		 'animations/Stand/Gestures/Explain_3': [0.05, 0.2, 0.5, 0.2, 0.05], 
		 'animations/Stand/BodyTalk/Speaking/BodyTalk_2': [0.1, 0.25, 0.3, 0.25, 0.1], 
		 'animations/Stand/Gestures/Desperate_1': [0.718, 0.138, 0.048, 0.048, 0.048], 
		 'animations/Stand/Emotions/Positive/Sure_1': [0.2, 0.5, 0.2, 0.05, 0.05]}


	def init_gesture_handler(self):
		print("init handler")
		self.memory_service = self.memory.session().service("ALMemory")
		self.sub = self.memory_service.subscriber("ALTextToSpeech/CurrentBookMark")
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
		sentiment_distribution = annotation["sentences"][0]["sentimentDistribution"]
		indx = np.argmax(self.gesture_to_probability[gesture])
		return sentiment_distribution[indx]


	def error_function(self, sentence, gesture):
		annotation = json.loads(self.nlp.annotate(sentence, properties=self.props))
		sentiment_distribution = annotation["sentences"][0]["sentimentDistribution"]
		gesture_distribution = self.gesture_to_probability[gesture]
		squared_errors = [(x - y)**2 for (x, y) in zip(sentiment_distribution, gesture_distribution)]
		return sum(squared_errors)


	def choose_most_app_gesture(self, sentence_index):
		sentence = self.sentences[sentence_index]
		gestures = list(self.gesture_to_probability.keys())
		quality_distribution = [self.error_function(sentence, gesture) for gesture in gestures]
		index = np.argmin(quality_distribution)
		gesture = gestures[index]
		return gesture

	def choose_least_app_gesture(self, sentence_index):
		sentence = self.sentences[sentence_index]
		gestures = list(self.gesture_to_probability.keys())
		quality_distribution = [self.error_function(sentence, gesture) for gesture in gestures]
		index = np.argmax(quality_distribution)
		gesture = gestures[index]
		return gesture


	def choose_best_gesture(self, sentence_index):

		""" TODO: Do this function using the quality.
		sentence = self.sentences[sentence_index]
		annotation = json.loads(self.nlp.annotate(sentence, properties=self.props))
    	if len(annotation["sentences"]) == 0:
     		# print(sentence)
      	return 0.0
    	sentiment_distribution = annotation["sentences"][0]["sentimentDistribution"] """

		gest_to_use = [
		 # 1. Once upon a time, there was a Baby Mouse and Mother Mouse.
	    'animations/Stand/Self & others/NAO/Center_Neutral_SAO_03',
		 # 2. They lived in a hole in the skirting board in a big, warm house with lots of cheese to eat, where they wanted for nothing.
		'animations/Stand/Negation/NAO/Center_Strong_NEG_01',
		 # 3. Then, one day, Mother Mouse decided to take Baby Mouse outside of their home.
		 'animations/Stand/BodyTalk/BodyLanguage/NAO/Center_Slow_AFF_01',
		 # 4. Waiting outside for them was a huge ginger tomcat, licking its lips and waiting to eat them both up. 
		 'animations/Stand/Gestures/Explain_8',
		 # 5, 6. "Mother, Mother! What should we do?"
  		 'animations/Stand/Emotions/Negative/Fearful_1',
		 # 7. Cried Baby Mouse, clinging to his mother's tail.
 		 'animations/Stand/Negation/NAO/Center_Strong_NEG_05',
		 # 8. Mother Mouse paused, staring up into the beady eyes of the hungry cat.
		 'animations/Stand/Emotions/Positive/Confident_1',
		 # 9. But she wasn't scared because she knew exactly how to deal with big, scary cats.
		'animations/Stand/Emotions/Positive/Proud_2',
		 # 10. She opened her mouth and took in a deep breath.
		 'animations/Stand/Emotions/Positive/Confident_1',
		 # 11, 12 "Woof! Woof! Bark bark bark!"
 		 'animations/Stand/Exclamation/NAO/Left_Strong_EXC_03',
		 # 13. She shouted, and the cat ran away as fast as he could.
  		 'animations/Stand/Emotions/Positive/Happy_4',
		 # 14, 15 "Wow, Mother! That was amazing!" 
 		 'animations/Stand/Emotions/Positive/Confident_1',
		 # 16. Baby Mouse said to his mother, smiling happily. 
		 'animations/Stand/BodyTalk/Speaking/BodyTalk_11',
		 # 17,18 "And that, my child, is why it is always best to have a second language. 
		 'animations/Stand/Gestures/Explain_9',
		 # 19, 20 Moral: It's always good to have a second language.
		 'animations/Stand/Gestures/Explain_1',
		 # 21 End gesture.
		 'animations/Stand/Gestures/Explain_4',
	  ]
		return gest_to_use[sentence_index - 1]


	def choose_gesture(self, sentence_index):
		gesture_index = randint(0, len(self.gesture_to_probability) - 1)
		gesture = self.gesture_to_probability.keys()[gesture_index]
		return gesture

	def choose_policy_gesture(self, sentence_index):
		# TODO move to utils
		sentence = self.sentences[sentence_index]
		state = self.get_corenlp_sentiment(sentence)
		state = tuple([round(x, 1) for x in state])
		return self.policy[state]


	def bookmark_handler(self, value):
		print("Value = ", value)
		#gesture = self.choose_gesture()
		#gesture = self.choose_best_gesture(int(value))
		if int(value) == 0:
			return
		index = int(value) - 1
		gesture = self.choose_policy_gesture(index)
		self.anim.run(gesture)
		quality = self.quality_function(self.sentences[index], gesture)
		self.quality_sum += quality


	def main(self):
		self.init_robot_connection()
		self.init_story()
		self.init_policy()

		self.quality_sum = 0.0
		memory_service = self.memory.session().service("ALMemory")
		self.sub = memory_service.subscriber("ALTextToSpeech/CurrentBookMark")
		self.sub.signal.connect(self.bookmark_handler)
		
		self.anim.run("animations/Stand/BodyTalk/BodyLanguage/NAO/Center_Slow_AFF_03")
		self.tts.say(str(self.story["title"]))
		self.tts.say(str(self.story["story"]))
		self.anim.run("animations/Stand/BodyTalk/BodyLanguage/NAO/Center_Slow_AFF_03")
		self.tts.say("Moral of the story: " + str(self.story["moral"]))
		
		self.quality_sum = self.quality_sum / len(self.sentences)
		print(self.quality_sum)


	def simulate(self, choose_gesture_function):

		self.error_sum = 0
		for i in range(len(self.sentences)):
			sentence = self.sentences[i]
			gesture = choose_gesture_function(i)
			error = self.error_function(sentence, gesture)
			self.error_sum += error

		self.error_sum = self.error_sum / (len(self.sentences))
		return self.error_sum

	def try_leds(self):
		self.init_robot_connection()
		# up
		self.leds.fadeRGB("FaceLed0", 1.0, 0.0, 0.0, 2)
		self.leds.fadeRGB("FaceLed1", 0.0, 1.0, 0.0, 2)
		# inner corner
		self.leds.fadeRGB("FaceLed2", 0.0, 0.0, 1.0, 2)
		self.leds.fadeRGB("FaceLed3", 1.0, 1.0, 0.0, 2)
		# down
		self.leds.fadeRGB("FaceLed4", 1.0, 0.0, 1.0, 2)
		self.leds.fadeRGB("FaceLed5", 0.0, 1.0, 1.0, 2)
		# outer corner
		self.leds.fadeRGB("FaceLed6", 1.0, 1.0, 1.0, 2)
		self.leds.fadeRGB("FaceLed7", 0.0, 0.0, 0.0, 2)
		#print(self.leds.listGroup("FaceLeds"))
		print(self.leds.listLEDs())



def run_simulations(story_teller):
	num_simulations = 50
	simulation_range = range(num_simulations)
	simulations = [story_teller.simulate(story_teller.choose_gesture) for _ in simulation_range]
	hardcoded = story_teller.simulate(story_teller.choose_best_gesture)
	best = story_teller.simulate(story_teller.choose_most_app_gesture)
	worst = story_teller.simulate(story_teller.choose_least_app_gesture)
	print(simulations)
	
	plt.ylim((0.0, 1.0))
	plt.plot(simulation_range, simulations, color="blue", label="Random choice")
	plt.plot(simulation_range, [hardcoded]*num_simulations, color="green", label="Hardcoded")
	plt.plot(simulation_range, [best]*num_simulations, color="red")
	plt.plot(simulation_range, [worst]*num_simulations, color="pink")
	plt.xlabel("Time")
	plt.ylabel("Loss")
	plt.legend()
	plt.show()
	plt.savefig('loss_over_time.png')



if __name__ == "__main__":
	story_teller = StoryTeller()
	#story_teller.main()
	#run_simulations(story_teller)
	#story_teller.try_leds()
	

