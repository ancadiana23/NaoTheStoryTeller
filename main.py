import json
import matplotlib.pyplot as plt
import naoqi 
import numpy as np
import re

from ast import literal_eval as make_tuple
from naoqi import ALProxy
from random import randint, choice
from stanfordcorenlp import StanfordCoreNLP


TTSAPI = "ALTextToSpeech"
ANIMTTS = "ALAnimatedSpeech"
ALMOOD = "ALMood" # Reads instantaneous emotion of persons and ambiance.
ALANIM = "ALAnimationPlayer"


class StoryTeller:
	def get_session(self, name):
	    return ALProxy(name, self.ip, self.port)

	def __init__(self):
		self.init_gestures()
		self.init_sentiment_analysis_client()


	def init_robot_connection(self):
		self.ip = "169.254.177.0"
		self.port = 9559
		self.tts = self.get_session(TTSAPI)
		self.memory = self.get_session("ALMemory")
		self.leds = self.get_session("ALLeds")
		self.anim = self.get_session("ALAnimationPlayer")
	

	def init_stories(self):
		with open("aesopFables.json") as file:
			dataset = json.load(file)["stories"]
		self.story = dataset[20]
		self.sentences = self.story["story"]
		processed_story = [" \\mrk=" + str(i + 2) + "\\ " + str(self.sentences[i]) \
					 for i in range(len(self.sentences))]
		
		self.story["story"] = "\\rspd=85\\" + \
							  "".join(processed_story)


	def init_policy(self):
		with open("best_policy_run0.json") as file:
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

		self.neutral_gestures = [x for x in self.gesture_to_probability \
									if np.argmax(self.gesture_to_probability[x]) == 2]


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


	def choose_gesture(self, _):
		gesture_index = randint(0, len(self.gesture_to_probability) - 1)
		gesture = self.gesture_to_probability.keys()[gesture_index]
		return gesture

	def choose_policy_gesture(self, sentence_index):
		# TODO move to utils
		sentence = self.sentences[sentence_index]
		state = self.get_corenlp_sentiment(sentence)
		state = tuple([round(x, 1) for x in state])
		if state in self.policy:
			return self.policy[state]

		# if the state is not contained in the policy
		# return a random gesture
		print("Random Neutral Gesture")
		return choice(self.neutral_gestures)


	def bookmark_handler(self, value):
		print("Value = ", value)
		#gesture = self.choose_gesture()
		#gesture = self.choose_best_gesture(int(value))
		index = int(value) - 2
		if index == -2:
			return
		elif index == -1:
			gesture = "animations/Stand/BodyTalk/BodyLanguage/NAO/Center_Slow_AFF_03"
		else:
			gesture = self.choose_policy_gesture(index)
		self.anim.run(gesture)
		quality = self.quality_function(self.sentences[index], gesture)
		self.quality_sum += quality


	def main(self):
		self.init_robot_connection()
		self.init_gesture_handler()
		self.init_stories()
		self.init_policy()

		self.quality_sum = 0.0
		memory_service = self.memory.session().service("ALMemory")
		self.sub = memory_service.subscriber("ALTextToSpeech/CurrentBookMark")
		self.sub.signal.connect(self.bookmark_handler)
		
		#self.anim.run("animations/Stand/BodyTalk/BodyLanguage/NAO/Center_Slow_AFF_03")
		self.tts.say(" \\mrk=1\\ "  + str(self.story["title"]))
		self.tts.say(str(self.story["story"]))
		#self.anim.run("animations/Stand/BodyTalk/BodyLanguage/NAO/Center_Slow_AFF_03")
		self.tts.say("Moral of the story: \\mrk=1\\ " + str(self.story["moral"]))
		
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
	story_teller.main()
	#run_simulations(story_teller)
	#story_teller.try_leds()
	

