from main import StoryTeller 

from naoqi import ALProxy
import naoqi 
import numpy as np


TTSAPI = "ALTextToSpeech"
ANIMTTS = "ALAnimatedSpeech"
ALMOOD = "ALMood" # Reads instantaneous emotion of persons and ambiance.
ALANIM = "ALAnimationPlayer"

class LabellingScript:
	def get_session(self, name):
		return ALProxy(name, self.ip, self.port)

	def __init__(self):
		self.init_gestures()

	def init_robot_connection(self):
		self.ip = "169.254.223.63"
		self.port = 9559
		self.tts = self.get_session(TTSAPI)
		self.memory = self.get_session("ALMemory")
		self.anim = self.get_session("ALAnimationPlayer")


	def init_gestures(self):
		self.gestures_list = ['animations/Stand/BodyTalk/Speaking/BodyTalk_11','animations/Stand/Negation/NAO/Right_Strong_NEG_01', 'animations/Stand/BodyTalk/Speaking/BodyTalk_14', 'animations/Stand/BodyTalk/Speaking/BodyTalk_17', 'animations/Stand/BodyTalk/Speaking/BodyTalk_18', 'animations/Stand/BodyTalk/Speaking/BodyTalk_19', 'animations/Stand/BodyTalk/Speaking/BodyTalk_2', 'animations/Stand/BodyTalk/Speaking/BodyTalk_4', 'animations/Stand/BodyTalk/Speaking/BodyTalk_5', 'animations/Stand/BodyTalk/Speaking/BodyTalk_6', 'animations/Stand/Emotions/Negative/Disappointed_1', 'animations/Stand/Emotions/Negative/Fearful_1', 'animations/Stand/Emotions/Negative/Frustrated_1', 'animations/Stand/Emotions/Neutral/AskForAttention_2', 'animations/Stand/Emotions/Positive/Confident_1', 'animations/Stand/Emotions/Positive/Excited_1', 'animations/Stand/Emotions/Positive/Happy_1', 'animations/Stand/Emotions/Positive/Happy_4', 'animations/Stand/Emotions/Positive/Proud_2', 'animations/Stand/Emotions/Positive/Sure_1', 'animations/Stand/Gestures/Desperate_1', 'animations/Stand/Gestures/Desperate_2', 'animations/Stand/Gestures/Desperate_5', 'animations/Stand/Gestures/Enthusiastic_1', 'animations/Stand/Gestures/Everything_1', 'animations/Stand/Gestures/Explain_1', 'animations/Stand/Gestures/Explain_10', 'animations/Stand/Gestures/Explain_3', 'animations/Stand/Gestures/Explain_4', 'animations/Stand/Gestures/Explain_5', 'animations/Stand/Gestures/Explain_6','animations/Stand/Gestures/Explain_8', 'animations/Stand/Gestures/Explain_9', 'animations/Stand/Gestures/Yes_3', 'animations/Stand/Gestures/You_3', 'animations/Stand/Self & others/NAO/Left_Neutral_SAO_01', 'animations/Stand/Self & others/NAO/Left_Neutral_SAO_06', 'animations/Stand/Self & others/NAO/Right_Neutral_SAO_02', 'animations/Stand/Self & others/NAO/Left_Neutral_SAO_02', 'animations/Stand/Self & others/NAO/Left_Strong_SAO_02', 'animations/Stand/Self & others/NAO/Center_Neutral_SAO_03', 'animations/Stand/Negation/NAO/Left_Strong_NEG_01', 'animations/Stand/Negation/NAO/Left_Strong_NEG_03', 'animations/Stand/Negation/NAO/Right_Strong_NEG_04', 'animations/Stand/Negation/NAO/Center_Neutral_NEG_02', 'animations/Stand/Negation/NAO/Center_Strong_NEG_01', 'animations/Stand/Negation/NAO/Center_Strong_NEG_05', 'animations/Stand/Exclamation/NAO/Left_Strong_EXC_03', 'animations/Stand/Exclamation/NAO/Center_Neutral_EXC_01', 'animations/Stand/Exclamation/NAO/Right_Strong_EXC_04', 'animations/Stand/BodyTalk/BodyLanguage/NAO/Center_Neutral_AFF_07', 'animations/Stand/BodyTalk/BodyLanguage/NAO/Center_Strong_AFF_06', 'animations/Stand/BodyTalk/BodyLanguage/NAO/Center_Slow_AFF_01', 'animations/Stand/BodyTalk/BodyLanguage/NAO/Center_Strong_AFF_05', 'animations/Stand/BodyTalk/BodyLanguage/NAO/Center_Neutral_AFF_01', 'animations/Stand/BodyTalk/BodyLanguage/NAO/Right_Slow_AFF_01', 'animations/Stand/BodyTalk/BodyLanguage/NAO/Center_Strong_AFF_01', 'animations/Stand/BodyTalk/BodyLanguage/NAO/Center_Slow_AFF_03', 'animations/Stand/Space & time/NAO/Left_Neutral_SAT_06', 'animations/Stand/Space & time/NAO/Left_Neutral_SAT_01', 'animations/Stand/Space & time/NAO/Right_Slow_SAT_01', 'animations/Stand/Space & time/NAO/Right_Strong_SAT_03', 'animations/Stand/Space & time/NAO/Left_Neutral_SAT_02']
		self.gestures_dict = {'0': ['animations/Stand/BodyTalk/Speaking/BodyTalk_11', 'animations/Stand/BodyTalk/Speaking/BodyTalk_17', 'animations/Stand/BodyTalk/Speaking/BodyTalk_18', 'animations/Stand/BodyTalk/Speaking/BodyTalk_4', 'animations/Stand/BodyTalk/Speaking/BodyTalk_5', 'animations/Stand/Gestures/Explain_1', 'animations/Stand/Gestures/Explain_4', 'animations/Stand/Gestures/Explain_8', 'animations/Stand/Gestures/Explain_9', 'animations/Stand/Self & others/NAO/Left_Neutral_SAO_01', 'animations/Stand/Self & others/NAO/Left_Neutral_SAO_02', 'animations/Stand/Self & others/NAO/Left_Strong_SAO_02', 'animations/Stand/Negation/NAO/Center_Strong_NEG_01', 'animations/Stand/Exclamation/NAO/Center_Neutral_EXC_01', 'animations/Stand/BodyTalk/BodyLanguage/NAO/Center_Slow_AFF_01', 'animations/Stand/BodyTalk/BodyLanguage/NAO/Center_Strong_AFF_01'], 
							  '+': ['animations/Stand/Emotions/Positive/Confident_1', 'animations/Stand/Emotions/Positive/Happy_4', 'animations/Stand/Emotions/Positive/Proud_2', 'animations/Stand/Self & others/NAO/Center_Neutral_SAO_03', 'animations/Stand/Exclamation/NAO/Left_Strong_EXC_03', 'animations/Stand/Exclamation/NAO/Right_Strong_EXC_04', 'animations/Stand/BodyTalk/BodyLanguage/NAO/Center_Slow_AFF_03', 'animations/Stand/Space & time/NAO/Right_Slow_SAT_01'], 
							  '-': ['animations/Stand/Emotions/Negative/Fearful_1', 'animations/Stand/Gestures/Desperate_2', 'animations/Stand/Gestures/Desperate_5', 'animations/Stand/Negation/NAO/Center_Strong_NEG_05', 'animations/Stand/BodyTalk/BodyLanguage/NAO/Center_Strong_AFF_05']}
		#self.gesture_to_valence = {'animations/Stand/Emotions/Negative/Fearful_1': 0, 'animations/Stand/BodyTalk/Speaking/BodyTalk_18': 1, 'animations/Stand/BodyTalk/BodyLanguage/NAO/Center_Slow_AFF_01': 2, 'animations/Stand/BodyTalk/BodyLanguage/NAO/Center_Slow_AFF_03': 3, 'animations/Stand/BodyTalk/Speaking/BodyTalk_11': 2, 'animations/Stand/BodyTalk/Speaking/BodyTalk_17': 2, 'animations/Stand/Emotions/Positive/Proud_2': 4, 'animations/Stand/Emotions/Positive/Confident_1': 4, 'animations/Stand/Exclamation/NAO/Center_Neutral_EXC_01': 3, 'animations/Stand/Exclamation/NAO/Right_Strong_EXC_04': 4, 'animations/Stand/BodyTalk/Speaking/BodyTalk_5': 3, 'animations/Stand/BodyTalk/Speaking/BodyTalk_4': 2, 'animations/Stand/BodyTalk/BodyLanguage/NAO/Center_Strong_AFF_05': 0, 'animations/Stand/Self & others/NAO/Center_Neutral_SAO_03': 2, 'animations/Stand/Gestures/Explain_4': 2, 'animations/Stand/Gestures/Explain_1': 2, 'animations/Stand/Emotions/Positive/Happy_4': 4, 'animations/Stand/BodyTalk/BodyLanguage/NAO/Center_Strong_AFF_01': 2, 'animations/Stand/Gestures/Explain_9': 2, 'animations/Stand/Gestures/Explain_8': 2, 'animations/Stand/Negation/NAO/Center_Strong_NEG_01': 4, 'animations/Stand/Negation/NAO/Center_Strong_NEG_05': 0, 'animations/Stand/Self & others/NAO/Left_Neutral_SAO_01': 2, 'animations/Stand/Self & others/NAO/Left_Neutral_SAO_02': 3, 'animations/Stand/Self & others/NAO/Left_Strong_SAO_02': 3, 'animations/Stand/Space & time/NAO/Right_Slow_SAT_01': 3, 'animations/Stand/Exclamation/NAO/Left_Strong_EXC_03': 4, 'animations/Stand/Gestures/Desperate_5': 0, 'animations/Stand/Gestures/Desperate_2': 1}
		#self.gesture_to_confidence = {'animations/Stand/Emotions/Negative/Fearful_1': 3, 'animations/Stand/BodyTalk/Speaking/BodyTalk_18': 1, 'animations/Stand/BodyTalk/BodyLanguage/NAO/Center_Slow_AFF_01': 2, 'animations/Stand/BodyTalk/BodyLanguage/NAO/Center_Slow_AFF_03': 2, 'animations/Stand/BodyTalk/Speaking/BodyTalk_11': 2, 'animations/Stand/BodyTalk/Speaking/BodyTalk_17': 3, 'animations/Stand/Emotions/Positive/Proud_2': 3, 'animations/Stand/Emotions/Positive/Confident_1': 2, 'animations/Stand/Exclamation/NAO/Center_Neutral_EXC_01': 1, 'animations/Stand/Exclamation/NAO/Right_Strong_EXC_04': 3, 'animations/Stand/BodyTalk/Speaking/BodyTalk_5': 2, 'animations/Stand/BodyTalk/Speaking/BodyTalk_4': 3, 'animations/Stand/BodyTalk/BodyLanguage/NAO/Center_Strong_AFF_05': 2, 'animations/Stand/Self & others/NAO/Center_Neutral_SAO_03': 1, 'animations/Stand/Gestures/Explain_4': 1, 'animations/Stand/Gestures/Explain_1': 1, 'animations/Stand/Emotions/Positive/Happy_4': 3, 'animations/Stand/BodyTalk/BodyLanguage/NAO/Center_Strong_AFF_01': 2, 'animations/Stand/Gestures/Explain_9': 2, 'animations/Stand/Gestures/Explain_8': 2, 'animations/Stand/Negation/NAO/Center_Strong_NEG_01': 2, 'animations/Stand/Negation/NAO/Center_Strong_NEG_05': 2, 'animations/Stand/Self & others/NAO/Left_Neutral_SAO_01': 2, 'animations/Stand/Self & others/NAO/Left_Neutral_SAO_02': 2, 'animations/Stand/Self & others/NAO/Left_Strong_SAO_02': 2, 'animations/Stand/Space & time/NAO/Right_Slow_SAT_01': 2, 'animations/Stand/Exclamation/NAO/Left_Strong_EXC_03': 3, 'animations/Stand/Gestures/Desperate_5': 2, 'animations/Stand/Gestures/Desperate_2': 2}
		self.gesture_to_valence = {'animations/Stand/Gestures/Explain_3': 2, 'animations/Stand/Gestures/You_3': 2, 'animations/Stand/BodyTalk/Speaking/BodyTalk_19': 0, 'animations/Stand/Emotions/Negative/Disappointed_1': 0, 'animations/Stand/Negation/NAO/Right_Strong_NEG_01': 1, 'animations/Stand/Emotions/Negative/Frustrated_1': 0, 'animations/Stand/Emotions/Positive/Excited_1': 4, 'animations/Stand/BodyTalk/Speaking/BodyTalk_2': 2, 'animations/Stand/Gestures/Desperate_1': 0, 'animations/Stand/Emotions/Positive/Sure_1': 1}
		self.gesture_to_confidence = {'animations/Stand/Gestures/Explain_3': 2, 'animations/Stand/Gestures/You_3': 2, 'animations/Stand/BodyTalk/Speaking/BodyTalk_19': 0, 'animations/Stand/Emotions/Negative/Disappointed_1': 3, 'animations/Stand/Negation/NAO/Right_Strong_NEG_01': 2, 'animations/Stand/Emotions/Negative/Frustrated_1': 2, 'animations/Stand/Emotions/Positive/Excited_1': 3, 'animations/Stand/BodyTalk/Speaking/BodyTalk_2': 1, 'animations/Stand/Gestures/Desperate_1': 3, 'animations/Stand/Emotions/Positive/Sure_1': 2}

		story_teller = StoryTeller()
		self.gesture_to_sentiment = story_teller.gesture_to_sentiment
		

	def bookmark_handler(self, value):
		self.anim.run('animations/Stand/Emotions/Negative/Angry_1')


	def main(self):
		self.init_robot_connection()
		memory_service = self.memory.session().service("ALMemory")
		sub = memory_service.subscriber("ALTextToSpeech/CurrentBookMark")
		gest = {"+":[],"0":[],"-":[]}
		
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
	

	def label_gestures(self):
		self.init_robot_connection()

		print(len(self.gestures_list))

		self.gesture_to_confidence = {}
		self.new_gesture_to_sentiment = {}
		old_keys = self.gesture_to_sentiment.keys()
		all_keys = self.gestures_list
		
		for gesture in all_keys:
			if gesture in old_keys:
				continue
			print(gesture)

			result = "r"		
			while result == "r":
				self.anim.run(gesture)
				result = raw_input()
			if result == 'q':
				break
			if result == 'n':
				continue
			print("Please give the new value")
			result = raw_input()
			self.new_gesture_to_sentiment[gesture] = int(result)
			print("Please give the confidence")
			result = raw_input()
			self.gesture_to_confidence[gesture] = int(result)

		print("Gesture confidence")
		print(self.gesture_to_confidence)
		print("Gesture valence")
		print(self.new_gesture_to_sentiment)



	def process_labels(self):
		confidence_to_probability = {1: [0.3, 0.25, 0.1],
									 2: [0.5, 0.2, 0.05],
									 3: [0.7, 0.12, 0.03]}
		gesture_to_probability = {}
		for gesture in self.gesture_to_confidence:
			if self.gesture_to_confidence[gesture] == 0:
				continue
			confidence = self.gesture_to_confidence[gesture]
			valence = self.gesture_to_valence[gesture]
			variance = confidence_to_probability[confidence]
			distribution = [variance[2]] * 5
			distribution[valence] = variance[0]
			
			if valence > 0:
				distribution[valence - 1] = variance[1]
			if valence < 4:
				distribution[valence + 1] = variance[1]
			if (1.0 - sum(distribution)) > 0.00001:
				aux = (1.0 - sum(distribution)) / 5.0
				distribution = [round(x + aux, 3) for x in distribution]
			#print(valence, confidence, distribution)
			gesture_to_probability[gesture] = distribution
		print(gesture_to_probability)
		print(len(gesture_to_probability))

if __name__ == "__main__":
	labelling_script = LabellingScript()
	#labelling_script.label_gestures()
	labelling_script.process_labels()
