import json
import numpy as np
import random
import time

import matplotlib.pyplot as plt

import pprint

import re

from collections import Counter
from main import StoryTeller
from multiprocessing.dummy import Pool as ThreadPool


class QLearner:

	def __init__(self, story_teller):
		self.story_teller = story_teller
		self.actions = self.story_teller.gesture_list
		self.start_learning_rate = 0.05
		self.end_learning_rate = 0.005
		self.decay = 0.9
		self.epsilon = 0.8
		self.num_epochs = 10000
		self.init_dataset()
		self.init_sentence_to_state()
		self.state_action_table = {}
		print("Learning rate {}", self.start_learning_rate)
		print("Decay {}", self.decay)
		print("Epsilon {}", self.epsilon)
		print("Num epoch {}", self.num_epochs)

	def init_dataset(self):
		with open("aesopFables.json") as file:
			self.dataset = json.load(file)["stories"]

	def init_sentence_to_state(self):
		self.sentence_to_state = {}
		start = time.time()
		for story in self.dataset:
			pool = ThreadPool(4)
			results = pool.map(self.get_corenlp_sentiment, story["story"])
			pool.close()
			pool.join()
			for i, sentence in enumerate(story["story"]):
				self.sentence_to_state[sentence] = results[i]
				#self.sentence_to_state[sentence] = self.get_corenlp_sentiment(sentence)
		print("TIMEIT - {:2f}".format(time.time() - start))

	def get_corenlp_sentiment(self, sentence):
		decmark_reg = re.compile('(?<=\d),(?=\d)')
		decmark_reg.sub('.',self.story_teller.nlp.annotate(sentence, \
                                      properties=self.story_teller.props))
		annotation = json.loads(decmark_reg.sub('.',self.story_teller.nlp.annotate(sentence, \
                                      properties=self.story_teller.props)))
		sentiment_distribution = annotation["sentences"][0][
		        "sentimentDistribution"]
		sentiment_distribution = [round(x, 1) for x in sentiment_distribution]
		return tuple(sentiment_distribution)

	def learn_transitions(self):
		self.transition_table = {}

		for story in self.dataset:
			previous_sentence = None
			for sentence in story["story"]:
				current_sentence = self.get_corenlp_sentiment(sentence)
				if current_sentence not in self.transition_table:
					self.transition_table[current_sentence] = []
				if previous_sentence is None:
					continue
				self.transition_table[previous_sentence] += [current_sentence]

		for current_sentence in self.transition_table:
			num_next_states = len(self.transition_table[current_sentence])
			counter = Counter(self.transition_table[current_sentence])
			self.transition_table[current_sentence] = \
                                          {sentence: occurences/num_next_states \
                                          for (sentence, occurences) in counter.most_common()}

	def choose_action(self, state):
		if random.uniform(0.0, 1.0) < self.epsilon:
			return self.best_action(state)
		return random.choice(self.story_teller.gesture_list)

	def best_action(self, state):
		if state not in self.state_action_table or \
                           not self.state_action_table[state]:
			return random.choice(self.story_teller.gesture_list)
		gesture = max(self.state_action_table[state].items(), \
                                   key=lambda (x, y): y)
		gesture = gesture[0]
		return gesture

	def error(self, state, gesture):
		gesture_distribution = self.story_teller.gesture_to_probability[gesture]
		squared_errors = [
		        (x - y)**2 for (x, y) in zip(state, gesture_distribution)
		]
		return sum(squared_errors)

	def reward(self, state, gesture):
		gesture_distribution = self.story_teller.gesture_to_probability[gesture]
		sentence_distribution = list(state)
		dot = np.dot(sentence_distribution, gesture_distribution)
		norma = np.linalg.norm(sentence_distribution)
		normb = np.linalg.norm(gesture_distribution)
		cos = dot / (norma * normb)
		return cos

	def train(self):
		start = time.time()
		qualities = []
		losses = []
		best_qual = 0.0
		best_state_action_table = {}
		learning_rate_step = (self.start_learning_rate -
		                      self.end_learning_rate) / self.num_epochs
		self.learning_rate = self.start_learning_rate
		epoch = 0
		while True:

			if epoch % 250 == 0:
				_, error = self.test()
				print("Epoch {} - Loss: {:4f}, time: {:2f}s".format(
				        epoch, error,
				        time.time() - start))
				#print("Iter %r: loss=%.4f, time=%.2fs" % (epoch, error, time.time()-start))
			if epoch % 500 == 0:
				quality, error = self.test()
				qualities.append(quality)
				losses.append(error)
				print("----> Epoch {}: quality: {:4f} - Loss: {:4f}".format(
				        epoch, quality, error))
				# save best model parameters
				if quality > best_qual:
					print("new highscore")
					best_qual = quality
					best_state_action_table = self.state_action_table.copy()
			if epoch == self.num_epochs:
				print("Done training")
				print("----> Best quality: {:4f}".format(best_qual))
				return qualities, losses, best_state_action_table

			self.learning_rate -= learning_rate_step
			random.shuffle(self.dataset)
			for story in self.dataset:
				for (i, sentence) in enumerate(story["story"][:-1]):
					state = self.sentence_to_state[sentence]
					if state not in self.state_action_table:
						self.state_action_table[state] = {}
					gesture = self.choose_action(state)

					reward = self.reward(state, gesture)
					next_state = self.sentence_to_state[story["story"][i + 1]]
					next_action = self.best_action(next_state)
					next_utility = self.state_action_table.get(
					        next_state, {}).get(next_action, 0.0)
					current_utility = self.state_action_table[state].get(
					        gesture, 0.0)
					self.state_action_table[state][gesture] = \
                                                                  current_utility + \
                                                                  self.learning_rate * \
                                                                   (reward + self.decay * next_utility - current_utility)
			epoch+=1

		#return best_state_action_table

	def test(self):
		error = 0.0
		quality = 0.0
		num_sen = 0
		for story in self.dataset:
			num_sen += len(story["story"])
			for sentence in story["story"]:
				state = self.sentence_to_state[sentence]
				gesture = self.best_action(state)
				error += self.error(state, gesture)
				quality += self.reward(state, gesture)
		return quality / num_sen, error

	def plot_data(self,data,title = "",mean=False):
		#if mean:
		#	data.append(np.mean(data,axis=0))
		fig, ax = plt.subplots(figsize=(10, 7))
		ax.set_title(title)
		#data.append(np.mean(data, axis = 0))
		#for i in range(len(data)):
		#label = "run #{}".format(i+1)
		#if mean and i == len(data):
		#	label = ("mean run")
		x = [x*500 for x in range(len(data))]
		y = data
		#ax.plot(x,y,label=label)
		ax.plot(x,y)
		#ax.legend(loc='upper right')
		axes = fig.gca()
		axes.set_xlim([None,len(data)*500])
		axes.set_ylim([0,None])
		fig.tight_layout()
		#fig.show()
		fig.savefig("{}.png".format(title))

	def main(self):
		for _ in range(1):
			print("-" * 10)
			self.state_action_table = {}
			quality, error = self.test()
			#print("Error ", error)
			print("----> Quality: {:4f} - Loss: {:4f}".format(quality, error))
			q,l, best_policy = self.train()
			with open("best_policy.json", "w") as file:
				best_policy = {str(state): best_action(state, best_policy) for state in best_policy}
				json.dump(best_policy,file)
			#print( len(q))
			#print(len(l))
			#sum_actions = sum([
			#        len(self.state_action_table[x])
			#        for x in self.state_action_table
			#])
			#print("Actions ", sum_actions, "; States: ",
			#      len(self.state_action_table.keys()))
			#print(sum_actions / len(self.state_action_table.keys()))
			#quality, error = self.test()
			#print("Error ", error)
			#print("----> Quality: {:4f} - Loss: {:4f}".format(quality, error))
			#self.plot_data(list(q),mean=True)
			#self.plot_data(list(l),mean=True)
			self.plot_data(q,title="Quality")
			self.plot_data(l,title="Loss")

def best_action(state, state_action_table):
	# move to utils
	gesture = max(state_action_table[state].items(), \
                               key=lambda (x, y): y)
	gesture = gesture[0]
	return gesture

if __name__ == "__main__":
	story_teller = StoryTeller()
	story_teller.init_gesture_list()
	learner = QLearner(story_teller)
	learner.main()
