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
		self.start_learning_rate = 0.01
		self.end_learning_rate = 0.002
		self.decay = 0.9
		self.start_epsilon = 0
		self.end_epsilon = 1.0
		self.num_epochs = 5
		self.init_dataset()
		self.init_sentence_to_state()
		self.state_action_table = {}
		self.test_while_training = True
		print("Learning rate {}", self.start_learning_rate)
		print("Decay {}", self.decay)
		print("Epsilon {}", self.start_epsilon)
		print("Num epoch {}", self.num_epochs)


	def init_dataset(self):
		with open("aesopFables.json") as file:
			self.dataset = json.load(file)["stories"]
		num_testing_stories = int((20.0 / 100.0) * len(self.dataset))
		# Take random 20% os the stories and delete of the duplicates.
		self.testing_set = set([random.randint(0, len(self.dataset) - 1) for _ in range(num_testing_stories)])
		self.testing_set = [self.dataset[x] for x in self.testing_set]

		# All the stories that are not in the testing set.
		self.training_set = [x for x in self.dataset if x not in self.testing_set]
		

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
		annotation = json.loads(decmark_reg.sub('.',self.story_teller.nlp.annotate(sentence, \
                                                  properties=self.story_teller.props)))
		sentiment_distribution = annotation["sentences"][0][
		        "sentimentDistribution"]
		sentiment_distribution = [round(x, 1) for x in sentiment_distribution]
		return tuple(sentiment_distribution)

	def learn_transitions(self):
		self.transition_table = {}

		for story in self.training_set:
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
		test_qualities = []
		test_losses = []
		train_qualities = []
		train_losses = []
		best_qual = 0.0
		best_state_action_table = {}
		learning_rate_step = (self.start_learning_rate -
		                      self.end_learning_rate) / self.num_epochs
		epsilon_rate_step = (self.end_epsilon -
		                      self.start_epsilon) / self.num_epochs
		self.learning_rate = self.start_learning_rate
		self.epsilon = self.start_epsilon
		
		for epoch in range(self.num_epochs):				
			self.learning_rate -= learning_rate_step
			self.epsilon += epsilon_rate_step
			random.shuffle(self.training_set)
			for story in self.training_set:
				test_quality, test_error = self.test("test")
				train_quality, train_error = self.test("train")
				
				test_qualities.append(test_quality)
				test_losses.append(test_error)

				train_qualities.append(train_quality)
				train_losses.append(train_error)
				
				if test_quality > best_qual:
					print("new highscore")
					best_qual = test_quality
					best_state_action_table = self.state_action_table.copy()

				for (i, sentence) in enumerate(story["story"]):
					state = self.sentence_to_state[sentence]
					if state not in self.state_action_table:
						self.state_action_table[state] = {}
					gesture = self.choose_action(state)

					reward = self.reward(state, gesture)
					next_utility = 0.0
					penalty = 0.0
					if i != len(story["story"]) - 1:
						next_state = self.sentence_to_state[story["story"][i +
						                                                   1]]
						next_action = self.best_action(next_state)
						next_utility = self.state_action_table.get(
						        next_state, {}).get(next_action, 0.0)
						if next_action == gesture:
							penalty = -3.0
					current_utility = self.state_action_table[state].get(
					        gesture, 0.0)

					self.state_action_table[state][gesture] = \
                                                        current_utility + \
                                                        self.learning_rate * \
                                                         (reward + self.decay * next_utility + penalty - current_utility)
			epoch += 1

		print("Done training")
		print("----> Best quality: {:4f}".format(best_qual))
		return train_qualities, train_losses, test_qualities, test_losses, best_state_action_table


	def test(self, select_set):
		error = 0.0
		quality = 0.0
		num_sen = 0
		selected_set = None
		if select_set == "train":
			selected_set = self.training_set
		elif select_set == "test":
			selected_set = self.testing_set
		for story in selected_set:
			num_sen += len(story["story"])
			for sentence in story["story"]:
				state = self.sentence_to_state[sentence]
				gesture = self.best_action(state)
				error += self.error(state, gesture)
				quality += self.reward(state, gesture)
		return quality / num_sen, error / num_sen

	def plot_data(self, train_data, test_data, title="Quality over time", mean=False):
		fig, ax = plt.subplots(figsize=(10, 7))
		ax.set_title(title)

		x_data = range(len(train_data))
		ax.plot(x_data, train_data, label="Training set")
		ax.plot(x_data, test_data, label="Testing set")
		for xc in range(0, len(x_data), len(self.training_set)):
			plt.axvline(x=xc)
		plt.xlabel("Stories")
		plt.ylabel("Quality")
		ax.legend(loc='upper right')
		
		axes = fig.gca()
		#axes.set_xlim([None, len(train_data) * 1])
		axes.set_ylim([0, 1])
		


		fig.savefig("{}.png".format(title))

	def main(self):
		total_q_train = []
		total_q_test = []
		for indx in range(1):
			print("-" * 10)
			self.state_action_table = {}
			train_qualities, train_losses, test_qualities, test_losses, best_policy = self.train()
			title = "best_policy_run{}.json".format(indx)
			quality_title = "quality_run{}".format(indx)
			loss_title = "loss_run{}".format(indx)
			with open(title, "w") as file:
				best_policy = {
				        str(state): best_action(state, best_policy)
				        for state in best_policy
				}
				json.dump(best_policy, file)
			self.plot_data(train_qualities, test_qualities, title=quality_title)
			self.plot_data(train_losses, test_losses, title=loss_title)
			total_q_train.append(train_qualities)
			total_q_test.append(test_qualities)
		with open("Quality_Total_Train_json", "w") as file:
			json.dump(total_q_train,file)
		with open("Loss_Total_Test_json", "w") as file:
			json.dump(total_q_test,file)


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
