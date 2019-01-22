import json
import numpy as np
import re

from os import listdir
from os.path import isfile, join
from stanfordcorenlp import StanfordCoreNLP


class SentenceLabellingScript:
	
	def __init__(self):
		self.nlp = StanfordCoreNLP('http://localhost', port=9000, timeout=30000)
		self.props = {
			'annotators': 'sentiment',
			'pipelineLanguage': 'en',
			'outputFormat': 'json'
		}
		with open("stories.txt", "r") as file:
			self.story = file.read()
		self.sentences = re.split("(?:[\\.\"]+\\s*)+", self.story)

		
	def main(self):
		self.extreme_sentences = {i:[] for i in range(5)}
		
		path = "/media/anca/New Volume/School/LM/NaoTheStoryTeller/CBTest/data"
		for file_name in listdir(path):
			file_path = join(path, file_name)
			with open(file_path, "r") as file:
				self.story = file.read()
			self.sentences = re.split("(?:[\\.\"]+\\s*)+", self.story)
			for sentence in self.sentences:
				annotation = json.loads(self.nlp.annotate(sentence, properties=self.props))
				if len(annotation["sentences"]) != 1:
					continue
				sentiment_distribution = annotation["sentences"][0]["sentimentDistribution"]
				if max(sentiment_distribution) >  0.7:
					self.extreme_sentences[np.argmax(sentiment_distribution)] += [sentence]
		
		print(len(self.extreme_sentences[0]))
		print(len(self.extreme_sentences[4]))
		with open("extreme_sentences.txt", "w+") as output_file:
			json_output = json.dumps(self.extreme_sentences)
			output_file.write(json_output)

		for i in range(5):
			self.extreme_sentences[i] = sorted(self.extreme_sentences[i], key=lambda l: len(l))
			self.extreme_sentences[i] = self.extreme_sentences[i][:20]
		print(self.extreme_sentences)


if __name__ == "__main__":
	labelling_script = SentenceLabellingScript()
	labelling_script.main()