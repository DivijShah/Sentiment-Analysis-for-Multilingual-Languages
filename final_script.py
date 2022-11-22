'''
This script implements all the modules that are required for emotion analysis. 
It contains modules that: 
1. prepares the dataset
2. trains a classifier
3. tests the classifier
4. can evaluate a audio signal
'''
from __future__ import division
import feature_extractor
from os import listdir
from os.path import isfile, join
import csv
from random import shuffle
import cPickle
import csv
import os
import scipy.io.wavfile as wav
import math
import numpy as np

class EmotionAnalysis():
	def __init__(self):
		self.extractor = feature_extractor.FeatureExtractor()

	def write_to_csv(self, inp, label, name):
		'''
		Writes the prepared dataset to a csv file.

		Parameter description -
		inp 	:	A list of feature vector
		label 	:	A list of labels for each feature vector
		name  	:	csv file-name 
		'''
		csvfile = open(name, 'wb')
		writer = csv.writer(csvfile, delimiter=',')
		for i in range(len(inp)):
			row = inp[i]+[label[i]]
			writer.writerow(row) 
			csvfile.flush()
		csvfile.close()

	def prepare_dataset(self, path):
		'''
		This method prepares the dataset from the audio files present in the provided path.

		Parameter description -
		path	:	Path where all the input audio files are stored

		Returns -	[dataset_inputs, dataset_labels]
		dataset_inputs	:	A list of feature vectors, where each vector is for one complete audio file.
		dataset_labels	:	A list of labels, where each label is for corresponding feature vector in dataset_inputs.
		'''
		i=0
		dataset_inputs = []
		dataset_labels = []
		audio_files = [f for f in listdir(path) if isfile(join(path, f))]
		for audio_file in audio_files:
			try:
				audio_features = self.extractor.extract_features(join(path, audio_file))
				i+=1
				if i%100==0:
					print ("Loaded audio files, count: "+str(i))
			except:
				print ("Skipping audio file '"+join(path, audio_file)+"'")
				continue
			audio_label = audio_file.split('_')[0]
			dataset_inputs.append(audio_features)
			dataset_labels.append(audio_label)
		return [dataset_inputs, dataset_labels]

	def prepare_dataset_per_frame(self, path):
		'''
		-- For future use --
		This method prepares the dataset 'per frame' from the audio files present in the provided path.

		Parameter description -
		path	:	Path where all the input audio files are stored

		Returns -	[dataset_inputs, dataset_labels]
		dataset_inputs	:	A list of feature vectors, where each vector is for one frame in a complete audio file.
		dataset_labels	:	A list of labels, where each label is for corresponding feature vector in dataset_inputs.
		'''
		from random import shuffle
		i=0
		dataset_inputs = []
		dataset_labels = []
		audio_files = [f for f in listdir(path) if isfile(join(path, f))]
		shuffle(audio_files)
		for audio_file in audio_files:
			try:
				audio_features = self.extractor.extract_features_per_frame(join(path, audio_file))
				i+=1
				if i%100==0:
					print ("Loaded audio files, count: "+str(i))
			except:
				print ("Skipping audio file '"+join(path, audio_file)+"'")
				continue
			audio_label = audio_file.split('_')[0]	
			dataset_inputs.extend(audio_features)
			dataset_labels.extend([audio_label for _ in range(len(audio_features))])
		return [dataset_inputs, dataset_labels]


	def train_classifier(self, dataset):
		'''
		This is the function which trains a classifier based on the given dataset and returns the trained classifier. 
		Currently using a Random Forest classifier, which was giving better accuracy on the test data.

		TO DO: Use of better performing classifiers

		Parameter description -
		dataset 	:	A list of feature vector list and labels list.

		Returns -
		clf 		:	A trained classifier that can be used for predictions. 	 

		'''
		from sklearn.ensemble import RandomForestClassifier
		inp, lab = dataset
		dataset = zip(inp, lab)
		shuffle(dataset)
		inp, lab = zip(*dataset)	
		clf = RandomForestClassifier(n_estimators=1000, max_depth = 100000, n_jobs = -1, oob_score = True, min_samples_split = 10)
		clf.fit(inp, lab)
		return clf

	def test_classifier(self, classifier, dataset):
		'''
		This is the function which tests the trained classifier on the test dataset and returns the precision, recall and f1 score.

		Parameter description -
		classifier	:	A trained classifier that can be used for testing and predictions. 	
		dataset 	:	A list of feature vector list and labels list.

		Returns - [precision, recall, f1_score]
		precision 	:	A list of precision values for each class label
		recall 		:	A list of recall values for each class label
		f1_score 	:	A list of f1_score values for each class label
		'''
		from sklearn.metrics import precision_recall_fscore_support
		y_pred = classifier.predict(dataset[0])
		y_true = dataset[1]
		precision, recall, f1_score, _ = precision_recall_fscore_support(y_true, y_pred)
		return [precision, recall, f1_score]

	def evaluate(self, classifier, audio_file = None, audio_signal = None):
		'''
		This function predicts the emotion in the audio signal input using the trained classifier. 

		Parameter description -
		classifier	:	A trained classifier that can be used for testing and predictions.
		audio_file 	:	An audio file path to analyse
		audio_signal: 	An audio signal to analyse

		Returns - 
		audio_label[0]	:	A label <positve | neutral | negative> for the input audio sample. 
		'''
		if audio_file is not None:
			audio_features = self.extractor.extract_features(audio_file = audio_file)
		else:
			audio_features = self.extractor.extract_features(audio_signal = audio_signal)
		audio_label = classifier.predict([audio_features])
		return audio_label[0]
'''
<Execution script>
This script is the main script that actually returns the analysed emotions present in a set of audio samples.
All the files to be analysed are kept in a folder named 'test_calls'.
The classifer is loaded/trained, each call is analysed and three things are evaluated for each call:
1. Overall call result
2. Transition from first half to second half
3. Emotions in each 20 seconds segments of the call.
Finally, all the results are entered into a csv file for further analysis/visualization.
'''

emo = emotion_analysis.EmotionAnalysis()

csvfile = open('call_analysis.csv', 'wb')
writer = csv.writer(csvfile, delimiter=',')

#Either load a pre-trained existing classifier or train a new classifier.
if(os.path.exists('my_dumped_classifier.pkl')):
	print ("Loading existing classifier...")
	with open('my_dumped_classifier.pkl', 'rb') as fid:
	    clf = cPickle.load(fid)	
#print ("Loaded...")
else:
	print ("Building a classifier...")
	print ("Loading data...")
	inp, lab = emo.prepare_dataset(os.path.join(os.path.dirname(os.path.abspath(__file__)),'data'))
	print ("Training on the data...")
	clf = emo.train_classifier([inp, lab])
	print ("\nClassifier score: "+str(clf.oob_score_)+"\n")
	# save the classifier
	with open('my_dumped_classifier.pkl', 'wb') as fid:
	    cPickle.dump(clf, fid)    

row = ['Name', 'Length of call', 'Overall call result', 'Transition: First half', 'Transition: Second half']
writer.writerow(row)

#Read each audio call present in the folde, perform the analysis and store in the csv file. 
path = 'test_calls'
audio_files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
counter = 0
for audio_file in audio_files:
	row = []
	counter += 1
	print ("Processing file: "+str(counter))
	try:
		rate_sig, sig = wav.read(os.path.join(path, audio_file))
		complete_call_result = emo.evaluate(clf, audio_signal = (rate_sig, sig))
		first_half_result = emo.evaluate(clf, audio_signal = (rate_sig, sig[:int(len(sig)/2)]))
		second_half_result = emo.evaluate(clf, audio_signal = (rate_sig, sig[int(len(sig)/2):]))
		segment_result = []
		for i in range(int(math.ceil(len(sig)/(rate_sig*20)))):
		    segment_result.append(emo.evaluate(clf, audio_signal = (rate_sig, sig[i*rate_sig*20:i*rate_sig*20+rate_sig*20])))
		row.append(audio_file)
		row.append(str(int((len(sig)/rate_sig)/60))+" min "+str(int((len(sig)/rate_sig)%60))+" sec")
		row.append(complete_call_result)
		row.append(first_half_result)
		row.append(second_half_result)
		row.extend(segment_result)
		writer.writerow(row)
		csvfile.flush()
	except:
		print ("Skipping file: "+str(counter)+", name: "+audio_file)
csvfile.close()
print ("Completed...")