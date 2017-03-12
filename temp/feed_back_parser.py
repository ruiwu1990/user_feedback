import pandas as pd
import math
import subprocess
import sys
import csv
import os
from collections import defaultdict
import json


def feedback_processing(input_file,output_file):
	'''
	This function parse the input files and
	output the cleaned data without unnecessary
	words, and return possible words in the samples
	'''
	fp_input = open(input_file,'r')
	fp_output = open(output_file,'w')

	# replace replace_pair[0] with replace_pair[1]
	replace_pair = [[['isn\'t','don\'t','aren\'t','cannot','can\'t','cant','dont','arent','isnt'],'not'],
					[['too','really','super','extremely'],'very'],
					[['long'],'slow'],
					[['terrible'],'very slow'],
					[['ages'],'very slow'],
					[['satisfactory','decent','ok','satisfied','happy'],'good'],
					[['excellent','superb','outstanding','fantastic','awesome','great'],'very good'],
					[['quick'],'fast']]

	unnecessary_words = ['we','i','you','my','should',
						'its','it\'s','im','i\'m','youre','you\'r',
						'like','would','may','our','your','system','thing'
	                    'maybe','to','server','service','queue','is','everything',
	                    'are','am','the','be','maight','can']
	
	# this arr records possible words in the sentence
	possible_words = []

	# skip the csv title
	title_line = fp_input.readline()
	fp_output.write(title_line)
	for line in fp_input:
		temp_line = ''
		# step 1: lower case all the words
		temp_line = line.lower()
		temp_sentences = temp_line.split(',')
		final_line_arr = []

		for sentence in temp_sentences:
			# assumption 1: question sentence means disappointed
			if '?' in sentence:
				final_line_arr.append('slow')
			else:	
				final_sentence_arr = []
				# remove period and collect words in the sentence
				words = sentence.replace('.','').split()

				for word in words:
					# step 2: replace synonyms in the sentences
					# replace word with "not"
					for word_pair in replace_pair:
						if word in word_pair[0]:
							final_sentence_arr.append(word_pair[1])
					
					# step 3: remove unnecessary words
					if word not in unnecessary_words:
						final_sentence_arr.append(word)

				final_line_arr.append(' '.join([str(i) for i in final_sentence_arr]))

				# fill array with possible words
				for word in final_sentence_arr:
					if word not in possible_words:
						possible_words.append(word)
		# step 4: write results into another file
		fp_output.write(','.join(final_line_arr)+'\n')

	fp_input.close()
	fp_output.close()
	# print possible_words
	return possible_words

def write_csv_col_into_files(temp_list,category,words_num):
	'''
	this function writes csv results into output files
	'''
	temp_output = ''
	for sentence in temp_list:
		words = sentence.split()
		# # very slow col, it belongs 0 category
		result_arr = [category]
		for count in range(words_num):
			if possible_words[count] in words:
				# 1 means have the feature (word)
				result_arr.append('1')
			else:
				# 0 means have the feature
				result_arr.append('0')

		# this record if the sentences contains other words not listed in the
		# possible words list
		flag_other = False
		for word in words:
			if word not in possible_words:
				flag_other = True
				print word
		
		if flag_other:
			result_arr.append('1')
		else:
			result_arr.append('0')

		temp_output = temp_output + ','.join(result_arr)+'\n'

	return temp_output

def feedback_training_input_generator(input_file,output_file,possible_words):
	'''
	This function output the input files for the Apache Spark
	machine learning function
	'''
	pd_input = pd.read_csv(input_file,skipinitialspace=True)
	fp_output = open(output_file,'w')
	# write csv title
	new_title_arr = ['category']+possible_words+['other']
	fp_output.write(','.join(new_title_arr)+'\n')

	words_num = len(possible_words)
	# four categories
	# very slow col, it belongs 0 category
	temp_list = pd_input['very slow'].tolist()
	fp_output.write(write_csv_col_into_files(temp_list,'0',words_num))
	# slow col, it belongs 1 category
	temp_list = pd_input['slow'].tolist()
	fp_output.write(write_csv_col_into_files(temp_list,'1',words_num))
	# fast col, it belongs 2 category
	temp_list = pd_input['fast'].tolist()
	fp_output.write(write_csv_col_into_files(temp_list,'2',words_num))
	# very fast col, it belongs 3 category
	temp_list = pd_input['very fast'].tolist()
	fp_output.write(write_csv_col_into_files(temp_list,'3',words_num))
	# want to pay more for better performance col, it belongs 4 category
	# temp_list = pd_input[' want to pay more for better performance'].tolist()
	# fp_output.write(write_csv_col_into_files(temp_list,'4',words_num))

	# # two categories
	# # very slow col, it belongs 0 category
	# temp_list = pd_input['very slow'].tolist()
	# fp_output.write(write_csv_col_into_files(temp_list,'0',words_num))
	# # slow col, it belongs 1 category
	# temp_list = pd_input['slow'].tolist()
	# fp_output.write(write_csv_col_into_files(temp_list,'0',words_num))
	# # fast col, it belongs 2 category
	# temp_list = pd_input['fast'].tolist()
	# fp_output.write(write_csv_col_into_files(temp_list,'1',words_num))
	# # very fast col, it belongs 3 category
	# temp_list = pd_input['very fast'].tolist()
	# fp_output.write(write_csv_col_into_files(temp_list,'1',words_num))

	fp_output.close()
	

# ------------
# from util.py
# ------------
# the following construct_line and convert_csv_into_libsvm
# convert csv into libsvm
# the function is basically 
# from https://github.com/zygmuntz/phraug/blob/master/csv2libsvm.py
def construct_line( label, line ):
	new_line = []
	if float( label ) == 0.0:
		label = "0"
	new_line.append( label )

	for i, item in enumerate( line ):
		if item == '' or float( item ) == 0.0:
			continue
		new_item = "%s:%s" % ( i + 1, item )
		new_line.append( new_item )
	new_line = " ".join( new_line )
	new_line += "\n"
	return new_line

def convert_csv_into_libsvm(input_file,output_file,label_index=0,skip_headers=True):
	'''
	the function converts csv into libsvm
	'''
	i = open( input_file, 'rb' )
	o = open( output_file, 'wb' )
	reader = csv.reader( i )

	if skip_headers:
		headers = reader.next()

	for line in reader:
		if label_index == -1:
			label = '1'
		else:
			label = line.pop( label_index )

		new_line = construct_line( label, line )
		o.write( new_line )
# ------------
# end
# ------------

# possible_words = feedback_processing('feed_back_rui.csv','temp.csv')
possible_words = feedback_processing('feed_back_short.csv','temp.csv')
feedback_training_input_generator('temp.csv','input_ml.csv',possible_words)
convert_csv_into_libsvm('input_ml.csv','temp_libsvm.txt')


