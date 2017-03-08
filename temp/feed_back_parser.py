def feedback_processing(input_file,output_file):
	'''
	This function parse the input files and
	output the cleaned data without unnecessary
	words, and return possible words in the samples
	'''
	fp_input = open(input_file,'r')
	fp_output = open(output_file,'w')

	# replace replace_pair[0] with replace_pair[1]
	replace_pair = [['isn\'t','don\'t','aren\'t','cannot','can\'t'],'not']

	unnecessary_words = ['we','i','you','should','like','would','may',
	                    'maybe','to','server','service','queue','is',
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
			final_sentence_arr = []
			# final_sentence = ''
			# remove period and collect words in the sentence
			words = sentence.replace('.','').split()

			for word in words:
				# step 2: replace words in the sentences
				# replace word with "not"
				if word in replace_pair[0]:
					final_sentence_arr.append(replace_pair[1])
				
				# # step 3: remove unnecessary words
				elif word not in unnecessary_words:
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

def feedback_training_input_generator(input_file,output_file,possible_words):
	'''
	This function output the input files for the Apache Spark
	machine learning function
	'''
	fp_input = open(input_file,'r')
	fp_output = open(output_file,'w')
	# write csv title
	new_title_arr = ['category']+possible_words+['other']
	fp_output.write(','.join(new_title_arr))
	

	fp_input.close()
	fp_output.close()
	


possible_words = feedback_processing('feed_back_rui.csv','temp.csv')
feedback_training_input_generator('temp.csv','input_ml.csv',possible_words)