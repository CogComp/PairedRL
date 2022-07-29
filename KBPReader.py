import xml.etree.ElementTree as ET
import spacy
import json
import os
import random
import pickle

import difflib

from classes import *


def read_2015(outfile, path):

	path = path+'2015/'

	nlp = spacy.load('en_core_web_sm')

	num_pos = 0
	num_neg = 0
	num_doc = 0
	num_event = 0
	num_singleton = 0
	num_cluster = 0

	for dir in ['training', 'eval']:
		data_path = path + dir
		source_path = data_path + '/source'
		ere_path = data_path + '/event_hopper'
		for filename in os.listdir(source_path):
			# print('File: ', filename)
			source_doc = open(source_path+'/'+filename, 'r')
			data = source_doc.read()
			text_sentences = nlp(data)

			num_doc += 1

			total_sen = 0
			total_char = 0

			sentence_mapping = []
			token_sen_mapping = []

			sentences = []
			sentences_str = ''
			sentences_offset = []
			events = {}
			clusters = {}
			for idx, sentence in enumerate(text_sentences.sents):

				total_char += len(sentence.text)
				sentences_str += sentence.text
				sentence_mapping += [idx]*len(sentence.text)
				total_sen += 1
				sentences.append(sentence.text)

			# print(len(data), total_char)
			i,j = 0,0
			last_sen_id = -1
			while i < len(data) and j < (len(sentences_str)):
				# print(data[i], sentences[j], i, j)
				if data[i] != sentences_str[j]:
					if data[i+1] == sentences_str[j]:
						token_sen_mapping.append(sentence_mapping[j])
						if sentence_mapping[j] != last_sen_id:
							last_sen_id = sentence_mapping[j]
							sentences_offset.append(i+1)
						i += 1
						continue
					if data[i] == sentences_str[j+1]:
						j += 1
						continue
				else:
					token_sen_mapping.append(sentence_mapping[j])
					if sentence_mapping[j] != last_sen_id:
						last_sen_id = sentence_mapping[j]
						sentences_offset.append(i)
				i += 1
				j += 1
			# print(i,j, len(token_sen_mapping), len(sentences_offset), len(sentences))
			if i != len(data) or j != len(sentences_str) or len(token_sen_mapping) != len(data):
				print('Data Error')
				break

			tree = ET.parse(ere_path + '/' + filename.replace('.txt', '') + '.event_hoppers.xml')
			root = tree.getroot()
			for event in root.iter('event_mention'):
				# print(event.tag, event.attrib)
				for trigger in event:
					# print(trigger.tag, trigger.text, trigger.attrib)
					events[event.attrib['id']] = {'text': trigger.text, 'offset': int(trigger.attrib['offset'])}
					num_event += 1
			for cluster in root.iter('hopper'):
				# print(cluster.attrib['id'])
				spans = []
				for event in cluster:
					# print(event.attrib['id'])
					spans.append(event.attrib['id'])
				for span in spans:
					clusters[span] = {}
					for span_2 in spans:
						clusters[span][span_2] = True
				num_cluster += 1
				if len(spans) == 1:
					num_singleton += 1
			event_id = list(events.keys())
			for i in range(len(event_id)):
				for j in range(i + 1, len(event_id)):
					if token_sen_mapping[events[event_id[i]]['offset']] <= token_sen_mapping[
						events[event_id[j]]['offset']]:
						event_i = events[event_id[i]]
						event_j = events[event_id[j]]
					else:
						event_i = events[event_id[j]]
						event_j = events[event_id[i]]

					outstr = []
					sen1_idx = token_sen_mapping[event_i['offset']]
					offset = sentences_offset[sen1_idx]
					event_i_token = event_i['text'].split(' ')
					if event_i['offset'] - offset + len(event_i['text']) < len(sentences[sen1_idx]) and \
							sentences[sen1_idx][event_i['offset'] - offset + len(event_i['text'])] != ' ':
						sen1 = sentences[sen1_idx][0:event_i['offset'] - offset + len(event_i['text'])] + ' ' + \
						       sentences[sen1_idx][event_i['offset'] - offset + len(event_i['text']):]
					else:
						sen1 = sentences[sen1_idx]

					# print(event_i['offset'])
					# print(len(sentences[sen1_idx]))
					# print(len(sen1))
					if event_i['offset'] - offset > 0 and sen1[event_i['offset'] - offset - 1] != ' ':
						sen1 = sen1[0:event_i['offset'] - offset] + ' ' + sen1[event_i['offset'] - offset:]
						offset -= 1

					sen1_token = sen1.replace('\n', ' ').replace('\t', ' ').split(' ')

					flag = 0
					outstr.append(' '.join(sen1_token))
					for idx, sen_token in enumerate(sen1_token):
						if offset == event_i['offset']:
							outstr.append(str(idx))
							outstr.append(str(idx + len(event_i_token)-1))
							# outstr.append(event_i['text'])
							flag = 1
							if sen1_token[idx: idx + len(event_i_token)] != event_i_token:
								print(outstr)
								print('I: Data Error')
								exit(0)
							break
						else:
							offset += len(sen_token) + 1
					if flag == 0:
						print(sen1)
						print(event_i['text'])
					sen2_idx = token_sen_mapping[event_j['offset']]
					offset = sentences_offset[sen2_idx]
					event_j_token = event_j['text'].split(' ')
					# print(sentences[sen2_idx])
					# print(sentences_offset[sen2_idx])
					# print(event_j['text'])
					# print(event_j['offset'])
					if event_j['offset'] - offset + len(event_j['text']) < len(sentences[sen2_idx]) and \
							sentences[sen2_idx][event_j['offset'] - offset + len(event_j['text'])] != ' ':
						sen2 = sentences[sen2_idx][0:event_j['offset'] - offset + len(event_j['text'])] + ' ' + \
						       sentences[sen2_idx][event_j['offset'] - offset + len(event_j['text']):]
					else:
						sen2 = sentences[sen2_idx]

					if event_j['offset'] - offset > 0 and sen2[event_j['offset'] - offset - 1] != ' ':
						sen2 = sen2[0:event_j['offset'] - offset] + ' ' + sen2[event_j['offset'] - offset:]
						offset -= 1

					sen2_token = sen2.replace('\n', ' ').replace('\t', ' ').split(' ')
					outstr.append(' '.join(sen2_token))
					flag = 0
					for idx, sen_token in enumerate(sen2_token):
						if offset == event_j['offset']:
							outstr.append(str(idx))
							outstr.append(str(idx + len(event_j_token)-1))
							# outstr.append(event_j['text'])
							if sen2_token[idx: idx + len(event_j_token)] != event_j_token:
								print(sen2_token)
								print(outstr)
								print('J: Data Error')
								exit(0)
							flag = 1
							break
						else:
							offset += len(sen_token) + 1
					if flag == 0:
						print(sen2)
						print(event_j['text'])
					if event_id[j] in clusters[event_id[i]]:
						outstr.append('1')
						num_pos += 1
					else:
						outstr.append('0')
						if random.random() < 0.5:
							continue
						num_neg += 1
					# outstr.append('\n')
					if len(outstr) != 7:
						print(outstr)
					outfile.write('\t'.join(outstr)+'\n')
	# print(num_pos, num_neg)
	print(num_doc, num_event, num_singleton, num_cluster)

		# 	break
		# break


def read_2016_2017(outfile, path, outfile2=None):

	if outfile2 is None:
		path = path+'2017/'
	else:
		path = path+'2016/'

	nlp = spacy.load('en_core_web_sm')
	num_pos = 0
	num_neg = 0
	num_doc = 0
	num_event = 0
	num_singleton = 0
	num_cluster = 0

	data = {}

	for dir in ['df', 'nw']:
		data_path = path + dir
		source_path = data_path + '/source'
		ere_path = data_path + '/ere'
		for doc_idx, filename in enumerate(os.listdir(source_path)):
			# print('File: ', filename)
			doc_id = dir+'_'+filename
			if doc_id not in data:
				data[doc_id] = {}

			source_doc = open(source_path+'/'+filename, 'r')
			doc_data = source_doc.read()
			text_sentences = nlp(doc_data)

			num_doc += 1

			total_cluster = 0
			total_sen = 0
			total_char = 0

			sentence_mapping = []
			token_sen_mapping = []

			sentences = []
			sentences_str = ''
			sentences_offset = []
			events = {}
			clusters = {}
			for idx, sentence in enumerate(text_sentences.sents):

				total_char += len(sentence.text)
				sentences_str += sentence.text
				sentence_mapping += [idx]*len(sentence.text)
				total_sen += 1
				sentences.append(sentence.text)

				doc = ACEDoc(doc_id)
				sen = Sentence(doc_id, str(idx))
				doc.sentences['0'] = sen
				data[doc_id][str(idx)] = doc

			# print(len(data), total_char)
			i,j = 0,0
			last_sen_id = -1
			while i < len(doc_data) and j < (len(sentences_str)):
				# print(data[i], sentences[j], i, j)
				if doc_data[i] != sentences_str[j]:
					if doc_data[i+1] == sentences_str[j]:
						token_sen_mapping.append(sentence_mapping[j])
						if sentence_mapping[j] != last_sen_id:
							last_sen_id = sentence_mapping[j]
							sentences_offset.append(i+1)
						i += 1
						continue
					if doc_data[i] == sentences_str[j+1]:
						j += 1
						continue
				else:
					token_sen_mapping.append(sentence_mapping[j])
					if sentence_mapping[j] != last_sen_id:
						last_sen_id = sentence_mapping[j]
						sentences_offset.append(i)
				i += 1
				j += 1
			# print(i,j, len(token_sen_mapping), len(sentences_offset), len(sentences))
			if i != len(doc_data) or j != len(sentences_str) or len(token_sen_mapping) != len(doc_data):
				print('Data Error')
				break

			tree = ET.parse(ere_path+'/'+filename.replace('.xml','')+'.rich_ere.xml')
			root = tree.getroot()

			for cluster in root.iter('hopper'):
				for event in cluster:
					# print(event.tag, event.attrib)
					for trigger in event:
						# print(trigger.tag, trigger.text, trigger.attrib)
						if trigger.tag == 'trigger':
							events[event.attrib['id']] = {'text': trigger.text, 'offset': int(trigger.attrib['offset'])}
							sen_idx = str(token_sen_mapping[int(trigger.attrib['offset'])])
							event = Event_ACE(sen_idx, [], int(trigger.attrib['offset']),
							                  int(trigger.attrib['offset'])+int(trigger.attrib['length'])-1,
							                  '_'.join([filename, event.attrib['id']]))
							event.gold_cluster = doc_id + '_' + cluster.attrib['id']
							data[doc_id][sen_idx].sentences['0'].events.append(event)
							num_event += 1

			for cluster in root.iter('hopper'):
				# print(cluster.attrib['id'])
				total_cluster += 1
				spans = []
				for event in cluster:
					# print(event.attrib['id'])
					spans.append(event.attrib['id'])
				for span in spans:
					clusters[span] = {}
					for span_2 in spans:
						clusters[span][span_2] = True
				num_cluster += 1
				if len(spans) == 1:
					num_singleton += 1

			# print(filename, total_cluster)

			event_id = list(events.keys())
			for i in range(len(event_id)):
				for j in range(i+1, len(event_id)):
					if token_sen_mapping[events[event_id[i]]['offset']] <= token_sen_mapping[events[event_id[j]]['offset']]:
						event_i = events[event_id[i]]
						event_j = events[event_id[j]]
					else:
						event_i = events[event_id[j]]
						event_j = events[event_id[i]]

					if outfile2 is None:
						outstr = ['_'.join([filename, event_id[i]]), '_'.join([filename, event_id[j]])]
					else:
						outstr = []
						# outstr = ['_'.join([filename, event_id[i]]), '_'.join([filename, event_id[j]])]

					sen1_idx = token_sen_mapping[event_i['offset']]
					offset = sentences_offset[sen1_idx]
					event_i_token = event_i['text'].split(' ')
					if event_i['offset']-offset + len(event_i['text']) < len(sentences[sen1_idx]) and \
							sentences[sen1_idx][event_i['offset']-offset + len(event_i['text'])] != ' ':
						sen1 = sentences[sen1_idx][0:event_i['offset']-offset + len(event_i['text'])] + ' ' + \
						       sentences[sen1_idx][event_i['offset'] - offset + len(event_i['text']): ]
					else:
						sen1 = sentences[sen1_idx]

					# print(event_i['offset'])
					# print(len(sentences[sen1_idx]))
					# print(len(sen1))
					if event_i['offset'] - offset > 0 and sen1[event_i['offset'] - offset - 1] != ' ':
						sen1 = sen1[0:event_i['offset']-offset] + ' ' + sen1[event_i['offset']-offset:]
						offset -= 1

					sen1_token = sen1.replace('\n', ' ').replace('\t', ' ').split(' ')
					
					flag = 0
					outstr.append(' '.join(sen1_token))
					for idx, sen_token in enumerate(sen1_token):
						if offset == event_i['offset']:
							outstr.append(str(idx))
							outstr.append(str(idx + len(event_i_token)-1))
							# outstr.append(event_i['text'])
							flag = 1
							if sen1_token[idx: idx + len(event_i_token)] != event_i_token:
								print(outstr)
								print('I: Data Error')
								exit(0)
							break
						else:
							offset += len(sen_token)+1
					if flag == 0:
						print(sen1)
						print(event_i['text'])
					sen2_idx = token_sen_mapping[event_j['offset']]
					offset = sentences_offset[sen2_idx]
					event_j_token = event_j['text'].split(' ')
					# print(sentences[sen2_idx])
					# print(sentences_offset[sen2_idx])
					# print(event_j['text'])
					# print(event_j['offset'])
					if event_j['offset']-offset + len(event_j['text']) < len(sentences[sen2_idx]) and \
							sentences[sen2_idx][event_j['offset']-offset + len(event_j['text'])] != ' ':
						sen2 = sentences[sen2_idx][0:event_j['offset']-offset + len(event_j['text'])] + ' ' + \
						       sentences[sen2_idx][event_j['offset'] - offset + len(event_j['text']):]
					else:
						sen2 = sentences[sen2_idx]

					if event_j['offset'] - offset > 0 and sen2[event_j['offset'] - offset - 1] != ' ':
						sen2 = sen2[0:event_j['offset']-offset] + ' ' + sen2[event_j['offset']-offset:]
						offset -= 1

					sen2_token = sen2.replace('\n', ' ').replace('\t', ' ').split(' ')
					outstr.append(' '.join(sen2_token))
					flag = 0
					for idx, sen_token in enumerate(sen2_token):
						if offset == event_j['offset']:
							outstr.append(str(idx))
							outstr.append(str(idx + len(event_j_token)-1))
							# outstr.append(event_j['text'])
							if sen2_token[idx: idx + len(event_j_token)] != event_j_token:
								print(sen2_token)
								print(outstr)
								print('J: Data Error')
								exit(0)
							flag = 1
							break
						else:
							offset += len(sen_token)+1
					if flag == 0:
						print(sen2)
						print(event_j['text'])
					if event_id[j] in clusters[event_id[i]]:
						num_pos += 1
						outstr.append('1')
					else:
						num_neg += 1
						outstr.append('0')
					outstr.append('\n')
					# if doc_idx <= len(os.listdir(source_path)) / 3:
					# 	outfile.write('\t'.join(outstr))
					# else:
					if outfile2 is None:
						outfile.write('\t'.join(outstr))
					else:
						outfile2.write('\t'.join(outstr))
	# print(num_pos, num_neg)
	print(num_doc, num_event, num_singleton, num_cluster)

	if outfile2 is None:
		file = open('data/preprocessed/kbp.test', 'wb')
		pickle.dump(data, file)
	else:
		file = open('data/preprocessed/kbp.dev', 'wb')
		pickle.dump(data, file)





if __name__ == "__main__":
	path = '../../data/KBP/'

	train_file = open('data/pairs_KBP_fixed.train', 'w')
	dev_file = open('data/pairs_KBP_fixed.dev', 'w')
	test_file = open('data/pairs_KBP_fixed.test', 'w')

	print('Reading Train:')
	read_2015(train_file, path)
	print('Reading Dev:')
	read_2016_2017(None, path, dev_file)
	print('Reading Test:')
	read_2016_2017(test_file, path, None)



	#
	# outfile = open('data/preprocessed/ecb+.'+split, 'wb')
	# pickle.dump(data, outfile)

	# Run_Nominal_SRL('train')