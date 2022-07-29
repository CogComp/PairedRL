import xml.dom.minidom
import os
import csv
# from allennlp.predictors.predictor import Predictor
import json
import spacy

import pickle

import numpy as np

from classes import *


data = {}
srl_result = {}
pos_result = {}

event_all = []
entity_all = []

validate_sen = {}

# dev = ['2', '5', '12', '18', '21', '23', '34', '35']
# train = [str(i) for i in range(1, 36) if str(i) not in dev]
# test = [str(i) for i in range(36, 46)]

topic_split = {}
topic_split['dev'] = ['2', '5', '12', '18', '21', '23', '34', '35']
topic_split['train'] = [str(i) for i in range(1, 36) if str(i) not in topic_split['dev'] and i not in [15,17]]
topic_split['test'] = [str(i) for i in range(36, 46)]





def ReadData(path, split):

	sen_sum = 0
	mention_sum = 0
	event_sum = 0
	entity_sum = 0

	for topic in os.listdir(path):
	# for topic in ['tmp']:
		if topic == ".DS_Store":
			continue
		if topic not in topic_split[split]:
			continue

		data[topic] = {}

		for filename in os.listdir(path + '/' + topic + '/'):

			doc = ECBDoc()
			xmlfile = xml.dom.minidom.parse(path + '/' + topic + '/' + filename)
			doc_info = xmlfile.getElementsByTagName("Document")
			doc.set_doc_info(doc_info[0].getAttribute("doc_id"), doc_info[0].getAttribute("doc_name"))

			tokens = xmlfile.getElementsByTagName("token")

			if doc.doc_name not in validate_sen[topic]:
				continue

			# print(doc.doc_name)
			# doc.sentences[0] = Sentence(topic, doc.doc_name)


			for token in tokens:
				# t_id is 1 base. tokens is 0 base
				sen_id = int(token.getAttribute('sentence'))
				if str(sen_id) not in validate_sen[topic][doc.doc_name]:
					continue
				# print(sen_id)
				doc.tokens.append(token.childNodes[0].data)
				if sen_id not in doc.sentences:
					doc.sentences[sen_id] = Sentence(topic, doc.doc_name)
					# print(sen_id)
					sen_sum += 1
				t_id = int(token.getAttribute("t_id")) - 1
				doc.sentences[sen_id].tokens.append((token.childNodes[0].data, t_id))
				doc.token2sentence[t_id] = (sen_id, int(token.getAttribute('number')))

			mentions = xmlfile.getElementsByTagName("Markables")[0].childNodes
			for mention in mentions:
				if mention.nodeName == '#text':
					continue
				mention_id = int(mention.getAttribute('m_id'))

				if len(mention.childNodes) == 0:
					doc.cluster_head.append(int(mention.getAttribute('m_id')))
					doc.cluster_type[int(mention.getAttribute('m_id'))] = mention.tagName
					if mention.getAttribute('instance_id') != '':
						doc.cluster_instance_id[int(mention.getAttribute('m_id'))] = mention.getAttribute('instance_id')
					else:
						doc.cluster_instance_id[int(mention.getAttribute('m_id'))] = mention.nodeName
				else:
					flag = False
					for token in mention.childNodes:
						if token.nodeName == '#text':
							continue
						t_id = int(token.getAttribute("t_id")) - 1

						if t_id not in doc.token2sentence:
							flag = True
							break

						if mention_id not in doc.mention2token:
							doc.mention2token[mention_id] = []
						doc.mention2token[mention_id].append(t_id)
						doc.token2mention[t_id] = mention_id

					if flag:
						continue

					doc.mentions_type[mention_id] = mention.tagName

					mention_sum += 1

					sen_id = doc.token2sentence[doc.mention2token[mention_id][0]][0]
					start_offset = doc.token2sentence[doc.mention2token[mention_id][0]][1]
					end_offset = doc.token2sentence[doc.mention2token[mention_id][-1]][1]
					if 'ACT' in mention.tagName or 'NEG' in mention.tagName:
						event_sum += 1

						event = Event(topic, doc.doc_name, sen_id, start_offset, end_offset, doc.mention2token[mention_id][0],
						              doc.mention2token[mention_id][-1])
						event_all.append(event)
						doc.sentences[sen_id].events.append(event)
						doc.mentions[mention_id] = event

					else:
						entity_sum += 1

						entity = Entity(topic, doc.doc_name, sen_id, start_offset, end_offset, doc.mention2token[mention_id][0],
						                doc.mention2token[mention_id][-1], doc.mentions_type[mention_id])
						entity_all.append(entity)
						doc.sentences[sen_id].entities.append(entity)
						doc.mentions[mention_id] = entity


			corefs = xmlfile.getElementsByTagName("Relations")[0].childNodes
			for coref in corefs:
				if coref.nodeName == '#text':
					continue
				coref_list = []
				head_id = 0
				for mention in coref.childNodes:
					if mention.nodeName == '#text':
						continue
					if mention.nodeName == 'source':
						coref_list.append(int(mention.getAttribute("m_id")))
					if mention.nodeName == 'target':
						head_id = int(mention.getAttribute("m_id"))
					# coref_list.append(head_id)
				# print(coref_list)
				doc.clusters[head_id] = coref_list
				for mention in coref_list:
					if mention in doc.mentions:
						if coref.nodeName == 'INTRA_DOC_COREF':
							doc.mentions[mention].gold_cluster = '_'.join(['INTRA', doc.cluster_instance_id[head_id],
							                                               coref.getAttribute("r_id"), doc.doc_name])
						else:
							doc.mentions[mention].gold_cluster = doc.cluster_instance_id[head_id]
					doc.coref_mention[mention] = coref_list
					doc.mention2cluster_id[mention] = doc.cluster_instance_id[head_id]

			data[topic][doc_info[0].getAttribute("doc_name")] = doc

	print("number of events: ", len(event_all))
	print("number of entities: ", len(entity_all))
	#
	# print(mention_sum, event_sum, entity_sum, sen_sum)

	global srl_result, nominal_result
	srl_result = Run_SRL(True)
	nominal_result = Run_Nominal_SRL(split)

	for topic in data:
		for doc_name in data[topic]:
			doc = data[topic][doc_name]
			for sen_id in doc.sentences:
				sen = doc.sentences[sen_id]
				srl_dict = srl_result[topic][doc_name][str(sen_id)]
				idx_mapping = {}
				i,j = 0,0

				srl_tokens = srl_dict['words']
				while i < len(sen.tokens):
					token = sen.tokens[i][0]
					while True:
						if token.startswith(srl_tokens[j]):
							start_idx = j
							token_srl = srl_tokens[j]
							j += 1
							break
						else:
							j += 1
					while token != token_srl and j < len(srl_tokens):
						if token.startswith((token_srl+srl_tokens[j])):
							token_srl += srl_tokens[j]
						j += 1
					idx_mapping[i] = (start_idx, j)
					i += 1

				for event in sen.events:
					for verb in srl_dict['verbs']:
						flag = False
						for i in range(idx_mapping[event.start_offset][0], idx_mapping[event.end_offset][1]):

							if '-V' in verb['tags'][i]:
								flag = True
								break
						if not flag:
							continue

						for entity in sen.entities:
							for i in range(idx_mapping[entity.start_offset][0], idx_mapping[entity.end_offset][1]):
								if verb['tags'][i] != 'O':
									type = verb['tags'][i].split('-')[-1]
									if type == 'ARG0' and 'LOC' not in entity.type and 'TIME' not in entity.type:
										event.arg0 = (entity.start_offset, entity.end_offset)
										break
									if type == 'ARG1' and 'LOC' not in entity.type and 'TIME' not in entity.type:
										event.arg1 = (entity.start_offset, entity.end_offset)
										break
									if type == 'LOC' and 'LOC' in entity.type:
										event.loc = (entity.start_offset, entity.end_offset)
										break
									if type == 'TMP' and 'TIME' in entity.type:
										event.time = (entity.start_offset, entity.end_offset)
										break

				nominal_dict = nominal_result[topic][doc_name][sen_id]
				idx_mapping = {}
				i,j = 0,0

				nominal_tokens = nominal_dict['words']
				while i < len(sen.tokens):
					token = sen.tokens[i][0]
					while True:
						if token.startswith(nominal_tokens[j]):
							start_idx = j
							token_srl = nominal_tokens[j]
							j += 1
							break
						else:
							j += 1
					while token != token_srl and j < len(nominal_tokens):
						if token.startswith((token_srl+nominal_tokens[j])):
							token_srl += nominal_tokens[j]
						j += 1
					idx_mapping[i] = (start_idx, j)
					i += 1
				for event in sen.events:
					for nominal in nominal_dict['nominals']:
						flag = False
						for idx in nominal['predicate_index']:
							if idx in range(idx_mapping[event.start_offset][0], idx_mapping[event.end_offset][1]):
								flag = True
								break
						if not flag:
							continue

						for entity in sen.entities:
							for i in range(idx_mapping[entity.start_offset][0], idx_mapping[entity.end_offset][1]):
								if nominal['tags'][i] != 'O':
									type = nominal['tags'][i].split('-')[-1]
									if event.arg0 == (-1,-1) and type == 'ARG0' and 'LOC' not in entity.type and 'TIME' not in entity.type:
										event.arg0 = (entity.start_offset, entity.end_offset)
										break
									if event.arg1 == (-1,-1) and type == 'ARG1' and 'LOC' not in entity.type and 'TIME' not in entity.type:
										event.arg1 = (entity.start_offset, entity.end_offset)
										break
									if event.loc == (-1,-1) and type == 'LOC' and 'LOC' in entity.type:
										event.loc = (entity.start_offset, entity.end_offset)
										break
									if event.time == (-1,-1) and type == 'TMP' and 'TIME' in entity.type:
										event.time = (entity.start_offset, entity.end_offset)
										break


def isSameTopic(mention1, mention2, useGoldSubtopic, predict_topic):

	if useGoldSubtopic:
		if mention1.topic == mention2.topic:
			if 'plus' in mention1.doc_id and 'plus' in mention2.doc_id:
				return True
			if 'plus' not in mention1.doc_id and 'plus' not in mention2.doc_id:
				return True
		return False

	else:
		for topics in predict_topic:
			if mention1.doc_id.replace('.xml', '') in topics and mention2.doc_id.replace('.xml', '') in topics:
				return True
		return False


def GeneratePairs(split):

	# outfile = open('data/pairs_trigger_only.' + split, 'w')
	outfile = open('data/pairs_trigger_only_verb.' + split, 'w')
	# outfile = open('data/pairs.' + split, 'w')

	predict_topic_file = open('data/preprocessed/predicted_topics', 'rb')
	predict_topic = pickle.load(predict_topic_file)

	num_pos = 0
	num_neg = 0
	num_single = 0


	num_doc = 0
	num_sen = 0
	for topic in topic_split[split]:
		num_doc += len(data[topic])
		for doc_id in data[topic]:
			doc = data[topic][doc_id]
			num_sen += len(doc.sentences.keys())
	# print(num_doc, num_sen)

	print(len(event_all))

	event_info_file = open('event_info.out', 'w')
	event_id = []
	event_cluster = {}
	num_cluster = 0
	for (i, event_i) in enumerate(event_all):
		if 'Singleton' in event_i.gold_cluster:
			num_single += 1
			continue
		if event_i.gold_cluster not in event_cluster:
			event_cluster[event_i.gold_cluster] = 1
			num_cluster += 1
		else:
			event_cluster[event_i.gold_cluster] += 1
		event_id.append(event_i.gold_cluster+'_'+event_i.mention_id)

	for key in event_cluster:
		if event_cluster[key] == 1:
			# print(key)
			num_single += 1
			num_cluster -= 1

	# print(num_single, num_cluster)
	event_id = sorted(event_id)
	for event in event_id:
		event_info_file.write(event+'\n')

	filterNominals = True
	if filterNominals:
		pos_result = Run_POS(False)
		filteredEvent = []
		for event in event_all:
			pos_list = pos_result[event.topic][event.doc_id][event.sen_id]

			flag = 0
			for i in range(event.start_offset, event.end_offset+1):
				if pos_list[i] == 'VERB':
					filteredEvent.append(event)
					flag = 1
					break
			if flag == 0:
				# print(data[event.topic][event.doc_id].sentences[event.sen_id].tokens)
				print(data[event.topic][event.doc_id].sentences[event.sen_id].tokens[event.start_offset: event.end_offset+1], pos_list[event.start_offset: event.end_offset+1])
	else:
		filteredEvent = event_all
	print('Verb Events number: ', len(filteredEvent))

	if split == 'test':
		useGoldSubtpoic = False
	else:
		useGoldSubtpoic = True

	for (i, event_i) in enumerate(filteredEvent):
		for (j, event_j) in enumerate(filteredEvent):
			if i >= j or not isSameTopic(event_i, event_j, useGoldSubtopic=useGoldSubtpoic, predict_topic=predict_topic):
				continue
			outstr = [event_i.mention_id, event_j.mention_id]
			# outstr = []
			sen_i = data[event_i.topic][event_i.doc_id].sentences[event_i.sen_id]
			outstr.append(' '.join([token[0].replace('\t','') for token in sen_i.tokens]))
			outstr.append(str(event_i.start_offset))
			outstr.append(str(event_i.end_offset))
			# outstr.append(str(event_i.arg0[0]))
			# outstr.append(str(event_i.arg0[1]))
			# outstr.append(str(event_i.arg1[0]))
			# outstr.append(str(event_i.arg1[1]))
			# outstr.append(str(event_i.loc[0]))
			# outstr.append(str(event_i.loc[1]))
			# outstr.append(str(event_i.time[0]))
			# outstr.append(str(event_i.time[1]))

			sen_j = data[event_j.topic][event_j.doc_id].sentences[event_j.sen_id]
			outstr.append(' '.join([token[0].replace('\t','') for token in sen_j.tokens]))
			outstr.append(str(event_j.start_offset))
			outstr.append(str(event_j.end_offset))
			# outstr.append(str(event_j.arg0[0]))
			# outstr.append(str(event_j.arg0[1]))
			# outstr.append(str(event_j.arg1[0]))
			# outstr.append(str(event_j.arg1[1]))
			# outstr.append(str(event_j.loc[0]))
			# outstr.append(str(event_j.loc[1]))
			# outstr.append(str(event_j.time[0]))
			# outstr.append(str(event_j.time[1]))

			if event_i.gold_cluster == event_j.gold_cluster:
				outstr.append('1')
				num_pos += 1
			else:
				outstr.append('0')
				num_neg += 1

			outstr.append('\n')

			outfile.write('\t'.join(outstr))
	print(num_pos, num_neg)


def GenerateEntityPairs(split):

	outfile = open('data/pairs_entity.' + split, 'w')
	# outfile = open('data/pairs.' + split, 'w')

	predict_topic_file = open('data/preprocessed/predicted_topics', 'rb')
	predict_topic = pickle.load(predict_topic_file)

	num_pos = 0
	num_neg = 0
	num_single = 0


	num_doc = 0
	num_sen = 0
	for topic in topic_split[split]:
		num_doc += len(data[topic])
		for doc_id in data[topic]:
			doc = data[topic][doc_id]
			num_sen += len(doc.sentences.keys())
	print(num_doc, num_sen)

	print(len(entity_all))

	# event_info_file = open('event_info.out','w')
	entity_id = []
	entity_cluster = {}
	num_cluster = 0
	for (i, entity_i) in enumerate(entity_all):
		if 'Singleton' in entity_i.gold_cluster:
			num_single += 1
			continue
		if entity_i.gold_cluster not in entity_cluster:
			entity_cluster[entity_i.gold_cluster] = 1
			num_cluster += 1
		else:
			entity_cluster[entity_i.gold_cluster] += 1
		entity_id.append(entity_i.gold_cluster+'_'+entity_i.mention_id)

	for key in entity_cluster:
		if entity_cluster[key] == 1:
			# print(key)
			num_single += 1
			num_cluster -= 1

	print(num_single, num_cluster)
	entity_id = sorted(entity_id)
	# for event in event_id:
	# 	event_info_file.write(event+'\n')

	for (i, entity_i) in enumerate(entity_all):
		for (j, entity_j) in enumerate(entity_all):
			if i >= j or not isSameTopic(entity_i, entity_j, True, predict_topic):
				continue
			# outstr = [entity_i.mention_id, entity_j.mention_id]
			outstr = []
			# print(entity_i.mention_id, entity_j.mention_id)
			# print(data[entity_i.topic][entity_j.doc_id].sentences.keys())
			sen_i = data[entity_i.topic][entity_i.doc_id].sentences[entity_i.sen_id]
			outstr.append(' '.join([token[0].replace('\t','') for token in sen_i.tokens]))
			outstr.append(str(entity_i.start_offset))
			outstr.append(str(entity_i.end_offset))
			# outstr.append(str(event_i.arg0[0]))
			# outstr.append(str(event_i.arg0[1]))
			# outstr.append(str(event_i.arg1[0]))
			# outstr.append(str(event_i.arg1[1]))
			# outstr.append(str(event_i.loc[0]))
			# outstr.append(str(event_i.loc[1]))
			# outstr.append(str(event_i.time[0]))
			# outstr.append(str(event_i.time[1]))

			sen_j = data[entity_j.topic][entity_j.doc_id].sentences[entity_j.sen_id]
			outstr.append(' '.join([token[0].replace('\t','') for token in sen_j.tokens]))
			outstr.append(str(entity_j.start_offset))
			outstr.append(str(entity_j.end_offset))
			# outstr.append(str(event_j.arg0[0]))
			# outstr.append(str(event_j.arg0[1]))
			# outstr.append(str(event_j.arg1[0]))
			# outstr.append(str(event_j.arg1[1]))
			# outstr.append(str(event_j.loc[0]))
			# outstr.append(str(event_j.loc[1]))
			# outstr.append(str(event_j.time[0]))
			# outstr.append(str(event_j.time[1]))

			if entity_i.gold_cluster == entity_j.gold_cluster:
				outstr.append('1')
				num_pos += 1
			else:
				outstr.append('0')
				num_neg += 1

			outstr.append('\n')

			# outfile.write('\t'.join(outstr))
	print(num_pos, num_neg)



def PrintData():
	outfile = open('out.csv', 'w', newline='')
	writer = csv.writer(outfile)
	writer.writerow(['Topic', 'Document', 'Sentence', 'Event', 'Event Type', 'Cluster ID', 'ARG0', 'ARG1', 'LOC', 'TMP', 'SRL'])

	# outfile_nominal = open('nominal_srl.input', 'w')

	for topic in data:
		for doc_name in data[topic]:
			doc = data[topic][doc_name]
			for sen_id in doc.sentences:
				sen = doc.sentences[sen_id]
				# print(json.dumps({'sentence': ' '.join([token[0] for token in sen.tokens])}))
				for event in sen.events:
					out_str = []
					out_str.append(topic)
					out_str.append(doc_name)
					out_str.append(' '.join([token[0] for token in sen.tokens]))
					out_str.append(' '.join([token[0] for token in sen.tokens[event.start_offset:event.end_offset+1]]))
					out_str.append('Event')
					out_str.append(event.gold_cluster)
					out_str.append(' '.join([token[0] for token in sen.tokens[event.arg0[0]:event.arg0[1]+1]]))
					out_str.append(' '.join([token[0] for token in sen.tokens[event.arg1[0]:event.arg1[1]+1]]))
					out_str.append(' '.join([token[0] for token in sen.tokens[event.loc[0]:event.loc[1]+1]]))
					out_str.append(' '.join([token[0] for token in sen.tokens[event.time[0]:event.time[1]+1]]))

					# srl_dict = srl_result[topic][doc_name][str(sen_id)]
					# for verb in srl_dict['verbs']:
					# 	surface = ''
					# 	for idx, tag in enumerate(verb['tags']):
					# 		if tag.startswith('B-'):
					# 			if surface != '':
					# 				surface += '; ' + tag +': '
					# 		if tag != 'O':
					# 			surface += tag + ': ' + srl_dict['words'][idx] + ' '
					# 	out_str.append(surface)

					nominal_dict = nominal_result[topic][doc_name][sen_id]
					for nominal in nominal_dict['nominals']:
						# print(nominal)
						surface = 'Nominal: ' + nominal['nominal'] + ' ; '
						for idx, tag in enumerate(nominal['tags']):
							if tag.startswith('B-'):
								if surface != '':
									surface += '; ' + tag +': '
							if tag != 'O':
								surface += tag + ': ' + nominal_dict['words'][idx] + ' '
						out_str.append(surface)

					writer.writerow(out_str)

				writer.writerow([''])

				for entity in sen.entities:
					out_str = []
					out_str.append(topic)
					out_str.append(doc_name)
					out_str.append(' '.join([token[0] for token in sen.tokens]))
					out_str.append(' '.join([token[0] for token in sen.tokens[entity.start_offset:entity.end_offset+1]]))
					out_str.append('Entity')
					out_str.append(entity.gold_cluster)

					writer.writerow(out_str)
				writer.writerow([''])


def print_cross_doc_clusters():
	clusters_doc_list = {}

	for topic in data:
		clusters_doc_list[topic] = {}
		for doc_name in data[topic]:
			doc = data[topic][doc_name]
			for key in doc.clusters:
				cluster_id = doc.cluster_instance_id[key]
				if cluster_id not in clusters_doc_list[topic]:
					clusters_doc_list[topic][cluster_id] = []
				clusters_doc_list[topic][cluster_id].append(doc_name)
	# print(clusters_doc_list)

	outfile = open('out.csv', 'w', newline='')
	writer = csv.writer(outfile)
	writer.writerow(['Cluster ID', 'Document', 'Sentence', 'Event', 'SRL output'])
	for topic in data:
		for cluster_id in clusters_doc_list[topic]:
			if cluster_id == '':
				continue
			writer.writerow([''])
			for doc_name in clusters_doc_list[topic][cluster_id]:
				# print(topic+'\t' + doc_name)
				doc = data[topic][doc_name]
				key = [key for key, cluster in doc.cluster_instance_id.items() if cluster == cluster_id][0]
				# print(key, doc.cluster_type[key])
				if 'ACTION' not in doc.cluster_type[key] and 'UNKNOWN' not in doc.cluster_type[key]:
					continue
				# writer.writerow([cluster_id])
				for mention in doc.clusters[key]:
					# print("add mention")
					out_str = []
					out_str.append(cluster_id)
					out_str.append(doc_name)
					sen_id = doc.token2sentence[doc.mention2token[mention][0]][0]
					sen_surface = ''
					for token in doc.sentences[sen_id]:
						sen_surface += token[0] + ' '
					out_str.append(sen_surface)

					mention_surface = ''
					for t_id in doc.mention2token[mention]:
						# print(doc.tokens[t_id] + " ")
						mention_surface += doc.tokens[t_id] + " "
					mention_surface = mention_surface.strip()
					out_str.append(mention_surface)
					# out_str += '\t'

					# out_str.append(doc.cluster_instance_id[key])

					surface = ''
					srl_dict = srl_result[topic][doc_name][str(sen_id)]
					for verb in srl_dict['verbs']:
						if verb['verb'] == mention_surface or mention_surface in verb['verb'] or verb[
							'verb'] in mention_surface:
							for idx, tag in enumerate(verb['tags']):
								if tag.startswith('B-'):
									if surface != '':
										surface += '; '
									surface += tag[2:] + ' : '
								if tag != 'O':
									surface += srl_dict['words'][idx] + ' '
							out_str.append(surface)

					writer.writerow(out_str)
			# writer.writerow([''])


def cross_sub_topics_link_experiment():
	clusters_doc_list = {}

	for topic in data:
		for doc_name in data[topic]:
			doc = data[topic][doc_name]
			for key in doc.clusters:
				cluster_id = doc.cluster_instance_id[key]
				if cluster_id not in clusters_doc_list:
					clusters_doc_list[cluster_id] = []
				clusters_doc_list[cluster_id].append(doc_name)

	print("Total clusters: %d \n" % len(clusters_doc_list.keys()))

	cross_clusters = 0
	for key in clusters_doc_list:
		is_plus = 0
		is_ecb = 0
		for doc in clusters_doc_list[key]:
			if 'plus' in doc:
				is_plus = 1
			else:
				is_ecb = 1
		if is_plus and is_ecb:
			cross_clusters += 1

	print("Clusters across ecb and plus: %d \n" % cross_clusters)


def Run_SRL(isProcessed):
	if isProcessed:
		infile = open("data/preprocessed/srl_result.json")
		srl_result = json.load(infile)
	else:
		srl_result = {}
		predictor = Predictor.from_path(
			"https://s3-us-west-2.amazonaws.com/allennlp/models/bert-base-srl-2019.06.17.tar.gz")
		for topic in data:
			print(topic)
			srl_result[topic] = {}
			for doc_name in data[topic]:
				srl_result[topic][doc_name] = {}
				doc = data[topic][doc_name]
				for sen_id in doc.sentences:
					sen_surface = ''
					for token in doc.sentences[sen_id].tokens:
						sen_surface += token[0] + ' '
					srl_result[topic][doc_name][sen_id] = predictor.predict(sentence=sen_surface)
		outfile = open("srl_result.json", 'w')
		json.dump(srl_result, outfile)
	return srl_result


def Run_Nominal_SRL(split):

	infile = open("data/preprocessed/nominal_srl_output."+split)

	nominal = []
	result = {}
	for line in infile:
		nominal.append(json.loads(line))
	idx = 0
	for topic in data:
		result[topic] = {}
		for doc_name in data[topic]:
			result[topic][doc_name] = {}
			doc = data[topic][doc_name]
			for sen_id in doc.sentences:
				result[topic][doc_name][sen_id] = nominal[idx]
				idx += 1
	return result


def Run_POS(isProcessed):
	if isProcessed:
		infile = open("pos_result.json")
		pos_result = json.load(infile)
	else:
		pos_result = {}
		nlp = spacy.load("en_core_web_sm")
		for topic in data:
			# print(topic)
			pos_result[topic] = {}
			for doc_name in data[topic]:
				pos_result[topic][doc_name] = {}
				doc = data[topic][doc_name]
				for sen_id in doc.sentences:
					sen_surface = ''
					for token in doc.sentences[sen_id].tokens:
						sen_surface += token[0] + ' '
					spacy_doc = nlp(sen_surface)
					pos_list = []
					for token in spacy_doc:
						pos_list.append(token.pos_)
					pos_result[topic][doc_name][sen_id] = pos_list
		outfile = open("pos_result.json", 'w')
		json.dump(pos_result, outfile)
	return pos_result


def ReadValidateSentence():
	validated_sentences = np.genfromtxt('../../data/ECB+/ECBplus_coreference_sentences.csv',
	                                    delimiter=',', dtype=np.str, skip_header=1)
	for topic, doc, sentence in validated_sentences:
		if topic not in validate_sen:
			validate_sen[topic] = {}
		doc_name = topic + '_' + doc + '.xml'
		if doc_name not in validate_sen[topic]:
			validate_sen[topic][doc_name] = []
		validate_sen[topic][doc_name].append(sentence)


if __name__ == "__main__":
	path = '../../data/ECB+/ECB+'
	ReadValidateSentence()
	split = 'test'
	ReadData(path, split)
	print("Reading Data Finished")

	GeneratePairs(split)
	# GenerateEntityPairs(split)

	# outfile = open('data/preprocessed/ecb+.'+split, 'wb')
	# pickle.dump(data, outfile)

	# Run_Nominal_SRL('train')