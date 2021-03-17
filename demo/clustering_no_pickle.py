import os
import gc
import sys
import json
import subprocess

import _pickle as cPickle
import logging
import argparse

import torch
from transformers import *
from operator import itemgetter
import collections



from classes import *

import operator

entailment_score = {}
topic_event_list = {}
coref_chain = {}

class Cluster(object):
	'''
	A class represents a coreference cluster
	'''
	def __init__(self, is_event):
		self.cluster_id = 0
		self.mentions = {}  # mention's dictionary, key is a mention id and value is a Mention object (either event or entity)
		self.is_event = is_event
		self.merged = False
		self.lex_vec = None
		self.arg0_vec = None
		self.arg1_vec = None
		self.loc_vec = None
		self.time_vec = None

	def __repr__(self):
		mentions_strings = []
		for mention in self.mentions.values():
			mentions_strings.append('{}_{}'.format(mention.gold_cluster, mention.mention_id))
		return str(mentions_strings)

	def __str__(self):
		mentions_strings = []
		for mention in self.mentions.values():
			mentions_strings.append('{}_{}'.format(mention.gold_cluster, mention.mention_id))
		return str(mentions_strings)

	def get_mentions_str_list(self):
		'''
		Returns a list contains the strings of all mentions in the cluster
		:return:
		'''
		mentions_strings = []
		for mention in self.mentions.values():
			mentions_strings.append(mention.mention_str)
		return mentions_strings


def mention_pair_scorer(mention1, mention2, cos):

	if mention1 not in entailment_score or mention2 not in entailment_score[mention1]:
		score = -1
	else:
		score = 1-entailment_score[mention1][mention2]

	return score


def cluster_pair_to_mention_pair(pair):
	mention_pairs = []
	cluster_1 = pair[0]
	cluster_2 = pair[1]

	c1_mentions = cluster_1.mentions.values()
	c2_mentions = cluster_2.mentions.values()

	for mention_1 in c1_mentions:
		for mention_2 in c2_mentions:
			mention_pairs.append((mention_1, mention_2))

	return mention_pairs


def merge_cluster(clusters, cluster_pairs, epoch, topics_counter,
		  topics_num, threshold, is_event):
	'''
	Merges cluster pairs in agglomerative manner till it reaches a pre-defined threshold. In each step, the function merges
	cluster pair with the highest score, and updates the candidate cluster pairs according to the
	current merge.
	Note that all Cluster objects in clusters should have the same type (event or entity but
	not both of them).
	other_clusters are fixed during merges and should have the opposite type
	i.e. if clusters are event clusters, so other_clusters will be the entity clusters.

	:param clusters: a list of Cluster objects of the same type (event/entity)
	:param cluster_pairs: a list of the cluster pairs (tuples)
	:param epoch: current epoch (relevant to training)
	:param topics_counter: current topic number
	:param topics_num: total number of topics
	:param threshold: merging threshold
	:param is_event: True if clusters are event clusters and false if they are entity clusters
	'''
	# print('Initialize cluster pairs scores... ')
	# logging.info('Initialize cluster pairs scores... ')
	# initializes the pairs-scores dict
	pairs_dict = {}
	mode = 'event' if is_event else 'entity'
	# init the scores (that the model assigns to the pairs)
	cos = torch.nn.CosineSimilarity(dim=0)
	# alpha = config_dict["alpha"]

	for cluster_pair in cluster_pairs:
		mention_pairs = cluster_pair_to_mention_pair(cluster_pair)
		pairs_num = len(mention_pairs)
		mention_score = 0.0
		for mention1, mention2 in mention_pairs:
			# mention_score += mention_pair_scorer(mention1, mention2, alpha, cos).data.cpu().numpy()
			score = mention_pair_scorer(mention1, mention2, cos)
			if score == -1:
				pairs_num -= 1
			else:
				mention_score += score
		if pairs_num == 0:
			pairs_dict[cluster_pair] = 0.0
		else:
			pairs_dict[cluster_pair] = (mention_score / pairs_num)

	while True:
		if len(pairs_dict) < 2:
			break
		# print(pairs_dict)
		(max_pair, max_score) = max(pairs_dict.items(), key=operator.itemgetter(1))
		# max_pair, max_score = key_with_max_val(pairs_dict)

		if max_score > threshold:


			cluster_i = max_pair[0]
			cluster_j = max_pair[1]
			new_cluster = Cluster(is_event)
			new_cluster.mentions.update(cluster_j.mentions)
			new_cluster.mentions.update(cluster_i.mentions)

			keys_pairs_dict = list(pairs_dict.keys())
			for pair in keys_pairs_dict:
				cluster_pair = (pair[0], pair[1])
				if cluster_i in cluster_pair or cluster_j in cluster_pair:
					del pairs_dict[pair]

			clusters.remove(cluster_i)
			clusters.remove(cluster_j)
			clusters.append(new_cluster)

			new_pairs = []
			for cluster in clusters:
				if cluster != new_cluster:
					new_pairs.append((cluster, new_cluster))

			# create scores for the new pairs

			for pair in new_pairs:
				mention_pairs = cluster_pair_to_mention_pair(pair)
				pairs_num = len(mention_pairs)
				mention_score = 0.0
				for mention1, mention2 in mention_pairs:
					score = mention_pair_scorer(mention1, mention2, cos)
					if score == -1:
						pairs_num -= 1
					else:
						mention_score += score
				if pairs_num == 0:
					pairs_dict[pair] = 0.0
				else:
					pairs_dict[pair] = (mention_score / pairs_num)

		else:
			break


def init_cd(mentions, is_event):
	'''
	Initialize a set of Mention objects (either EventMention or EntityMention) to a set of
	cross-document singleton clusters (a cluster which contains a single mentions).
	:param mentions:  a set of Mention objects (either EventMention or EntityMention)
	:param is_event: whether the mentions are event or entity mentions.
	:return: a list contains initial cross-document singleton clusters.
	'''
	clusters = []
	for mention in mentions:
		cluster = Cluster(is_event=is_event)
		cluster.mentions[mention] = mention
		clusters.append(cluster)
	return clusters


def calc_q(cluster_1, cluster_2):
	'''
	Calculates the quality of merging two clusters, denotes by the proportion between
	the number gold coreferrential mention pairwise links (between the two clusters) and all the
	pairwise links.
	:param cluster_1: first cluster
	:param cluster_2: second cluster
	:return: the quality of merge (a number between 0 to 1)
	'''
	true_pairs = 0
	false_pairs = 0
	for mention_c1 in cluster_1.mentions.values():
		for mention_c2 in cluster_2.mentions.values():
			if mention_c1.gold_tag == mention_c2.gold_tag:
				true_pairs += 1
			else:
				false_pairs += 1

	return true_pairs/float(true_pairs + false_pairs)


def generate_cluster_pairs(clusters, is_train):
	'''
	Given list of clusters, this function generates candidate cluster pairs (for training/inference).
	The function under-samples cluster pairs without any coreference links
	 when generating cluster pairs for training and the current number of clusters in the
	current topic is larger than 300.
	:param clusters: current clusters
	:param is_train: True if the function generates candidate cluster pairs for training time
	and False, for inference time (without under-sampling)
	:return: pairs - generated cluster pairs (potentially with under-sampling)
	, test_pairs -  all cluster pairs
	'''

	positive_pairs_count = 0
	negative_pairs_count = 0
	pairs = []
	test_pairs = []

	use_under_sampling = True if (len(clusters) > 300 and is_train) else False

	if len(clusters) < 500:
		p = 0.7
	else:
		p = 0.6

	# print('Generating cluster pairs...')
	# logging.info('Generating cluster pairs...')

	# print('Initial number of clusters = {}'.format(len(clusters)))
	# logging.info('Initial number of clusters = {}'.format(len(clusters)))

	# if use_under_sampling:
		# print('Using under sampling with p = {}'.format(p))
		# logging.info('Using under sampling with p = {}'.format(p))

	for cluster_1 in clusters:
		for cluster_2 in clusters:
			if cluster_1 != cluster_2:
				if is_train:
					q = calc_q(cluster_1, cluster_2)
					if (cluster_1, cluster_2, q) \
							not in pairs and (cluster_2, cluster_1, q) not in pairs:
						add_to_training = False if use_under_sampling else True
						if q > 0:
							add_to_training = True
							positive_pairs_count += 1
						if q == 0 and random.random() < p:
							add_to_training = True
							negative_pairs_count += 1
						if add_to_training:
							pairs.append((cluster_1, cluster_2, q))
						test_pairs.append((cluster_1, cluster_2))
				else:
					if (cluster_1, cluster_2) not in pairs and \
							(cluster_2, cluster_1) not in pairs:
						pairs.append((cluster_1, cluster_2))

	# print('Number of generated cluster pairs = {}'.format(len(pairs)))
	# logging.info('Number of generated cluster pairs = {}'.format(len(pairs)))

	return pairs, test_pairs


def set_coref_chain_to_mentions(clusters, is_event, is_gold, intersect_with_gold,):
	'''
	Sets the predicted cluster id to all mentions in the cluster
	:param clusters: predicted clusters (a list of Corpus objects)
	:param is_event: True, if clusters are event clusters, False otherwise - currently unused.
	:param is_gold: True, if the function sets gold mentions and false otherwise
	 (it sets predicted mentions) - currently unused.
	:param intersect_with_gold: True, if the function sets predicted mentions that were matched
	with gold mentions (used in setting that requires to match predicted mentions with gold
	mentions - as in Yang's setting) , and false otherwise - currently unused.
	:param remove_singletons: True if the function ignores singleton clusters (as in Yang's setting)
	'''
	global clusters_count
	for cluster in clusters:
		cluster.cluster_id = clusters_count
		for mention in cluster.mentions.values():
			coref_chain[mention] = clusters_count
		clusters_count += 1


def topic_to_mention_list(topic, data, is_gold, dataset):

	event_mentions = []
	entity_mentions = []
	if dataset == 'all':
		for topic_name in data:
			for doc_name in data[topic_name]:
				doc = data[topic_name][doc_name]
				for sen_id in doc.sentences.keys():
					sen = doc.sentences[sen_id]
					event_mentions.extend(sen.events)
					entity_mentions.extend(sen.entities)

	if dataset == 'ace' or dataset == 'kbp' or dataset == 'on':
		for doc_name in data[topic]:
			doc = data[topic][doc_name]
			for sen_id in doc.sentences.keys():
				sen = doc.sentences[sen_id]
				event_mentions.extend(sen.events)
				entity_mentions.extend(sen.entities)

	if dataset == 'ecb':
		for t in data:
			for doc_name in data[t]:
				# print(doc_name)
				if doc_name.replace('.xml', '') not in topic:
					continue
				doc = data[t][doc_name]
				for sen_id in doc.sentences.keys():
					# print(sen_id)
					# print(sen.events)
					# print(sen.entities)
					sen = doc.sentences[sen_id]
					event_mentions.extend(sen.events)
					entity_mentions.extend(sen.entities)
				# if is_gold:
				# 	for sen_id in doc.sentences.keys():
				# 		sen = doc.sentences[sen_id]
				# 		event_mentions.extend(sen.events)
				# 		entity_mentions.extend(sen.entities)
				# else:
				# 	event_mentions.extend(sent.pred_event_mentions)
				# 	entity_mentions.extend(sent.pred_entity_mentions)
	return event_mentions, entity_mentions


def write_span_based_cd_clusters(corpus, is_event, is_gold, out_file, use_gold_mentions):
	'''
	This function writes the predicted clusters to a file (in a CoNLL format) in a span based manner,
	means that each token is written to the file
	and the coreference chain id is marked in a parenthesis, wrapping each mention span.
	Used in any setup that requires matching of a predicted mentions with gold mentions.
	:param corpus: A Corpus object, contains the documents of each split, grouped by topics.
	:param out_file: filename of the CoNLL output file
	:param is_event: whether to write event or entity mentions
	:param is_gold: whether to write a gold-standard file (key) which contains the gold clusters
	or to write a system file (response) that contains the predicted clusters.
	:param use_gold_mentions: whether to use the gold mentions or predicted mentions
	'''

	mentions_count = 0
	out_coref = open(out_file, 'w')

	cd_coref_chain_to_id = {}
	cd_coref_chain_to_id_counter = 0

	topic_ids = {}
	topic_part_num = {}
	for id in sorted(corpus.keys()):
		doc_id, part_id = id.rsplit('_', 1)
		topic_ids[doc_id] = True
		part_id = int(part_id)
		if doc_id not in topic_part_num:
			topic_part_num[doc_id] = part_id
		if part_id > topic_part_num[doc_id]:
			topic_part_num[doc_id] = part_id

	cluster_size = {}
	for topic_id in sorted(corpus.keys()):
		cluster_size[topic_id] = {}
		cur_topic = corpus[topic_id]
		doc_keys = sorted(cur_topic.keys())
		for doc_id in doc_keys:
			cur_doc = cur_topic[doc_id]
			for sen_id in sorted(cur_doc.sentences.keys()):
				cur_sen = cur_doc.sentences[sen_id]
				mentions = cur_sen.events if is_event else cur_sen.entities
				mentions.sort(key=lambda x: x.start_offset, reverse=True)
				for mention in mentions:
					# map the gold coref tags to unique ids
					if is_gold:  # creating the key files
						if mention.gold_cluster not in cluster_size[topic_id]:
							cluster_size[topic_id][mention.gold_cluster] = 0
						cluster_size[topic_id][mention.gold_cluster] += 1
					else:  # writing the clusters at test time (response files)
						if mention.cd_coref_chain not in cluster_size[topic_id]:
							cluster_size[topic_id][mention.cd_coref_chain] = 0
						cluster_size[topic_id][mention.cd_coref_chain] += 1

	for topic in sorted(topic_ids.keys()):
		for part_id in range(topic_part_num[topic] + 1):
			topic_id = topic+'_'+str(part_id)
			# print(topic_id)
			curr_topic = corpus[topic_id]
			part_id = (3 - len(str(part_id))) * '0' + str(part_id)
			out_coref.write("#begin document (" + topic + "); part " + part_id)
			for idx in range(int(len(curr_topic.keys()))):
				doc_id = str(idx)
				curr_doc = curr_topic[doc_id]
				for sent_id in sorted(curr_doc.sentences.keys()):
					out_coref.write('\n')
					start_map = collections.defaultdict(list)
					end_map = collections.defaultdict(list)
					word_map = collections.defaultdict(list)
					curr_sent = curr_doc.sentences[sent_id]
					sent_toks = curr_sent.tokens
					sent_mentions = curr_sent.events
					for mention in sent_mentions:
						mentions_count += 1
						if is_gold:  # writing key file
							if cluster_size[topic_id][mention.gold_cluster] <= 1:
								continue
							if mention.gold_cluster not in cd_coref_chain_to_id:
								cd_coref_chain_to_id_counter += 1
								cd_coref_chain_to_id[mention.gold_cluster] = cd_coref_chain_to_id_counter
							coref_chain = cd_coref_chain_to_id[mention.gold_cluster]
						else:  # writing response file
							if cluster_size[topic_id][mention.cd_coref_chain] <= 1:
								continue
							coref_chain = mention.cd_coref_chain

						start = mention.start_offset
						end = mention.end_offset

						if start == end:
							word_map[start].append(coref_chain)
						else:
							start_map[start].append((coref_chain, end))
							end_map[end].append((coref_chain, start))

					for k, v in start_map.items():
						start_map[k] = [cluster_id for cluster_id, end in
										sorted(v, key=operator.itemgetter(1), reverse=True)]
					for k, v in end_map.items():
						end_map[k] = [cluster_id for cluster_id, start in
									  sorted(v, key=operator.itemgetter(1), reverse=True)]

					for tok_idx, tok in enumerate(sent_toks):
						word_index = tok_idx
						coref_list = []
						if word_index in end_map:
							for coref_chain in end_map[word_index]:
								coref_list.append("{})".format(coref_chain))
						if word_index in word_map:
							for coref_chain in word_map[word_index]:
								coref_list.append("({})".format(coref_chain))
						if word_index in start_map:
							for coref_chain in start_map[word_index]:
								coref_list.append("({}".format(coref_chain))

						if len(coref_list) == 0:
							token_tag = "-"
						else:
							token_tag = "|".join(coref_list)

						out_coref.write('\t'.join([topic_id, '0', str(tok_idx), tok, token_tag]) + '\n')

			out_coref.write("#end document\n")
			out_coref.write('\n')

	out_coref.close()
	# logger.info('{} mentions have been written.'.format(mentions_count))
	# print('{} mentions have been written.'.format(mentions_count))


def write_mention_based_cd_clusters(corpus, is_event, is_gold, out_file, dataset):
	'''
	This function writes the cross-document (CD) predicted clusters to a file (in a CoNLL format)
	in a mention based manner, means that each token represents a mention and its coreference chain id is marked
	in a parenthesis.
	Used in Cybulska setup, when gold mentions are used during evaluation and there is no need
	to match predicted mention with a gold one.
	:param corpus: A Corpus object, contains the documents of each split, grouped by topics.
	:param out_file: filename of the CoNLL output file
	:param is_event: whether to write event or entity mentions
	:param is_gold: whether to write a gold-standard file (key) which contains the gold clusters
	or to write a system file (response) that contains the predicted clusters.
	'''
	out_coref = open(out_file, 'w')
	cd_coref_chain_to_id = {}
	cd_coref_chain_to_id_counter = 0

	if dataset == 'ecb':
		generic = 'ECB+/ecbplus_all'
	else:
		generic = dataset

	if dataset == 'ecb':
		out_coref.write("#begin document (" + generic + "); part 000" + '\n')

	cluster_size = {}
	for topic_id in sorted(corpus.keys()):
		cluster_size[topic_id] = {}
		cur_topic = corpus[topic_id]
		doc_keys = sorted(cur_topic.keys())
		for doc_id in doc_keys:
			cur_doc = cur_topic[doc_id]
			for sen_id in sorted(cur_doc.sentences.keys()):
				cur_sen = cur_doc.sentences[sen_id]
				mentions = cur_sen.events if is_event else cur_sen.entities
				mentions.sort(key=lambda x: x.start_offset, reverse=True)
				for mention in mentions:
					# map the gold coref tags to unique ids
					if is_gold:  # creating the key files
						if mention.gold_cluster not in cluster_size[topic_id]:
							cluster_size[topic_id][mention.gold_cluster] = 0
						cluster_size[topic_id][mention.gold_cluster] += 1
					else:  # writing the clusters at test time (response files)
						if mention.cd_coref_chain not in cluster_size[topic_id]:
							cluster_size[topic_id][mention.cd_coref_chain] = 0
						cluster_size[topic_id][mention.cd_coref_chain] += 1

	for topic_id in sorted(corpus.keys()):
		if dataset != 'ecb':
			out_coref.write("#begin document (" + topic_id + "); part 000" + '\n')
		cur_topic = corpus[topic_id]
		doc_keys = sorted(cur_topic.keys())
		for doc_id in doc_keys:
			if 'plus' in doc_id:
				continue
			cur_doc = cur_topic[doc_id]
			for sen_id in sorted(cur_doc.sentences.keys()):
				cur_sen = cur_doc.sentences[sen_id]
				mentions = cur_sen.events if is_event else cur_sen.entities
				mentions.sort(key=lambda x: x.start_offset, reverse=True)
				for mention in mentions:
					# map the gold coref tags to unique ids
					if is_gold:  # creating the key files
						if mention.gold_cluster not in cd_coref_chain_to_id:
							cd_coref_chain_to_id_counter += 1
							cd_coref_chain_to_id[mention.gold_cluster] = cd_coref_chain_to_id_counter
						coref_chain = cd_coref_chain_to_id[mention.gold_cluster]
						# if cluster_size[topic_id][mention.gold_cluster] > 1:
						# 	coref_chain = cd_coref_chain_to_id[mention.gold_cluster]
						# else:
						# 	coref_chain = '-'
					else:  # writing the clusters at test time (response files)
						# if cluster_size[topic_id][mention.cd_coref_chain] > 1:
						# 	coref_chain = mention.cd_coref_chain
						# else:
						# 	coref_chain = '-'
						coref_chain = mention.cd_coref_chain
					if dataset == 'ecb':
						out_coref.write('{}\t({})\n'.format(generic, coref_chain))
					else:
						out_coref.write('{}\t({})\n'.format(topic_id, coref_chain))
		if dataset != 'ecb':
			out_coref.write('#end document\n')

	for topic_id in sorted(corpus.keys()):
		cur_topic = corpus[topic_id]
		doc_keys = sorted(cur_topic.keys())
		for doc_id in doc_keys:
			if 'plus' not in doc_id:
				continue
			cur_doc = cur_topic[doc_id]
			for sen_id in sorted(cur_doc.sentences.keys()):
				cur_sen = cur_doc.sentences[sen_id]
				mentions = cur_sen.events if is_event else cur_sen.entities
				mentions.sort(key=lambda x: x.start_offset, reverse=True)
				for mention in mentions:
					# map the gold coref tags to unique ids
					if is_gold:  # creating the key files
						if mention.gold_cluster not in cd_coref_chain_to_id:
							cd_coref_chain_to_id_counter += 1
							cd_coref_chain_to_id[mention.gold_cluster] = cd_coref_chain_to_id_counter
						coref_chain = cd_coref_chain_to_id[mention.gold_cluster]
					else:  # writing the clusters at test time (response files)
						coref_chain = mention.cd_coref_chain
					out_coref.write('{}\t({})\n'.format(generic,coref_chain))

	if dataset == 'ecb':
		out_coref.write('#end document\n')
	out_coref.close()


def write_event_coref_results(corpus, out_dir, isGold, dataset):

	# if not isGold:
	#     out_file = os.path.join(out_dir, 'CD_test_event_span_based.response_conll')
	#     write_span_based_cd_coref_clusters(corpus, out_file, is_event=True, is_gold=False,
	#                                        use_gold_mentions=config_dict["test_use_gold_mentions"])
	# else:
	#     out_file = os.path.join(out_dir, 'CD_test_event_mention_based.response_conll')
	#     write_mention_based_cd_clusters(corpus, is_event=True, is_gold=False, out_file=out_file)
	#
	#     out_file = os.path.join(out_dir, 'WD_test_event_mention_based.response_conll')
	#     write_mention_based_wd_clusters(corpus, is_event=True, is_gold=False, out_file=out_file)

	# out_file = os.path.join('data/preprocessed/gold/', 'ECB_CD_test_entity_mention_based.key_conll')
	# write_mention_based_cd_clusters(corpus, is_event=False, is_gold=True, out_file=out_file, dataset=dataset)

	if dataset == 'on':
		out_file = os.path.join(out_dir, 'CD_test_event_span_based.response_conll')
		write_span_based_cd_clusters(corpus, is_event=True, is_gold=False, out_file=out_file, use_gold_mentions=True)

	out_file = os.path.join(out_dir, 'CD_test_event_mention_based.response_conll')
	# write_mention_based_cd_clusters(corpus, is_event=True, is_gold=False, out_file=out_file, dataset=dataset)
	write_mention_based_cd_clusters(corpus, is_event=True, is_gold=False, out_file=out_file, dataset=dataset)




def test_models(write_clusters, out_dir, isProcessed, threshold):
	'''
	Runs the inference procedure for both event and entity models calculates the B-cubed
	score of their predictions.
	:param test_set: Corpus object containing the test documents.
	:param write_clusters: whether to write predicted clusters to file (for analysis purpose)
	:param out_dir: output files directory
	chains predicted by an external (WD) entity coreference system.
	:return: B-cubed scores for the predicted event and entity clusters
	'''

	global clusters_count
	clusters_count = 1


	topics_counter = 0
	epoch = 0 #
	all_event_mentions = []
	all_entity_mentions = []

	with torch.no_grad():
		for topic_id, topic in enumerate(topic_event_list.keys()):

			topics_counter += 1

			event_mentions = list(topic_event_list[topic])
			all_event_mentions.extend(event_mentions)

			topic_event_clusters = init_cd(event_mentions, is_event=True)

			cluster_pairs, _ = generate_cluster_pairs(topic_event_clusters, is_train=False)
			merge_cluster(topic_event_clusters, cluster_pairs, epoch, topics_counter, 0, threshold, True)

			# cluster_pairs, _ = generate_cluster_pairs(topic_entity_clusters, is_train=False)
			# merge_cluster(topic_entity_clusters, cluster_pairs, epoch, topics_counter, topics_num, threshold, alpha,
			#               True)

			set_coref_chain_to_mentions(topic_event_clusters, is_event=True, is_gold=True, intersect_with_gold=True)

		if write_clusters:
			# write_event_coref_results(test_set, out_dir, True, dataset)
			clusters = {}

			for mention in all_event_mentions:
				if coref_chain[mention] not in clusters:
					clusters[coref_chain[mention]] = []
				clusters[coref_chain[mention]].append(mention)

			# outfile = open('out/clusters.out', 'w')
			outfile = open('clusters.out', 'w')
			for cluster in clusters.values():
				outfile.write('\t'.join(cluster))
				outfile.write('\n')

			# infile = open('data/sample.ta', 'r')
			# ta = json.load(infile)
			#
			# # mentions_dict = {}
			# for v_idx, view in enumerate(ta['views']):
			# 	if view['viewName'] == 'Event_extraction':
			# 		# mentions_dict = view['viewData'][0]['relations']
			# 		break
			#
			# print(clusters)
			#
			# for cluster in clusters.values():
			# 	for i, event_i in enumerate(cluster):
			# 		for j, event_j in enumerate(cluster):
			# 			if i >= j:
			# 				break
			# 			ta['views'][v_idx]['viewData'][0]['relations'].append(
			# 				{'relationName': 'coref', 'srcConstituent':event_i.mention_id.split('_')[-1],
			# 			     'targetConstituent':event_j.mention_id.split('_')[-1]})
			# outfile = open('out/sample.ta', 'w')
			# json.dump(outfile, ta)
				# print(cluster)
				# outfile.write('\t'.join(cluster))
				# outfile.write('\n')
	# print(ta)
	# return ta

def test_models_demo(ta, write_clusters, out_dir, isProcessed, threshold):
	'''
	Runs the inference procedure for both event and entity models calculates the B-cubed
	score of their predictions.
	:param test_set: Corpus object containing the test documents.
	:param write_clusters: whether to write predicted clusters to file (for analysis purpose)
	:param out_dir: output files directory
	chains predicted by an external (WD) entity coreference system.
	:return: B-cubed scores for the predicted event and entity clusters
	'''

	global clusters_count
	clusters_count = 1


	topics_counter = 0
	epoch = 0 #
	all_event_mentions = []
	all_entity_mentions = []

	with torch.no_grad():
		for topic_id, topic in enumerate(topic_event_list.keys()):

			topics_counter += 1

			event_mentions = list(topic_event_list[topic])
			all_event_mentions.extend(event_mentions)

			topic_event_clusters = init_cd(event_mentions, is_event=True)

			cluster_pairs, _ = generate_cluster_pairs(topic_event_clusters, is_train=False)
			merge_cluster(topic_event_clusters, cluster_pairs, epoch, topics_counter, 0, threshold, True)

			# cluster_pairs, _ = generate_cluster_pairs(topic_entity_clusters, is_train=False)
			# merge_cluster(topic_entity_clusters, cluster_pairs, epoch, topics_counter, topics_num, threshold, alpha,
			#               True)

			set_coref_chain_to_mentions(topic_event_clusters, is_event=True, is_gold=True, intersect_with_gold=True)

		if write_clusters:
			# write_event_coref_results(test_set, out_dir, True, dataset)
			clusters = {}

			for mention in all_event_mentions:
				if coref_chain[mention] not in clusters:
					clusters[coref_chain[mention]] = []
				clusters[coref_chain[mention]].append(mention)

			# outfile = open('out/clusters.out', 'w')
			outfile = open('clusters.out', 'w')
			for cluster in clusters.values():
				outfile.write('\t'.join(cluster))
				outfile.write('\n')
			outfile.close()


			# mentions_dict = {}
			for v_idx, view in enumerate(ta['views']):
				if view['viewName'] == 'Event_extraction':
					# mentions_dict = view['viewData'][0]['relations']
					break

			# print(clusters)

			for cluster in clusters.values():
				for i, event_i in enumerate(cluster):
					for j, event_j in enumerate(cluster):
						if i >= j:
							continue

						ta['views'][v_idx]['viewData'][0]['relations'].append(
							{'relationName': 'coref', 'srcConstituent':event_i.split('_')[-1],
						     'targetConstituent':event_j.split('_')[-1]})

			# outfile = open('out/sample.ta', 'w')
			# json.dump(outfile, ta)
			# 	print(cluster)
			# 	outfile.write('\t'.join(cluster))
			# 	outfile.write('\n')
	# print(ta)
	return ta


def read_conll_f1(filename):
	'''
	This function reads the results of the CoNLL scorer , extracts the F1 measures of the MUS,
	B-cubed and the CEAF-e and calculates CoNLL F1 score.
	:param filename: a file stores the scorer's results.
	:return: the CoNLL F1
	'''
	f1_list = []
	with open(filename, "r") as ins:
		for line in ins:
			new_line = line.strip()
			if new_line.find('F1:') != -1:
				f1_list.append(float(new_line.split(': ')[-1][:-1]))

	muc_f1 = f1_list[1]
	bcued_f1 = f1_list[3]
	ceafe_f1 = f1_list[7]

	return (muc_f1 + bcued_f1 + ceafe_f1)/float(3)


def run_conll_scorer(isGold, gold_file_path, out_dir):

	if isGold:
		event_response_filename = os.path.join(out_dir, 'CD_test_event_mention_based.response_conll')
		# entity_response_filename = os.path.join(args.out_dir, 'CD_test_entity_mention_based.response_conll')
	else:
		event_response_filename = os.path.join(out_dir, 'CD_test_event_span_based.response_conll')
		# entity_response_filename = os.path.join(args.out_dir, 'CD_test_entity_span_based.response_conll')

	event_conll_file = os.path.join(out_dir,'event_scorer_cd_out.txt')
	# entity_conll_file = os.path.join(args.out_dir,'entity_scorer_cd_out.txt')

	event_scorer_command = ('perl scorer/scorer.pl all {} {} none > {} \n'.format
			(gold_file_path, event_response_filename, event_conll_file))

	# entity_scorer_command = ('perl scorer/scorer.pl all {} {} none > {} \n'.format
	#         (config_dict["entity_gold_file_path"], entity_response_filename, entity_conll_file))

	processes = []
	# print('Run scorer command for cross-document event coreference: {} \n'.format(event_scorer_command))
	processes.append(subprocess.Popen(event_scorer_command, shell=True))

	# print('Run scorer command for cross-document entity coreference')
	# processes.append(subprocess.Popen(entity_scorer_command, shell=True))

	while processes:
		status = processes[0].poll()
		if status is not None:
			processes.pop(0)

	print ('Running scorers has been done.')
	print ('Save results...')

	scores_file = open(os.path.join(out_dir, 'conll_f1_scores.txt'), 'w')

	event_f1 = read_conll_f1(event_conll_file)
	# entity_f1 = read_conll_f1(entity_conll_file)
	scores_file.write('Event CoNLL F1: {}\n'.format(event_f1))
	# scores_file.write('Entity CoNLL F1: {}\n'.format(entity_f1))

	scores_file.close()
	return event_f1

def demo_cluster(ta):

	logger.info('Test data have been loaded.')

	threshold = 0.25

	global entailment_score, topic_event_list, coref_chain
	entailment_score = {}
	topic_event_list = {}
	coref_chain = {}

	with open('test_scores.txt', 'r') as entailment_file:

		for line in entailment_file:
			mention1, mention2, score = line.replace('\n', '').split('\t')

			if mention1 not in entailment_score:
				entailment_score[mention1] = {}
			if mention2 not in entailment_score:
				entailment_score[mention2] = {}

			score = float(score)

			entailment_score[mention1][mention2] = float(score)
			entailment_score[mention2][mention1] = float(score)

			topic1 = mention1.split('_')[0]
			if topic1 not in topic_event_list:
				topic_event_list[topic1] = set()
			topic_event_list[topic1].add(mention1)
			topic2 = mention2.split('_')[0]
			if topic2 not in topic_event_list:
				topic_event_list[topic2] = set()
			topic_event_list[topic2].add(mention2)

		entailment_file.close()

	return test_models_demo(ta, write_clusters=True, out_dir='out/', isProcessed=True, threshold=threshold)


# test_models_for_heng('all', test_data, write_clusters=True, out_dir='out/', isProcessed=True, threshold=threshold)

def main():

	global entailment_score, topic_event_list, coref_chain
	entailment_score = {}
	topic_event_list = {}
	coref_chain = {}

	logger.info('Test data have been loaded.')

	threshold = 0.4

	with open('test_scores.txt', 'r') as entailment_file:

		for line in entailment_file:
			mention1, mention2, score = line.replace('\n', '').split('\t')

			if mention1 not in entailment_score:
				entailment_score[mention1] = {}
			if mention2 not in entailment_score:
				entailment_score[mention2] = {}

			score = float(score)

			entailment_score[mention1][mention2] = float(score)
			entailment_score[mention2][mention1] = float(score)

			topic1 = mention1.split('_')[0]
			if topic1 not in topic_event_list:
				topic_event_list[topic1] = set()
			topic_event_list[topic1].add(mention1)
			topic2 = mention2.split('_')[0]
			if topic2 not in topic_event_list:
				topic_event_list[topic2] = set()
			topic_event_list[topic2].add(mention2)

		entailment_file.close()
	print(entailment_score)
	# test_models(write_clusters=True, out_dir='out/', isProcessed=True, threshold=threshold)
	# test_models_for_heng('all', test_data, write_clusters=True, out_dir='out/', isProcessed=True, threshold=threshold)
	data_path = 'data/sample.ta'
	infile = open(data_path, 'r')
	ta = json.load(infile)
	test_models_demo(ta, write_clusters=True, out_dir='out/', isProcessed=True, threshold=threshold)


if __name__ == '__main__':
	logging.basicConfig(level=logging.DEBUG)
	logger = logging.getLogger(__name__)
	main()
