from __future__ import absolute_import, division, print_function

import argparse
import csv
import logging
import os
import random
import sys
import codecs
import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from scipy.stats import beta
from torch.nn import CrossEntropyLoss, MSELoss
from scipy.special import softmax
# from scipy.stats import pearsonr, spearmanr
# from sklearn.metrics import matthews_corrcoef, f1_score

from transformers.tokenization_roberta import RobertaTokenizer
from transformers.optimization import AdamW
from transformers.modeling_roberta import RobertaModel  # RobertaForSequenceClassification
from transformers import XLMRobertaTokenizer, XLMRobertaModel


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

# from pytorch_transformers.modeling_bert import BertPreTrainedModel, BertModel
# import torch.nn as nn

bert_hidden_dim = 1024
pretrain_model_dir = 'roberta-large'  # 'roberta-large' , 'roberta-large-mnli', 'bert-large-uncased'


def store_transformers_models(model, tokenizer, output_dir, flag_str):
	'''
	store the model
	'''
	output_dir += '/' + flag_str
	# if not os.path.exists(output_dir):
	#     os.makedirs(output_dir)
	print('starting model storing....')
	# model.save_pretrained(output_dir)
	torch.save(model.state_dict(), output_dir)
	# tokenizer.save_pretrained(output_dir)
	print('store succeed')


class RobertaForSequenceClassification(nn.Module):
	def __init__(self, tagset_size):
		super(RobertaForSequenceClassification, self).__init__()
		self.tagset_size = tagset_size

		self.roberta_single = RobertaModel.from_pretrained(pretrain_model_dir)
		# self.roberta_single = XLMRobertaModel.from_pretrained("xlm-roberta-large")
		self.hidden_layer_0 = nn.Linear(bert_hidden_dim * 3, bert_hidden_dim)
		self.hidden_layer_1 = nn.Linear(bert_hidden_dim, bert_hidden_dim)
		self.hidden_layer_2 = nn.Linear(bert_hidden_dim, bert_hidden_dim)
		self.single_hidden2tag = RobertaClassificationHead(bert_hidden_dim, tagset_size)


	def forward(self, input_ids, input_mask, span_a_mask, span_b_mask):

		outputs_single = self.roberta_single(input_ids, input_mask, None)
		output_last_layer_tensor3 = outputs_single[0]  # (batch_size, sequence_length, hidden_size)`)
		span_a_reps = torch.sum(output_last_layer_tensor3 * span_a_mask.unsqueeze(2), dim=1)  # (batch, hidden)
		span_b_reps = torch.sum(output_last_layer_tensor3 * span_b_mask.unsqueeze(2), dim=1)  # (batch, hidden)

		combined_rep = torch.cat([span_a_reps, span_b_reps, span_a_reps * span_b_reps], dim=1)  # (batch, 3*hidden)
		MLP_input = torch.tanh(self.hidden_layer_0(combined_rep))  # (batch, hidden)

		hidden_states_single = torch.tanh(
			self.hidden_layer_2(torch.tanh(self.hidden_layer_1(MLP_input))))  # (batch, hidden)

		score_single = self.single_hidden2tag(hidden_states_single)  # (batch, tag_set)
		return score_single


class RobertaClassificationHead(nn.Module):
	"""wenpeng overwrite it so to accept matrix as input"""

	def __init__(self, bert_hidden_dim, num_labels):
		super(RobertaClassificationHead, self).__init__()
		# self.dense = nn.Linear(bert_hidden_dim, bert_hidden_dim)
		# self.dropout = nn.Dropout(0.1)
		self.out_proj = nn.Linear(bert_hidden_dim, num_labels)

	def forward(self, features):
		x = features  # [:, 0, :]  # take <s> token (equiv. to [CLS])
		# x = self.dropout(x)
		# x = self.dense(x)
		# x = torch.tanh(x)
		# x = self.dropout(x)
		x = self.out_proj(x)
		return x


class InputExample(object):
	"""A single training/test example for simple sequence classification."""

	def __init__(self, guid, text_a=None, span_a_left=None, span_a_right=None, text_b=None, span_b_left=None,
	             span_b_right=None, label=None, pair_id=None):
		"""Constructs a InputExample.
		Args:
			guid: Unique id for the example.
			text_a: string. The untokenized text of the first sequence. For single
			sequence tasks, only this sequence must be specified.
			text_b: (Optional) string. The untokenized text of the second sequence.
			Only must be specified for sequence pair tasks.
			label: (Optional) string. The label of the example. This should be
			specified for train and dev examples, but not for test examples.
		"""
		self.guid = guid
		self.text_a = text_a
		self.span_a_left = span_a_left
		self.span_a_right = span_a_right

		self.text_b = text_b
		self.span_b_left = span_b_left
		self.span_b_right = span_b_right
		self.label = label
		self.pair_id = pair_id


class InputFeatures(object):
	"""A single set of features of data."""

	def __init__(self, input_ids, input_mask, segment_ids, span_a_mask, span_b_mask,
	             label_id, pair_id):
		self.input_ids = input_ids
		self.input_mask = input_mask
		self.segment_ids = segment_ids
		self.span_a_mask = span_a_mask
		self.span_b_mask = span_b_mask

		self.label_id = label_id
		self.pair_id = pair_id


class DataProcessor(object):
	"""Base class for data converters for sequence classification data sets."""

	def get_train_examples(self, data_dir):
		"""Gets a collection of `InputExample`s for the train set."""
		raise NotImplementedError()

	def get_dev_examples(self, data_dir):
		"""Gets a collection of `InputExample`s for the dev set."""
		raise NotImplementedError()

	def get_labels(self):
		"""Gets the list of labels for this data set."""
		raise NotImplementedError()

	@classmethod
	def _read_tsv(cls, input_file, quotechar=None):
		"""Reads a tab separated value file."""
		with open(input_file, "r") as f:
			reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
			lines = []
			for line in reader:
				if sys.version_info[0] == 2:
					line = list(unicode(cell, 'utf-8') for cell in line)
				lines.append(line)
			return lines


class RteProcessor(DataProcessor):
	"""Processor for the RTE data set (GLUE version)."""

	def get_ECB_plus_NLI(self, filename, is_train=True):
		'''
		can read the training file, dev and test file
		'''
		examples = []
		readfile = open(filename, 'r')
		line_co = 0
		pos_size = 0
		for row in readfile:
			line = row.strip().split('\t')
			if len(line) == 7:
				guid = "train-" + str(line_co - 1)
				text_a = line[0].strip()
				span_a_left = int(line[1].strip())
				span_a_right = int(line[2].strip())
				text_b = line[3].strip()
				span_b_left = int(line[4].strip())
				span_b_right = int(line[5].strip())
				label = int(line[6].strip())
				if label == 1:
					pos_size += 1

				examples.append(
					InputExample(guid=guid, text_a=text_a, span_a_left=span_a_left, span_a_right=span_a_right,
					             text_b=text_b, span_b_left=span_b_left, span_b_right=span_b_right, label=label,
					             pair_id=None))
				if is_train:
					examples.append(
						InputExample(guid=guid, text_a=text_b, span_a_left=span_b_left, span_a_right=span_b_right,
						             text_b=text_a, span_b_left=span_a_left, span_b_right=span_a_right, label=label,
						             pair_id=None))
			else:
				print(row)
				print(row.split('\t'))
				print(line)
			line_co += 1
		# if line_co > 20000:
		#     break
		readfile.close()
		print('data line: ', line_co)
		print('loaded  size:', len(examples), ' pos_size:', pos_size)
		return examples

	def get_ECB_plus_NLI_unlabeled_test(self, filename):
		'''
		can read the training file, dev and test file
		'''
		examples = []
		readfile = open(filename, 'r')
		line_co = 0
		pos_size = 0
		for row in readfile:

			line = row.strip().split('\t')
			if len(line) == 9:
				guid = "test-" + str(line_co - 1)
				event_id_1 = line[0].strip()
				event_id_2 = line[1].strip()
				text_a = line[2].strip()
				span_a_left = int(line[3].strip())
				span_a_right = int(line[4].strip())
				text_b = line[5].strip()
				span_b_left = int(line[6].strip())
				span_b_right = int(line[7].strip())
				label = int(line[8].strip())
				if label == 1:
					pos_size += 1
				examples.append(
					InputExample(guid=guid, text_a=text_a, span_a_left=span_a_left, span_a_right=span_a_right,
					             text_b=text_b, span_b_left=span_b_left, span_b_right=span_b_right, label=label,
					             pair_id=event_id_1 + '&&' + event_id_2))
			else:
				print(line)
			line_co += 1
		# if line_co > 20000:
		#     break
		readfile.close()
		print('data line: ', line_co)
		print('loaded  size:', len(examples), ' pos_size:', pos_size)
		return examples

	def get_labels(self):
		'here we keep the three-way in MNLI training '
		return ["entailment", "not_entailment"]
	# return ["entailment", "neutral", "contradiction"]


def wordpairID_2_tokenpairID(sentence, wordindex_left, wordindex_right, full_token_id_list, tokenizer, sent_1=True,
                             only_1_sen=False):
	'''pls note that the input indices pair include the b in (a,b), but the output doesn't'''
	'''first find the position of [2,2]'''
	position_two_two = 0
	for i in range(len(full_token_id_list)):
		if full_token_id_list[i] == 2 and full_token_id_list[i + 1] == 2:
			position_two_two = i
			break
	span = ' '.join(sentence.split()[wordindex_left: wordindex_right + 1])
	if wordindex_left != 0:
		'''this span is the begining of the sent'''
		span = ' ' + span

	span_token_list = tokenizer.tokenize(span)
	span_id_list = tokenizer.convert_tokens_to_ids(span_token_list)
	# print('span:', span, 'span_id_list:', span_id_list)
	if sent_1:
		# for i in range(wordindex_left, len(full_token_id_list)-len(span_id_list)):
		if only_1_sen:
			position_two_two = len(full_token_id_list)
		for i in range(wordindex_left, position_two_two):
			if full_token_id_list[i:i + len(span_id_list)] == span_id_list:
				return i, i + len(span_id_list), span_token_list

		return None, None, span_token_list
	else:
		# print('position_two_two:', position_two_two)
		for i in range(position_two_two + 2, len(full_token_id_list)):
			if full_token_id_list[i:i + len(span_id_list)] == span_id_list:
				return i, i + len(span_id_list), span_token_list

		return None, None, span_token_list


def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer, output_mode,
                                 cls_token_at_end=False,
                                 cls_token='[CLS]',
                                 cls_token_segment_id=1,
                                 sep_token='[SEP]',
                                 sep_token_extra=False,
                                 pad_on_left=False,
                                 pad_token=0,
                                 pad_token_segment_id=0,
                                 sequence_a_segment_id=0,
                                 sequence_b_segment_id=1,
                                 mask_padding_with_zero=True):
	""" Loads a data file into a list of `InputBatch`s
		`cls_token_at_end` define the location of the CLS token:
			- False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
			- True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
		`cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
	"""

	label_map = {label: i for i, label in enumerate(label_list)}

	features = []
	give_up = 0
	for (ex_index, example) in enumerate(examples):
		if ex_index % 10000 == 0:
			logger.info("Writing example %d of %d" % (ex_index, len(examples)))

		tokens_a = tokenizer.tokenize(example.text_a)
		# print('tokens_a:', tokens_a)

		tokens_b = None
		if example.text_b:
			tokens_b = tokenizer.tokenize(example.text_b)
			# print('tokens_b:', tokens_b)
			# Modifies `tokens_a` and `tokens_b` in place so that the total
			# length is less than the specified length.
			# Account for [CLS], [SEP], [SEP] with "- 3". " -4" for RoBERTa.
			special_tokens_count = 4 if sep_token_extra else 3
			_truncate_seq_pair(tokens_a, tokens_b, max_seq_length - special_tokens_count)
		else:
			# Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
			special_tokens_count = 3 if sep_token_extra else 2
			if len(tokens_a) > max_seq_length - special_tokens_count:
				tokens_a = tokens_a[:(max_seq_length - special_tokens_count)]

		tokens = tokens_a + [sep_token]
		# tokens_a += [sep_token]
		if sep_token_extra:
			# roberta uses an extra separator b/w pairs of sentences
			tokens += [sep_token]
		segment_ids = [sequence_a_segment_id] * len(tokens)

		if tokens_b:
			tokens += tokens_b + [sep_token]
			segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)


		if cls_token_at_end:
			tokens = tokens + [cls_token]
			segment_ids = segment_ids + [cls_token_segment_id]
		else:
			tokens = [cls_token] + tokens
			segment_ids = [cls_token_segment_id] + segment_ids

		input_ids = tokenizer.convert_tokens_to_ids(tokens)

		input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

		padding_length = max_seq_length - len(input_ids)

		if pad_on_left:
			input_ids = ([pad_token] * padding_length) + input_ids
			input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
			segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids

		else:
			input_ids = input_ids + ([pad_token] * padding_length)
			input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
			segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)

		assert len(input_ids) == max_seq_length
		assert len(input_mask) == max_seq_length
		assert len(segment_ids) == max_seq_length

		if output_mode == "classification":
			label_id = label_map[example.label]
		elif output_mode == "regression":
			label_id = float(example.label)
		else:
			raise KeyError(output_mode)

		span_a_left, span_a_right, span_a_token_list = wordpairID_2_tokenpairID(example.text_a, example.span_a_left,
		                                                                        example.span_a_right, input_ids,
		                                                                        tokenizer, sent_1=True,
		                                                                        only_1_sen=False)

		span_b_left, span_b_right, span_b_token_list = wordpairID_2_tokenpairID(example.text_b, example.span_b_left,
		                                                                        example.span_b_right, input_ids,
		                                                                        tokenizer, sent_1=False,
		                                                                        only_1_sen=False)


		if span_a_left is None or span_b_left is None:
			'''give up this pair'''
			give_up += 1
			continue
		else:
			span_a_mask = [0] * len(input_ids)
			for i in range(span_a_left, span_a_right):
				span_a_mask[i] = 1
			span_b_mask = [0] * len(input_ids)
			for i in range(span_b_left, span_b_right):
				span_b_mask[i] = 1

			features.append(
				InputFeatures(input_ids=input_ids,
				              input_mask=input_mask,
				              segment_ids=segment_ids,
				              span_a_mask=span_a_mask,
				              span_b_mask=span_b_mask,
				              label_id=label_id,
				              pair_id=example.pair_id))
	print('input example size:', len(examples), ' give_up:', give_up, ' remain:', len(features))
	return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
	"""Truncates a sequence pair in place to the maximum length."""

	# This is a simple heuristic which will always truncate the longer sequence
	# one token at a time. This makes more sense than truncating an equal percent
	# of tokens from each, since if one sequence is very short then each token
	# that's truncated likely contains more information than a longer sequence.
	while True:
		total_length = len(tokens_a) + len(tokens_b)
		if total_length <= max_length:
			break
		if len(tokens_a) > len(tokens_b):
			tokens_a.pop()
		else:
			tokens_b.pop()


def feature2vector(feature, batch_size):
	all_input_ids = torch.tensor([f.input_ids for f in feature], dtype=torch.long)
	all_input_mask = torch.tensor([f.input_mask for f in feature], dtype=torch.long)
	all_segment_ids = torch.tensor([f.segment_ids for f in feature], dtype=torch.long)
	all_span_a_mask = torch.tensor([f.span_a_mask for f in feature], dtype=torch.long)
	all_span_b_mask = torch.tensor([f.span_b_mask for f in feature], dtype=torch.long)

	all_label_ids = torch.tensor([f.label_id for f in feature], dtype=torch.long)

	data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_span_a_mask, all_span_b_mask,
	                     all_label_ids)

	sampler = SequentialSampler(data)
	dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)
	return data, sampler, dataloader


def demo_predict(args, file):

	processors = {
		"rte": RteProcessor
	}

	output_modes = {
		"rte": "classification"
	}

	if args.local_rank == -1 or args.no_cuda:
		device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
		n_gpu = torch.cuda.device_count()
	else:
		torch.cuda.set_device(args.local_rank)
		device = torch.device("cuda", args.local_rank)
		n_gpu = 1
		# Initializes the distributed backend which will take care of sychronizing nodes/GPUs
		torch.distributed.init_process_group(backend='nccl')
	logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
		device, n_gpu, bool(args.local_rank != -1), args.fp16))

	if args.gradient_accumulation_steps < 1:
		raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
			args.gradient_accumulation_steps))

	args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

	random.seed(args.seed)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	if n_gpu > 0:
		torch.cuda.manual_seed_all(args.seed)

	if not args.do_train and not args.do_eval:
		raise ValueError("At least one of `do_train` or `do_eval` must be True.")

	task_name = args.task_name.lower()

	if task_name not in processors:
		raise ValueError("Task not found: %s" % (task_name))

	processor = processors[task_name]()
	output_mode = output_modes[task_name]

	test_examples = processor.get_ECB_plus_NLI_unlabeled_test(file)

	label_list = [0, 1]
	# train_examples = get_data_hulu_fewshot('train', 5)
	# train_examples, dev_examples, test_examples, label_list = load_CLINC150_with_specific_domain_sequence(args.DomainName, args.kshot, augment=False)
	num_labels = len(label_list)
	print('num_labels:', num_labels, 'test size:', len(test_examples))

	model = RobertaForSequenceClassification(num_labels)
	tokenizer = RobertaTokenizer.from_pretrained(pretrain_model_dir, do_lower_case=args.do_lower_case)
	# tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-large', do_lower_case=args.do_lower_case)

	# model.load_state_dict(torch.load('model_original_83/f1_0.7854086361654428.pt'))

	model.to(device)

	param_optimizer = list(model.named_parameters())
	no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
	optimizer_grouped_parameters = [
		{'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
		{'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
	]

	max_test_acc = 0.0

	if args.event:
		if args.english:
			print("Loading English Event Model.")
			model.load_state_dict(torch.load('/shared/xdyu/event_coref/EventCoref/NLI/demo/model/event/f1_0.7909405520391443.pt'))
		else:
			print("Loading Spanish Event Model.")
			model.load_state_dict(torch.load('/shared/xdyu/event_coref/EventCoref/NLI/demo/model/event_xlmr/f1_0.593213087813911.pt'))
	else:
		if args.english:
			model.load_state_dict(torch.load('/shared/xdyu/event_coref/EventCoref/NLI/demo/model/entity/f1_.886795317197986.pt'))
		else:
			model.load_state_dict(torch.load('/shared/xdyu/event_coref/EventCoref/NLI/demo/model/entity_xlmr/f1_0.4332229784922077.pt'))

	model.to(device)
	model.eval()

	test_features = convert_examples_to_features(
		test_examples, label_list, args.max_seq_length, tokenizer, output_mode,
		cls_token_at_end=False,  # bool(args.model_type in ['xlnet']),            # xlnet has a cls token at the end
		cls_token=tokenizer.cls_token,
		cls_token_segment_id=0,  # 2 if args.model_type in ['xlnet'] else 0,
		sep_token=tokenizer.sep_token,
		sep_token_extra=True,
		# bool(args.model_type in ['roberta']),           # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
		pad_on_left=False,  # bool(args.model_type in ['xlnet']),                 # pad on the left for xlnet
		pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
		pad_token_segment_id=0)  # 4 if args.model_type in ['xlnet'] else 0,)

	eval_all_pair_ids = [f.pair_id for f in test_features]

	eval_data, eval_sampler, test_dataloader = feature2vector(test_features, args.eval_batch_size)

	logger.info("***** Running test *****")
	logger.info("  Num examples = %d", len(test_features))

	eval_loss = 0
	nb_eval_steps = 0
	preds = []
	gold_label_ids = []
	# print('Evaluating...')
	for input_ids, input_mask, segment_ids, span_a_mask, span_b_mask, label_ids in test_dataloader:

		input_ids = input_ids.to(device)
		input_mask = input_mask.to(device)
		segment_ids = segment_ids.to(device)
		span_a_mask = span_a_mask.to(device)
		span_b_mask = span_b_mask.to(device)

		label_ids = label_ids.to(device)
		gold_label_ids += list(label_ids.detach().cpu().numpy())

		with torch.no_grad():
			logits = model(input_ids, input_mask, span_a_mask, span_b_mask)
		if len(preds) == 0:
			preds.append(logits.detach().cpu().numpy())
		else:
			preds[0] = np.append(preds[0], logits.detach().cpu().numpy(), axis=0)

	preds = preds[0]

	pred_probs = softmax(preds, axis=1)
	score_for_print = list(pred_probs[:, 0])
	assert len(eval_all_pair_ids) == len(score_for_print)
	pred_label_ids = list(np.argmax(pred_probs, axis=1))

	gold_label_ids = gold_label_ids
	assert len(pred_label_ids) == len(gold_label_ids)
	hit_co = 0
	for k in range(len(pred_label_ids)):
		if pred_label_ids[k] == gold_label_ids[k]:
			hit_co += 1
	test_acc = hit_co / len(gold_label_ids)

	overlap = 0
	for k in range(len(pred_label_ids)):
		if pred_label_ids[k] == gold_label_ids[k] and gold_label_ids[k] == 1:
			overlap += 1
	recall = overlap / (1e-6 + sum(gold_label_ids))
	precision = overlap / (1e-6 + sum(pred_label_ids))
	f1 = 2 * recall * precision / (1e-6 + recall + precision)

	# this is test
	if f1 > max_test_acc:
		max_test_acc = f1

	'''write new scores to test file'''
	writescore = codecs.open('test_scores.txt', 'w', 'utf-8')
	for id, score in enumerate(score_for_print):
		pair_idd = eval_all_pair_ids[id].split('&&')
		writescore.write(pair_idd[0] + '\t' + pair_idd[1] + '\t' + str(score) + '\n')
	print('test score written over')
	writescore.close()
	final_test_performance = f1
	print('\ntest:', [test_acc, f1], ' max_test_f1:', max_test_acc, '\n')


def main():
	parser = argparse.ArgumentParser()

	## Required parameters
	parser.add_argument("--data_dir",
	                    default=None,
	                    type=str,
	                    required=True,
	                    )
	parser.add_argument("--task_name",
	                    default=None,
	                    type=str,
	                    required=True,
	                    help="The name of the task to train.")
	parser.add_argument("--output_dir",
	                    default=None,
	                    type=str,
	                    required=True,
	                    help="The output directory where the model predictions and checkpoints will be written.")

	## Other parameters
	parser.add_argument("--cache_dir",
	                    default="",
	                    type=str,
	                    help="Where do you want to store the pre-trained models downloaded from s3")
	parser.add_argument("--max_seq_length",
	                    default=128,
	                    type=int,
	                    help="The maximum total input sequence length after WordPiece tokenization. \n"
	                         "Sequences longer than this will be truncated, and sequences shorter \n"
	                         "than this will be padded.")
	parser.add_argument("--do_train",
	                    action='store_true',
	                    help="Whether to run training.")

	parser.add_argument('--kshot',
	                    type=int,
	                    default=5,
	                    help="random seed for initialization")
	parser.add_argument("--do_eval",
	                    action='store_true',
	                    help="Whether to run eval on the dev set.")
	parser.add_argument("--do_lower_case",
	                    action='store_true',
	                    help="Set this flag if you are using an uncased model.")
	parser.add_argument("--train_batch_size",
	                    default=16,
	                    type=int,
	                    help="Total batch size for training.")
	parser.add_argument("--eval_batch_size",
	                    default=64,
	                    type=int,
	                    help="Total batch size for eval.")
	parser.add_argument("--learning_rate",
	                    default=1e-5,
	                    type=float,
	                    help="The initial learning rate for Adam.")
	parser.add_argument("--num_train_epochs",
	                    default=3.0,
	                    type=float,
	                    help="Total number of training epochs to perform.")
	parser.add_argument("--warmup_proportion",
	                    default=0.1,
	                    type=float,
	                    help="Proportion of training to perform linear learning rate warmup for. "
	                         "E.g., 0.1 = 10%% of training.")
	parser.add_argument("--no_cuda",
	                    action='store_true',
	                    help="Whether not to use CUDA when available")
	parser.add_argument("--use_mixup",
	                    action='store_true',
	                    help="Whether not to use CUDA when available")
	parser.add_argument("--local_rank",
	                    type=int,
	                    default=-1,
	                    help="local_rank for distributed training on gpus")
	parser.add_argument('--seed',
	                    type=int,
	                    default=42,
	                    help="random seed for initialization")
	parser.add_argument('--beta_sampling_times',
	                    type=int,
	                    default=10,
	                    help="random seed for initialization")
	parser.add_argument('--gradient_accumulation_steps',
	                    type=int,
	                    default=1,
	                    help="Number of updates steps to accumulate before performing a backward/update pass.")
	parser.add_argument('--fp16',
	                    action='store_true',
	                    help="Whether to use 16-bit float precision instead of 32-bit")
	parser.add_argument('--loss_scale',
	                    type=float, default=0,
	                    help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
	                         "0 (default value): dynamic loss scaling.\n"
	                         "Positive power of 2: static loss scaling value.\n")
	parser.add_argument('--server_ip', type=str, default='', help="Can be used for distant debugging.")
	parser.add_argument('--server_port', type=str, default='', help="Can be used for distant debugging.")

	parser.add_argument("--event",
	                    action='store_true',
	                    help="Whether to run event coref.")

	parser.add_argument("--english",
	                    action='store_true',
	                    help="Whether to run event coref.")

	args = parser.parse_args()

	processors = {
		"rte": RteProcessor
	}

	output_modes = {
		"rte": "classification"
	}

	if args.local_rank == -1 or args.no_cuda:
		device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
		n_gpu = torch.cuda.device_count()
	else:
		torch.cuda.set_device(args.local_rank)
		device = torch.device("cuda", args.local_rank)
		n_gpu = 1
		# Initializes the distributed backend which will take care of sychronizing nodes/GPUs
		torch.distributed.init_process_group(backend='nccl')
	logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
		device, n_gpu, bool(args.local_rank != -1), args.fp16))

	if args.gradient_accumulation_steps < 1:
		raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
			args.gradient_accumulation_steps))

	args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

	random.seed(args.seed)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	if n_gpu > 0:
		torch.cuda.manual_seed_all(args.seed)

	if not args.do_train and not args.do_eval:
		raise ValueError("At least one of `do_train` or `do_eval` must be True.")

	task_name = args.task_name.lower()

	if task_name not in processors:
		raise ValueError("Task not found: %s" % (task_name))

	processor = processors[task_name]()
	output_mode = output_modes[task_name]

	test_examples = processor.get_ECB_plus_NLI_unlabeled_test(args.data_dir)

	label_list = [0, 1]
	# train_examples = get_data_hulu_fewshot('train', 5)
	# train_examples, dev_examples, test_examples, label_list = load_CLINC150_with_specific_domain_sequence(args.DomainName, args.kshot, augment=False)
	num_labels = len(label_list)
	print('num_labels:', num_labels, 'test size:', len(test_examples))

	model = RobertaForSequenceClassification(num_labels)
	tokenizer = RobertaTokenizer.from_pretrained(pretrain_model_dir, do_lower_case=args.do_lower_case)
	# tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-large', do_lower_case=args.do_lower_case)

	# model.load_state_dict(torch.load('model_original_83/f1_0.7854086361654428.pt'))

	model.to(device)

	param_optimizer = list(model.named_parameters())
	no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
	optimizer_grouped_parameters = [
		{'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
		{'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
	]

	max_test_acc = 0.0

	if args.event:
		if args.english:
			print("Loading English Event Model.")
			# model.load_state_dict(torch.load('model/event/f1_0.7909405520391443.pt'))
			model.load_state_dict(torch.load('/shared/xdyu/event_coref/EventCoref/NLI/model_better/1e-06/f1_0.5072590402798975.pt'))
		else:
			print("Loading Spanish Event Model.")
			model.load_state_dict(torch.load('model/event_xlmr/f1_0.593213087813911.pt'))
	else:
		if args.english:
			model.load_state_dict(torch.load('model/entity/f1_.886795317197986.pt'))
		else:
			model.load_state_dict(torch.load('model/entity_xlmr/f1_0.4332229784922077.pt'))

	model.to(device)
	model.eval()

	test_features = convert_examples_to_features(
		test_examples, label_list, args.max_seq_length, tokenizer, output_mode,
		cls_token_at_end=False,  # bool(args.model_type in ['xlnet']),            # xlnet has a cls token at the end
		cls_token=tokenizer.cls_token,
		cls_token_segment_id=0,  # 2 if args.model_type in ['xlnet'] else 0,
		sep_token=tokenizer.sep_token,
		sep_token_extra=True,
		# bool(args.model_type in ['roberta']),           # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
		pad_on_left=False,  # bool(args.model_type in ['xlnet']),                 # pad on the left for xlnet
		pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
		pad_token_segment_id=0)  # 4 if args.model_type in ['xlnet'] else 0,)

	eval_all_pair_ids = [f.pair_id for f in test_features]

	eval_data, eval_sampler, test_dataloader = feature2vector(test_features, args.eval_batch_size)

	logger.info("***** Running test *****")
	logger.info("  Num examples = %d", len(test_features))

	eval_loss = 0
	nb_eval_steps = 0
	preds = []
	gold_label_ids = []
	# print('Evaluating...')
	for input_ids, input_mask, segment_ids, span_a_mask, span_b_mask, label_ids in test_dataloader:

		input_ids = input_ids.to(device)
		input_mask = input_mask.to(device)
		segment_ids = segment_ids.to(device)
		span_a_mask = span_a_mask.to(device)
		span_b_mask = span_b_mask.to(device)

		label_ids = label_ids.to(device)
		gold_label_ids += list(label_ids.detach().cpu().numpy())

		with torch.no_grad():
			logits = model(input_ids, input_mask, span_a_mask, span_b_mask)
		if len(preds) == 0:
			preds.append(logits.detach().cpu().numpy())
		else:
			preds[0] = np.append(preds[0], logits.detach().cpu().numpy(), axis=0)

	preds = preds[0]

	pred_probs = softmax(preds, axis=1)
	score_for_print = list(pred_probs[:, 0])
	assert len(eval_all_pair_ids) == len(score_for_print)
	pred_label_ids = list(np.argmax(pred_probs, axis=1))

	gold_label_ids = gold_label_ids
	assert len(pred_label_ids) == len(gold_label_ids)
	hit_co = 0
	for k in range(len(pred_label_ids)):
		if pred_label_ids[k] == gold_label_ids[k]:
			hit_co += 1
	test_acc = hit_co / len(gold_label_ids)

	overlap = 0
	for k in range(len(pred_label_ids)):
		if pred_label_ids[k] == gold_label_ids[k] and gold_label_ids[k] == 1:
			overlap += 1
	recall = overlap / (1e-6 + sum(gold_label_ids))
	precision = overlap / (1e-6 + sum(pred_label_ids))
	f1 = 2 * recall * precision / (1e-6 + recall + precision)

	# this is test
	if f1 > max_test_acc:
		max_test_acc = f1

	'''write new scores to test file'''
	writescore = codecs.open('test_scores.txt', 'w', 'utf-8')
	for id, score in enumerate(score_for_print):
		pair_idd = eval_all_pair_ids[id].split('&&')
		writescore.write(pair_idd[0] + '\t' + pair_idd[1] + '\t' + str(score) + '\n')
	print('test score written over')
	writescore.close()
	final_test_performance = f1
	print('\ntest:', [test_acc, f1], ' max_test_f1:', max_test_acc, '\n')


if __name__ == "__main__":
	main()

'''
CUDA_VISIBLE_DEVICES=6 python -u train_event_coref.py --task_name rte --do_train --do_lower_case --num_train_epochs 10 --data_dir '' --output_dir '' --train_batch_size 32 --eval_batch_size 80 --learning_rate 1e-6 --max_seq_length 128 --seed 42 --kshot 3 --beta_sampling_times 1
'''