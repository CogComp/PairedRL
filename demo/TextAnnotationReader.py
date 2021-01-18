import json
import pickle

from classes import *

event_all = []
data = {}

def ReadData(ta):
	# infile = open(path, 'r')
	# ta = json.load(infile)

	# for doc_id in ta.keys():
	doc_dict = ta
	doc_id = 'input'
	doc = ECBDoc()
	doc.set_doc_info(str(doc_id), '')
	last_index = 0
	sen_token_idx = []
	for sen_idx, sen_end_idx in enumerate(doc_dict['sentences']['sentenceEndPositions']):
		doc.sentences[sen_idx] = Sentence('0', doc_id)
		sen_token_idx.append((last_index, sen_end_idx))
		last_index = sen_end_idx + 1

	token_sen_mapping = {}
	sen_token_mapping = {}

	string = ''
	text = doc_dict['text']
	for t_idx, token in enumerate(doc_dict['tokens']):
		for sen_id, sen_idx in enumerate(sen_token_idx):
			if sen_idx[0] <= t_idx <= sen_idx[1]:
				token_sen_mapping[t_idx] = sen_id
				sen_token_mapping[sen_id] = t_idx
				doc.sentences[sen_id].tokens.append(token)
				break
	print(sen_token_idx)
	print(token_sen_mapping)
	mentions_dict = {}
	for view in doc_dict['views']:
		if view['viewName'] == 'Event_extraction':
			mentions_dict = view['viewData'][0]['constituents']

	event_idx = 0
	for m_idx, mention in enumerate(mentions_dict):
		if 'properties' in mention and 'predicate' in mention['properties']:
			sen_id = mention['properties']['sentence_id']
			event = Event('0', doc_id, sen_id, mention['start'], mention['end']-1, mention['start'], mention['end']-1)
			event.mention_id = '_'.join([doc_id.replace('.xml', ''), str(sen_id), str(m_idx)])
			event.event_idx = event_idx
			event_all.append(event)
			doc.sentences[sen_id].events.append(event)
			event_idx += 1

	data[doc_id] = doc

	outfile = open('data/pairs.input', 'w')

	for (i, event_i) in enumerate(event_all):
		for (j, event_j) in enumerate(event_all):
			if i >= j:
				continue
			outstr = [event_i.mention_id, event_j.mention_id]
			# outstr = []
			sen_i = data[event_i.doc_id].sentences[event_i.sen_id]
			outstr.append(' '.join(sen_i.tokens))
			outstr.append(str(event_i.start_offset))
			outstr.append(str(event_i.end_offset))

			sen_j = data[event_j.doc_id].sentences[event_j.sen_id]
			outstr.append(' '.join(sen_j.tokens))
			outstr.append(str(event_j.start_offset))
			outstr.append(str(event_j.end_offset))

			if event_i.gold_cluster == event_j.gold_cluster:
				outstr.append('1')
			else:
				outstr.append('0')

			outstr.append('\n')
			outfile.write('\t'.join(outstr))

	outfile = open('data/pickle.test', 'wb')
	pickle.dump(data, outfile)


def main():
	data_path = 'data/sample.ta'
	ReadData(data_path)


if __name__ == "__main__":
	main()