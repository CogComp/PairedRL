
class ECBDoc:

	def __init__(self):
		self.tokens = []
		self.sentences = {}
		self.topic = -1
		self.doc_id = 0
		self.doc_name = ''
		self.mentions = {}
		self.mention2token = {}
		self.mentions_type = {}
		self.token2mention = {}
		self.token2sentence = {}
		self.coref_mention = {}
		self.mention2cluster_id = {}
		self.cluster_head = []
		self.cluster_type = {}
		self.cluster_instance_id = {}
		self.clusters = {}

	def set_doc_info(self, doc_id, name):
		self.doc_id = doc_id
		self.doc_name = name


class Sentence:

	def __init__(self, topic, doc_id):
		self.topic = topic
		self.doc_id = doc_id
		self.tokens = []
		self.events = []
		self.entities = []


class Event:

	def __init__(self, topic, doc_id, sen_id, start_offset, end_offset, token_start_idx, token_end_idx):

		self.topic = topic
		self.doc_id = doc_id
		self.sen_id = sen_id
		self.start_offset = start_offset
		self.end_offset = end_offset
		self.token_start_idx = token_start_idx
		self.token_end_idx = token_end_idx

		self.arg0 = (-1,-1)
		self.arg1 = (-1,-1)
		self.time = (-1,-1)
		self.loc = (-1,-1)

		self.gold_cluster = '_'.join(['Singleton', doc_id, str(token_start_idx), str(token_end_idx)]) # Will be overwritten if belongs to a cluster
		self.cd_coref_chain = -1

		self.mention_id = '_'.join(
			[self.doc_id.replace('.xml', ''), str(sen_id), str(self.start_offset),str(self.end_offset)])


class Entity:

	def __init__(self, topic, doc_id, sen_id, start_offset, end_offset, token_start_idx, token_end_idx, type):
		self.topic = topic
		self.doc_id = doc_id
		self.sen_id = sen_id
		self.start_offset = start_offset
		self.end_offset = end_offset
		self.token_start_idx = token_start_idx
		self.token_end_idx = token_end_idx
		self.type = type
		self.gold_cluster = '_'.join(['Singleton', doc_id, str(token_start_idx), str(token_end_idx)]) # Will be overwritten if belongs to a cluster
		self.cd_coref_chain = -1

		self.mention_id = '_'.join(
			[self.doc_id.replace('.xml', ''), str(sen_id), str(self.start_offset), str(self.end_offset)])


class ACEDoc:

	def __init__(self, doc_name):
		self.sentences = {}
		self.doc_name = doc_name

	def set_doc_info(self, doc_id, name):
		self.doc_id = doc_id
		self.doc_name = name


class Event_ACE:

	def __init__(self, doc_id, sen, start_offset, end_offset, event_id):

		self.doc_id = doc_id
		self.sen = sen
		self.start_offset = start_offset
		self.end_offset = end_offset

		self.arguments = []
		self.arg0 = (-1,-1)
		self.arg1 = (-1,-1)
		self.time = (-1,-1)
		self.loc = (-1,-1)

		self.mention_id = event_id
		self.cd_coref_chain = '-'


class Entity_ACE:

	def __init__(self, doc_id, sen, start_offset, end_offset, event_id, type):

		self.doc_id = doc_id
		self.sen = sen
		self.start_offset = start_offset
		self.end_offset = end_offset

		self.type = type

		self.gold_cluster = event_id.rsplit('-', 1)[0]
		self.mention_id = event_id

		self.cd_coref_chain = -1