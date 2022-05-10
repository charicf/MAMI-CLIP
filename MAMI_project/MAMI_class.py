import argparse
import numpy as np
import clip
import pdb

import os
import cv2
import gc
import numpy as np
import pandas as pd
import itertools
from tqdm.autonotebook import tqdm
import albumentations as A # For image augmentation to increase the manage images and create new training samples from the existing data.

import torch
from torch import nn
import torch.nn.functional as F
import timm #PyTorch Image Models (timm) is a collection of IMAGE models, layers, utilities, optimizers, schedulers, data-loaders / augmentations, and reference training / validation scripts that aim to pull together a wide variety of state of the art models with ability to reproduce ImageNet training results.
from transformers import DistilBertModel, DistilBertConfig, DistilBertTokenizer # Transformers provides thousands of pretrained models to perform tasks on TEXTS such as classification, information extraction, question answering, summarization, translation, text generation and more in over 100 languages.

import matplotlib
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import math

import re
#NLTK
import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
# This allows to create individual objects from a bog of words
#nltk.download('punkt')
from nltk.tokenize import word_tokenize
# Lemmatizer helps to reduce words to the base form
#nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
#nltk.download('words')
# Ngrams allows to group words in common pairs or trigrams..etc
from nltk import ngrams
# We can use counter to count the objects
from collections import Counter

import spacy
#python -m spacy download en_core_web_sm

from sklearn.metrics import classification_report

class CFG:


	debug = False
	#image_path = "C:/Moein/AI/Datasets/Flicker-8k/Images"
	#image_path = "/content/gdrive/MyDrive/Fourth semester/MS_project/MAMI/data/TRAINING/"
	home_path = os.path.dirname(os.path.realpath(__file__))
	image_path = f"{home_path}/TRAINING"
	#image_path = f"{home_path}/TEST"
	batch_size = 1
	num_workers = 0
	head_lr = 1e-3
	image_encoder_lr = 1e-4
	text_encoder_lr = 1e-5
	weight_decay = 1e-3
	patience = 1
	factor = 0.8
	epochs = 4
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	model_name = 'resnet50' #'vit_base_patch16_224'
	image_embedding = 2048
	text_encoder_model = "distilbert-base-uncased"
	text_embedding = 768
	text_tokenizer = "distilbert-base-uncased"
	max_length = 200
	labels_max_length = 50

	pretrained = True # for both image encoder and text encoder
	trainable = True # for both image encoder and text encoder
	temperature = 1.0

	# image size
	size = 224 # for resnet50, 'vit_large_patch16_224', 'vit_base_patch16_224'
	#size = 384 # for 'vit_large_patch16_384'

	# for projection head; used for both image and text encoders
	num_projection_layers = 1
	projection_dim = 256 
	dropout = 0.1

# Calculates the average Loss between all batches. At the end of loop it returns the obtained average Loss
class AvgMeter:
	def __init__(self, name="Metric"):
		self.name = name
		self.reset()

	def reset(self):
		self.avg, self.sum, self.count = [0] * 3

	def update(self, val, count=1):
		self.count += count # count brings the number of images in the batch
		self.sum += val * count # val is the mean loss from the CLIP model
		self.avg = self.sum / self.count

	def __repr__(self):
		text = f"{self.name}: {self.avg:.4f}"
		return text

class CLIPDataset(torch.utils.data.Dataset):
	def __init__(self, dataframe, tokenizer, transforms):
		"""
		image_filenames and cpations must have the same length; so, if there are
		multiple captions for each image, the image_filenames must have repetitive
		file names 
		"""

		self.image_filenames = dataframe['file_name']
		self.captions = list(dataframe['transcripts'])  
		self.labels = list(dataframe['misogynous'])# Comment for testing
		# encoded_captions returns a dictionary with keys input_ids and attention_mask
		self.encoded_captions = tokenizer(
			list(dataframe['transcripts']), padding=True, truncation=True, max_length=CFG.max_length
		) # Dic with two keys: 'input_ids' and 'attention_mask'. Each key has len(examples) arrays
		self.encoded_labels_text = tokenizer(list(dataframe['misogynous']), padding=True, truncation=True, max_length=CFG.labels_max_length) # Dic with two keys: 'input_ids' and 'attention_mask'. Each key has len(examples) arrays
		self.transforms = transforms

	def __getitem__(self, idx):
		item = {
			key: torch.tensor(values[idx])
			for key, values in self.encoded_captions.items()
		} # Item contains the values for the 'input_ids' and 'attention_mask' for the caption idx

		
		for key, values in self.encoded_labels_text.items():
			item[key+"_labels"] = torch.tensor(values[idx])

		image = cv2.imread(f"{CFG.image_path}/{self.image_filenames[idx]}")
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		image = self.transforms(image=image)['image']
		item['image'] = torch.tensor(image).permute(2, 0, 1).float()
		item['caption'] = self.captions[idx]
		item['label'] = self.labels[idx]

		return item


	def __len__(self):
		return len(self.captions)

class ImageEncoder(nn.Module):
	"""
	Encode images to a fixed size vector. In case of ResNet50 the vector size will be 2048
	"""

	def __init__(
		self, model_name=CFG.model_name, pretrained=CFG.pretrained, trainable=CFG.trainable
	):
		super().__init__()
		self.model = timm.create_model(
			model_name, pretrained, num_classes=0, global_pool="avg"
		)
		# Print size of embedding agter image encoder
		o = self.model(torch.randn(2, 3, CFG.size, CFG.size))
		print(f'Image embeddings shape: {o.shape}')
		for p in self.model.parameters():
			p.requires_grad = trainable

	def forward(self, x):
		return self.model(x)

class TextEncoder(nn.Module):

	#Output hidden representation for each token is a vector with size 768

	def __init__(self, model_name=CFG.text_encoder_model, pretrained=CFG.pretrained, trainable=CFG.trainable):
		super().__init__()
		if pretrained:
			self.model = DistilBertModel.from_pretrained(model_name)
		else:
			self.model = DistilBertModel(config=DistilBertConfig())
			
		for p in self.model.parameters():
			p.requires_grad = trainable

		# we are using the CLS token hidden representation as the sentence's embedding
		self.target_token_idx = 0

	def forward(self, input_ids, attention_mask):
		output = self.model(input_ids=input_ids, attention_mask=attention_mask)
		last_hidden_state = output.last_hidden_state
		return last_hidden_state[:, self.target_token_idx, :]

class ProjectionHead(nn.Module):
	def __init__(
		self,
		embedding_dim, # Size of input vector  images (2048) and text (768)
		projection_dim=CFG.projection_dim, # Size of output vector : 256
		dropout=CFG.dropout
	):
		super().__init__()
		self.projection = nn.Linear(embedding_dim, projection_dim)
		self.gelu = nn.GELU()
		self.fc = nn.Linear(projection_dim, projection_dim)
		self.dropout = nn.Dropout(dropout)
		self.layer_norm = nn.LayerNorm(projection_dim)
	
	def forward(self, x):
		projected = self.projection(x)
		x = self.gelu(projected)
		x = self.fc(x)
		x = self.dropout(x)
		x = x + projected
		x = self.layer_norm(x)
		return x

class CLIPModel(nn.Module):
	def __init__(
		self,
		temperature=CFG.temperature,
		image_embedding=CFG.image_embedding,
		text_embedding=CFG.text_embedding,
	):
		super().__init__()
		self.image_encoder = ImageEncoder()
		self.text_encoder = TextEncoder()
		self.image_projection = ProjectionHead(embedding_dim=image_embedding)
		self.text_projection = ProjectionHead(embedding_dim=text_embedding)
		self.img_text_projection1 = ProjectionHead(embedding_dim=image_embedding + text_embedding)
		self.img_text_projection = ProjectionHead(embedding_dim=2*CFG.projection_dim)
		self.temperature = temperature
		#self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07)
		self.logit_scale = nn.Parameter(torch.ones([]) * -0.2)

	def forward(self, batch):
		# Getting Image and Text Features
		image_features = self.image_encoder(batch["image"])
		text_features = self.text_encoder(
			input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
		)
		labels_features = self.text_encoder(input_ids=batch["input_ids_labels"], attention_mask=batch["attention_mask_labels"])

		img_text_features = torch.cat((image_features, text_features),1)
		
		'''
		# Getting Image and Text Embeddings (with same dimension)
		image_embeddings = self.image_projection(image_features)
		text_embeddings = self.text_projection(text_features)
		
		# Calculating the Loss
		#   Compute the similarity matrix
		#logits = (text_embeddings @ image_embeddings.T) / self.temperature # Logits contains the similarity matrix between the text and image embeddings. Its size is batch_size x batch_size
		logit_scale = self.logit_scale.exp()
		logits = logit_scale *  image_embeddings @ text_embeddings.t()
		#   Compute the target
		images_similarity = image_embeddings @ image_embeddings.T # Similarity between same images should output higher values in the diagonal. This means they are similar (both are same matrix)
		texts_similarity = text_embeddings @ text_embeddings.T # Similarity between same images should output higher values in the diagonal. This means they are similar (both are same matrix)
		targets = F.softmax(
			(images_similarity + texts_similarity) / 2 * self.temperature, dim=-1
		) # Target will contain the correct similarity between images and texts combined, wich after softmax should be a matrix with values close to one in its diagonal and 0 otw
		texts_loss = cross_entropy(logits, targets, reduction='none') #With the targets matrix, we will use simple cross entropy to calculate the actual loss.
		images_loss = cross_entropy(logits.T, targets.T, reduction='none')
		loss =  (images_loss + texts_loss) / 2.0 # shape: (batch_size)
		'''

		# Getting Image and Text Embeddings (with same dimension)
		image_embeddings = self.image_projection(image_features)
		img_txt_embeddings1 = self.img_text_projection1(img_text_features)
		texts_embeddings = self.text_projection(text_features)
		img_txt_embeddings = torch.cat((image_embeddings, texts_embeddings),1)
		img_txt_embeddings = self.img_text_projection(img_txt_embeddings)
		pdb.set_trace()
		texts_embeddings = self.text_projection(labels_features)

		# Calculating the Loss
		#   Compute the similarity matrix
		#logits = (text_embeddings @ image_embeddings.T) / self.temperature # Logits contains the similarity matrix between the text and image embeddings. Its size is batch_size x batch_size
		logit_scale = self.logit_scale.exp()
		logits = logit_scale *  img_txt_embeddings @ texts_embeddings.t() # Shape is batch_size by batch_size
		#   Compute the target
		img_text_similarity = img_txt_embeddings @ img_txt_embeddings.T # Similarity between same images should output higher values in the diagonal. This means they are similar (both are same matrix)
		texts_similarity = texts_embeddings @ texts_embeddings.T # Similarity between same images should output higher values in the diagonal. This means they are similar (both are same matrix)
		targets = F.softmax(
			(img_text_similarity + texts_similarity) / 2 * self.temperature, dim=-1
		) # Target will contain the correct similarity between images and texts combined, wich after softmax should be a matrix with values close to one in its diagonal and 0 otw
		texts_loss = cross_entropy(logits, targets, reduction='none') #With the targets matrix, we will use simple cross entropy to calculate the actual loss.
		image_text_loss = cross_entropy(logits.T, targets.T, reduction='none')
		loss =  (image_text_loss + texts_loss) / 2.0 # shape: (batch_size)
		#pdb.set_trace()

		return loss.mean()


def preprocess_regex(text):

	#text = str.lower(text)
	text = re.sub(r'[-a-zA-Z0-9@:%._\+~#=]{1,256}\.(?!(zip))([a-zA-Z()]{1,6})', '', text)# web pages except .zip
	text = re.sub(r'@[a-zA-Z0-9.-]+', '', text)# Matches social media users
	text = re.sub(r'\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,4}\b', '', text)# Matches emails
	text = re.sub(r'([0-1]?[0-9]|2[0-3]):[0-5][0-9](\s+\b(am|pm)\b)?\s+', '', text)# Find time HH:MM
	text = re.sub(r'(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s+([0-2][0-9]|3[0-1]),?\s+(\d{4})', '', text)

	# general
	text = re.sub(r"n\'t", " not", text)
	text = re.sub(r"\'re", " are", text)
	text = re.sub(r"\'s", " is", text)
	text = re.sub(r"\'d", " would", text)
	text = re.sub(r"\'ll", " will", text)
	text = re.sub(r"\'t", " not", text)
	text = re.sub(r"\'ve", " have", text)
	text = re.sub(r"\'m", " am", text)

	#text = re.sub('[^a-zA-Z.\d\s]', '', text)# Delete special characters
	text = re.sub('\.{2,}', '.', text) # Replace multiple dots by one dot
	text = re.sub('\s{2,}', ' ', text) # replace multiple spaces by one space
	text = re.sub('^\s+|\s+$', '', text) # Delete spaces at the beggining or end of text

	return text

nlp = spacy.load('en_core_web_sm')
words = set(nltk.corpus.words.words())
selected_words = {'fuckin', 'vs.', 'toes', 'fuck', 'rap', 'hoe', 'ear', 'females', 'twitter', 'fang', 'reddit', 'islam', 'tittys', 'fuck', 'photoshop', 'boobs+cleavage=', 'tattoos', 'jenner', 'wars', 'covid-19', 'coronavirus', 'democrats', 'india', 'mcdonalds', 'christmas', 'biden', 'schoolgirls', 'girlfriend', 'instagram', 'coworker', 'amazon', 'halloween', 'girls', 'bernie', 'milf', 'milfs', 'hinduism', 'jainism', 'buddhism', 'sikhism', 'christainnity', 'christian', 'jain', 'hindu', 'buddhist', 'sikh', 'muslim', 'scarlett', 'johansson', 'hormones', 'prostitutein', 'prostituã', 'prostituierte', 'prostituta', 'hillary', 'clinton', 'teresa', 'heinz', 'barbara', 'hellen', 'thomas', 'cindy', 'sheehan', 'monica', 'lewinsky', 'christiane', 'amanpour', 'michelle', 'obama', 'susan', 'estrich', 'pelosi', 'rosie', 'wouldonnell', 'barbara', 'streisand', 'madeleine', 'albright', 'janeane', 'garofalo', 'schoolgirls', 'cougar'}

def pre_process_POS(text, nlp, words, namelist, profanity_words, spell):

	#Creating doc object
	doc = nlp(text)
	#Extracting POS
	lemma_tokens = []
	#discarded = []
	for token in doc:
		if token.pos_ != 'X' and token.pos_ != 'PROPN':
			if token.lemma_ == '-PRON-':
				lemma_tokens.append(token.text)
			else:
				lemma_tokens.append(token.lemma_)
		elif (token.text in words) or (token.text in namelist) or (len(spell.unknown([token.text]))==0) or (token.text in profanity_words):
			if token.lemma_ == '-PRON-':
				lemma_tokens.append(token.text)
			else:
				lemma_tokens.append(token.lemma_)

		#print(token.text)
	#else:
		# discarded.append([token.text, token.pos_])
	#print('lemma: ', lemma_tokens)
	#print(discarded)

	# Take out the stopwords
	text_tokens =[t for t in lemma_tokens if t not in stopwords.words('english')]
	#print('stopwords: ',text_tokens)

	# Extract only the alphabetic characters (deletes numbers too)
	non_text_tokens = [t for t in text_tokens if t.isalpha() == False]
	#print(non_text_tokens)
	text_tokens = [t for t in text_tokens if t.isalpha()]

	return ' '.join(text_tokens) # Returns a tokens as string

def process_data(dir_path, clean_data_file=1):
	if clean_data_file==1:
		df_clean = pd.read_csv(os.path.join(home_dir, 'cleaned_texts.csv'), header='infer')
	else:
		# Read data from original csv file and lower case the texts
		data_info = pd.read_csv(os.path.join(home_dir, 'training.csv'), sep='\t', header='infer')
		#data_info = pd.read_csv(os.path.join(home_dir, 'Test.csv'), sep='\t', header='infer')
		data_info = data_info.rename({'Text Transcription': 'transcripts'}, axis=1)
		data_info['transcripts'] = data_info['transcripts'].str.lower()

		#Clean the texts bby applying the preprocess_regex. Save the dataframe into 'cleaned_texts.csv'
		df_clean = data_info.copy()
		df_clean['transcripts'] = df_clean.apply(lambda row : preprocess_regex(row[6]), axis = 1) #Apply preprocess_regex to row 6 ('transcripts')
		df_clean.to_csv('cleaned_texts.csv', index=False)
		#%cp -av cleaned_texts.csv $dir_path # Save data with cleaned texts from regex

	##### Apply pre_process_POS to pre-process the texts (applying POS tagging). Save the new dataframe as 'processed_texts.csv' #####

	# Construct a list of classified names, using the names corpus.
	namelist = ([name.lower() for name in names.words('male.txt')] + [name.lower() for name in names.words('female.txt')])
	#Swear words from google_profanity_words.txt
	profanity_words = {'chink', 'b00bs', 'donkeyribber', 'kunilingus', 'mutherfucker', 'fudge packer', 'semen', 'motherfuckings', 'jiz', 'pussies', 'fux', 'cipa', 'boner', 'jerk-off', 'jackoff', 'nobjokey', 'beastiality', 'poop', '5hit', 'shited', 'fukkin', 'fistfuck', 'lusting', 's hit', 'jap', 'cumming', 'fingerfucking', 'pissing', 'gangbang', 'boooobs', 'whore', 'fukwhit', 'kum', 'masterb8', 'dick', 'muff', 'fagging', 'nigger', 'v1gra', 'twunter', 'pecker', 'l3i+ch', 'fuckings', 'duche', 'breasts', 'pimpis', 'kums', 'cuntlick', 'tittie5', 'fingerfucked', 'rimjaw', 'fuckwit', 'tittywank', 'fucked', 'beastial', 'ma5terbate', 'dogging', 'l3itch', 'mothafuck', 'feck', 'm0f0', 'cok', 'w00se', 'm0fo', 'shittings', 'horniest', 'phuked', 'penis', 'boobs', 'masterbation', 'cunnilingus', 'a_s_s', 'assfukka', 'cockmuncher', 'orgasm', 'muthafuckker', 'cuntlicking', 'knobead', 'mothafuckaz', 'pissin', 'clit', 'shemale', 'nazi', 'damn', 'goddamn', 'cunilingus', 'b!tch', 'mutha', 'fuks', 'ejaculation', 'bastard', 'gaylord', 'phuking', 'xxx', 'fags', 'faggot', 'nob', 'faggitt', 'god-dam', 'buttplug', 'mothafucka', 'muther', 'p0rn', 'cocksukka', 'cl1t', 'shitey', 'tw4t', 'ejaculate', 'teets', 'tittyfuck', 'f u c k e r', 'pussy', 'dirsa', 'ejakulate', 'shitted', 'willies', 'fellate', 'cums', 'fuckwhit', 'ma5terb8', 'mof0', 'bloody', 'asswhole', 't1tt1e5', 'masochist', 'motherfuckin', 'hardcoresex', 'heshe', 'mothafucks', 'asses', 'goddamned', 'phuks', 'homo', 'horny', 'balls', 'hotsex', 'kumming', 'mothafuckin', 'bi+ch', 'ass-fucker', 'fooker', 'numbnuts', 'cyberfuckers', 'fuckme', 'fannyflaps', 'coon', 'jizz', 'booooooobs', 'pissoff', 'wang', 'prick', 'fingerfuckers', 'f u c k', 'bitcher', 'smegma', 'testical', 'skank', 'twat', 'nobhead', 'fistfuckers', 'boiolas', 'pusse', 'fellatio', 'cumshot', 'fcuk', 'anal', 'motherfuckers', 'knobhead', 'porn', 'tits', 'motherfucks', 'knobjokey', 'kock', 'ejaculating', 'kummer', 'anus', 'labia', 'fudgepacker', 'flange', 'hore', 'assfucker', 'kondum', 'clitoris', 'fistfuckings', 'titwank', 'nigga', 'fuk', 'fuckers', 'mothafuckas', 'tittiefucker', 'pisses', 'shitting', 'fistfucking', 'n1gger', 'motherfuckka', 'God', 'ass', 'cokmuncher', 'cunillingus', 'shag', 'a55', 'bitchin', 'schlong', 'wanky', 'cockhead', 'dildos', 'felching', 'f4nny', 'biatch', 'sluts', 'cum', 'mo-fo', 'orgasms', 'cyberfucking', 'fuck', 'testicle', 'vagina', 'cyalis', 'kawk', 'bitches', 'scrotum', 'fukwit', 'cunts', 'pisser', 'willy', 'bugger', 'v14gra', 'shitings', 'fucking', 'booooobs', 'nigg4h', 'titfuck', 'rectum', 'ar5e', 'fux0r', 'masturbate', 'fucks', '5h1t', 'masterbat3', 'pawn', 'c0cksucker', 'viagra', 'gangbanged', 'ejaculated', 'motherfucker', 'muthafecker', 'hoer', 'fcuker', 'motherfuck', 'clits', 'snatch', 'motherfucking', 'fucka', 'mofo', 'penisfucker', 'turd', 'smut', 'cockmunch', 'b1tch', 'pornos', 'shagging', 'fingerfucks', 'god-damned', 'mothafucking', 'shitfull', 'dickhead', 'pissed', 'pussys', 'shi+', 'phonesex', 'niggaz', 'cocksuka', 'blow job', 'piss', 'arse', 'scrote', 'mother fucker', 'fanyy', 'fatass', 'niggers', 'porno', 'm45terbate', 'shit', 'nutsack', 'knobend', 'dlck', 'shitfuck', 'cock-sucker', 'knobjocky', 'shithead', 'mothafucked', 'cocksucking', 'fcuking', 'mothafucker', 'faggs', 'screwing', 'fuker', 'cockface', 'bum', 'master-bate', 'masterbations', 'blowjobs', 'ballbag', 'tosser', 'cyberfuck', 'cyberfucked', 'vulva', 'nobjocky', 'scroat', 'bestiality', 'ejaculatings', 'phukked', 'cocks', 'fingerfucker', 'hoar', 'fistfucker', 'gaysex', 'd1ck', 'cnut', 'pron', 'shite', 'titties', 'xrated', 'cuntlicker', 'butt', 'orgasims', 'fecker', 'b17ch', 'cyberfucker', 'niggas', 'cocksucks', 'fag', 'phukking', 'pube', 'pussi', 'fuckingshitmotherfucker', 'sh!+', 'rimming', 'shiting', 'jizm', 'coksucka', 'sex', 'motherfucked', 's_h_i_t', 'goatse', 'sh!t', 's.o.b.', 'c0ck', 'butthole', 't1tties', 'lust', 'fanny', 'fucker', 'pigfucker', 'dink', 'twunt', 'mothafuckings', 'shits', 'slut', 'dyke', 'spac', 'fook', 'cock', 'bunny fucker', 'sh1t', 'dildo', 'fukker', 'jack-off', '4r5e', 'dinks', 'wanker', 'son-of-a-bitch', 'pissers', 'fistfucked', 'bellend', 'booobs', 'fuckin', 'wank', 'asshole', 'buttmunch', 'cyberfuc', 'teez', 'bollock', 'n1gga', 'dog-fucker', 'nob jokey', 'cocksucked', 'bitching', 'nigg3r', 'hell', 'sadist', 'shitdick', 'shaggin', 'masterbat*', 'bitch', 'doggin', 'fannyfucker', 'boob', 'masterbate', 'shagger', 'knobed', 'shitters', 'crap', 'fagots', 'blowjob', 'arrse', 'spunk', 'lmfao', 'pissflaps', 'niggah', 'whoar', 'f_u_c_k', 'mothafuckers', 'buceta', 'phuck', 'fuckheads', 'pornography', 'fistfucks', 'carpet muncher', 'bestial', 'shitty', 'kondums', 'jism', 'shitter', 'titt', 'cawk', 'phuq', 'tit', 'bitchers', 'ballsack', 'phuk', 'cox', 'twathead', 'twatty', 'assholes', 'doosh', 'cocksucker', 'bollok', 'fingerfuck', 'knob', 'cocksuck', 'orgasim', 'hoare', 'fagot', 'ejaculates', 'cummer', 'gangbangs', 'pricks', 'fuckhead', 'cunt', 'retard'}
	nlp = spacy.load('en_core_web_sm')
	english_words = set(nltk.corpus.words.words())
	spell = SpellChecker() # It does not discard Twitter, Verizon, iphone...

	df_clean['transcripts'] = df_clean.apply(lambda row : pre_process_POS(row[6], nlp, english_words, namelist, profanity_words, spell), axis = 1)
	df_clean.transcripts = df_clean.transcripts.fillna('') # replace na's with empty strings
	print(len(df_clean[df_clean['transcripts'] == '']), 'rows have empty srtings')

	df_clean.to_csv('processed_texts.csv', index=False, na_rep='')
	#%cp -av processed_texts.csv $dir_path
	#df_processed = pd.read_csv('/content/gdrive/MyDrive/Fourth semester/MS_project/MAMI/data/processed_texts.csv', header='infer', keep_default_na=False)

	return df_clean

def split_sets(data, dir_path, train_perc=0.9, vali_perc=0.2):

	#msk = np.random.rand(len(no_miso)) < 0.8
	#train = no_miso[msk]
	#test = no_miso[~msk]

	# Vali_perc != 0 means that there is a validation set
	if vali_perc!=0:
		# Split data into train (0.9) and test(0.1)
		train_val = data.groupby('misogynous').sample(frac=train_perc)
		test = data.loc[data.index.difference(train_val.index)]
		# Split train into training (0.8) and validating(0.2)
		train = train_val.groupby('misogynous').sample(frac=1-vali_perc)
		validation = train_val.loc[train_val.index.difference(train.index)]

		# Save set: Validation
		validation.to_csv('validation.csv', index=False)
		#%cp -av validation.csv $dir_path

	else:
		train = data.groupby('misogynous').sample(frac=train_perc)
		test = data.loc[data.index.difference(train.index)]

	# Save sets: Training, validating, and testing
	train.to_csv('train.csv', index=False)
	#%cp -av train.csv $dir_path

	test.to_csv('test.csv', index=False)
	#%cp -av test.csv $dir_path

def make_small_train_set(df, dir_path, num_elems_class):
	# Get num_elems_class rows with label misogynous and num_elems_class rows with label non misogynous
	data = pd.concat([df.loc[df['misogynous'] == 0][:num_elems_class], df.loc[df['misogynous'] == 1][:num_elems_class]], ignore_index=True)
	train_val = data.groupby('misogynous').sample(frac=.9)
	test_small = data.loc[data.index.difference(train_val.index)]

	train_small = train_val.groupby('misogynous').sample(frac=.8)
	validation_small = train_val.loc[train_val.index.difference(train_small.index)]

	train_small.to_csv('train_small.csv', index=False)
	#%cp -av train_small.csv $dir_path

	test_small.to_csv('test_small.csv', index=False)
	#%cp -av test_small.csv $dir_path

	validation_small.to_csv('validation_small.csv', index=False)
	#%cp -av validation_small.csv $dir_path

def append_label_as_text(dataframe):

	dataframe.loc[dataframe['misogynous'] == 1, 'transcripts'] = dataframe.loc[dataframe['misogynous'] == 1].apply(lambda row : row[6]+' [SEP] a misogynist meme' , axis = 1)
	dataframe.loc[dataframe['misogynous'] == 0, 'transcripts'] = dataframe.loc[dataframe['misogynous'] == 0].apply(lambda row : row[6]+' [SEP] a meme' , axis = 1)

	dataframe["misogynous"] = dataframe["misogynous"].replace({0: 'a meme', 1:'a misogynist meme'})

	return dataframe

def get_transforms(mode="train"):
	if mode == "train":
		return A.Compose(
			[
				A.Resize(CFG.size, CFG.size, always_apply=True),
				A.Normalize(max_pixel_value=255.0, always_apply=True),
			]
		)
	else:
		return A.Compose(
			[
				A.Resize(CFG.size, CFG.size, always_apply=True),
				A.Normalize(max_pixel_value=255.0, always_apply=True),
			]
		)

def get_lr(optimizer):
	for param_group in optimizer.param_groups:
		return param_group["lr"]

def cross_entropy(preds, targets, reduction='none'):
	log_softmax = nn.LogSoftmax(dim=-1)
	loss = (-targets * log_softmax(preds)).sum(1)
	if reduction == "none":
		return loss
	elif reduction == "mean":
		return loss.mean()

def build_loaders(dataframe, tokenizer, mode):
	transforms = get_transforms(mode=mode)
	dataset = CLIPDataset(
		dataframe,
		tokenizer=tokenizer,
		transforms=transforms,
	)
	dataloader = torch.utils.data.DataLoader(
		dataset,
		batch_size=CFG.batch_size,
		num_workers=CFG.num_workers,
		shuffle=True if mode == "train" else False,
	)
	return dataloader

def train_epoch(model, train_loader, optimizer, lr_scheduler, step):
	loss_meter = AvgMeter()
	tqdm_object = tqdm(train_loader, total=len(train_loader))
	for batch in tqdm_object: # Each batch is size 4xnum_examples_in_batch. It has four keys: ['input_ids', 'attention_mask', 'images', 'caption'] and each key has num_examples_in_batch arrays
		batch = {k: v.to(CFG.device) for k, v in batch.items() if k != "caption" and k != "label"} # The batch that is sent to the model contains the image embedding and the transcriptions embeddings ('input_ids', 'attention_mask'). The raw text ('caption') is not sent to the CLIP model
		loss = model(batch)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		if step == "batch":
			lr_scheduler.step()

		count = batch["image"].size(0)
		loss_meter.update(loss.item(), count)

		tqdm_object.set_postfix(train_loss=loss_meter.avg, lr=get_lr(optimizer))
	return loss_meter

def valid_epoch(model, valid_loader):
	loss_meter = AvgMeter()

	tqdm_object = tqdm(valid_loader, total=len(valid_loader))
	for batch in tqdm_object:
		batch = {k: v.to(CFG.device) for k, v in batch.items() if k != "caption"}
		loss = model(batch)

		count = batch["image"].size(0)
		loss_meter.update(loss.item(), count)

		tqdm_object.set_postfix(valid_loss=loss_meter.avg)
	return loss_meter

def train_CLIP(train, validation):
	tokenizer = DistilBertTokenizer.from_pretrained(CFG.text_tokenizer)
	train_loader = build_loaders(train, tokenizer, mode="train")
	valid_loader = build_loaders(validation, tokenizer, mode="valid")


	model = CLIPModel().to(CFG.device)
	params = [
		{"params": model.image_encoder.parameters(), "lr": CFG.image_encoder_lr},
		{"params": model.text_encoder.parameters(), "lr": CFG.text_encoder_lr},
		{"params": itertools.chain(
			model.image_projection.parameters(), model.text_projection.parameters()
		), "lr": CFG.head_lr, "weight_decay": CFG.weight_decay}
	]
	optimizer = torch.optim.AdamW(params, weight_decay=0.)
	lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
		optimizer, mode="min", patience=CFG.patience, factor=CFG.factor
	)
	step = "epoch"

	best_loss = float('inf')
	for epoch in range(CFG.epochs):
		print(f"Epoch: {epoch + 1}")
		model.train()
		train_loss = train_epoch(model, train_loader, optimizer, lr_scheduler, step)
		model.eval()

		torch.save(model.state_dict(), f"{CFG.home_path}/best.pt")
		
		with torch.no_grad():
			valid_loss = valid_epoch(model, valid_loader)
		
		if valid_loss.avg < best_loss:
			best_loss = valid_loss.avg
			torch.save(model.state_dict(), "best.pt")
			print("Saved Best Model!")
		
		lr_scheduler.step(valid_loss.avg)

def get_image_embeddings(test, model_path):
	tokenizer = DistilBertTokenizer.from_pretrained(CFG.text_tokenizer)
	test_loader = build_loaders(test, tokenizer, mode="valid")

	model = CLIPModel().to(CFG.device)
	model.load_state_dict(torch.load(model_path, map_location=CFG.device))
	model.eval()
	
	test_image_embeddings = []
	with torch.no_grad():
		for batch in tqdm(test_loader):
			image_features = model.image_encoder(batch["image"].to(CFG.device))
			image_embeddings = model.image_projection(image_features)
			test_image_embeddings.append(image_embeddings)
	return model, torch.cat(test_image_embeddings)

def find_matches(model, image_embeddings, query, image_filenames, n=9):
	tokenizer = DistilBertTokenizer.from_pretrained(CFG.text_tokenizer)
	#pdb.set_trace()
	#qu = [tokenizer(c, padding=True) for c in query]
	encoded_query = tokenizer(query, padding=True) # Padding true to pad to the longest sequence in the query so both tensors are same size

	batch = {
			key: torch.tensor(values).to(CFG.device)
			for key, values in encoded_query.items()
		}
	with torch.no_grad():
		text_features = model.text_encoder(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
		text_embeddings = model.text_projection(text_features)
	#pdb.set_trace()
	image_embeddings_n = F.normalize(image_embeddings, p=2, dim=-1)
	text_embeddings_n = F.normalize(text_embeddings, p=2, dim=-1)
	dot_similarity = text_embeddings_n @ image_embeddings_n.T # Find similarity between the query text embedding and the images in the batch. Returns  vector of size 1 x #images in batch 

	
	# cosine similarity as logits
	logit_scale = (torch.ones([]) * np.log(1 / 0.05)).exp()
	logits = logit_scale * text_embeddings_n @ image_embeddings_n.T

	probs = logits.T.softmax(dim=-1).cpu().numpy()

	#pdb.set_trace()
	
	'''
	values, indices = torch.topk(dot_similarity.squeeze(0), n * 1)
	matches = [image_filenames[idx] for idx in indices]
	
	_, axes = plt.subplots(int(math.sqrt(n)), int(math.sqrt(n)), figsize=(10, 10))
	for match, ax in zip(matches, axes.flatten()):
		image = cv2.imread(f"{CFG.image_path}/{match}") 
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		ax.imshow(image)
		ax.axis("off")
	
	plt.show()
	'''
	return probs

def main():

	# You can add any args you want here
	parser = argparse.ArgumentParser(description='Hyperparams')
	parser.add_argument('-p',  nargs='?', type=str, default='dataset', help='Path to dataset')
	parser.add_argument('-mode',  nargs='?', type=int, default=0, help='0: to test, 1: to train')
	parser.add_argument('-cleaned',  nargs='?', type=int, default=1, help='0: otw, 1: if cleaned_texts.csv exists')
	parser.add_argument('-processed',  nargs='?', type=int, default=1, help='0: otw, 1: if processed_texts.csv exists')
	parser.add_argument('-split', nargs='?', type=int, default=0, help='1: Perform the split of dataset, otw: Do not perform split')
	parser.add_argument('-stat', nargs='?', type=int, default=0, help='1: Show statistics, otw: Do not show stats')
	parser.add_argument('-train',  nargs='?', type=str, default="train.csv", help='name of csv file')
	parser.add_argument('-test',  nargs='?', type=str, default="test.csv", help='name of csv file')
	parser.add_argument('-valid',  nargs='?', type=str, default="validation.csv", help='name of csv file')
	parser.add_argument('-si',  nargs='?', type=int, default=8000, help='Index to split training from testing')
	parser.add_argument('-seed', type=int, default=1, help='random seed (default: 1)')

	args = parser.parse_args()
	files = ["train.csv", "test.csv"]

	mean = torch.Tensor([0.4977])
	std = torch.Tensor([0.2136])

	home_dir = os.path.dirname(os.path.realpath(__file__))
	folder_name = args.p
	folder_path = os.path.join(home_dir, folder_name)

	train_csv_path = os.path.join(home_dir, args.train)
	valid_csv_path = os.path.join(home_dir, args.valid)
	test_csv_path = os.path.join(home_dir, args.test)
	train_model = args.mode
	clean_data_file = args.cleaned # 1 if the file with cleaned data already exists (after regex)
	processed_data = args.processed # 1 if the file with pre-processed data already exists (after POS)
	split_set = args.split # 1 to split data into training, validation and testing sets
	show_statistics = args.stat # 1 to show statistics, including plots

	swearWords= set()
	with open(os.path.join(home_dir, "google_profanity_words.txt")) as f:
		for line in f:
			swearWords.add(line.rstrip())
	#print(swearWords)

	#train_df, valid_df = make_train_valid_dfs()

	#clean_data_file = 1 # 1 if the file with cleaned data already exists (after regex)
	#processed_data = 1 # 1 if the file with pre-processed data already exists (after POS)
	#split_set = 1 # 1 to split data into training, validation and testing sets
	#show_statistics = 1 # 1 to show statistics, including plots

	#----------- Pre-process the data -----------
	if processed_data == 1:
		# If csv file processed_data exits, load the data
		df_processed = pd.read_csv(os.path.join(home_dir, 'processed_texts.csv'), header='infer', keep_default_na=False)
	else:
		# pre-process the transcripts and return the processed dataframe
		df_processed = process_data(home_dir, clean_data_file)
	
	#--------- Show statistics --------------------
	if show_statistics == 1:
		show_corpus_statistics(df_processed)

	# -------- split the data into three sets ------
	if split_set == 1:
		# If data has not been split, split it into training, validating, testing
		#split_sets(df_processed, dir_path)
		make_small_train_set(df_processed, dir_path, 500)
	
	# ---------- Load the data -----------------
	train = pd.read_csv(os.path.join(home_dir, 'train_small.csv'), header='infer', keep_default_na=False)
	append_label_as_text(train)
	validation = pd.read_csv(os.path.join(home_dir, 'validation_small.csv'), header='infer', keep_default_na=False)
	append_label_as_text(validation)

	#----------- Train model -----------------
	if train_model == 1:
		print('Training the model...')
		train_CLIP(train, validation)
		
	test = pd.read_csv(os.path.join(home_dir,'test_small.csv'), header='infer', keep_default_na=False)
	#append_label_as_text(test)

	model, image_embeddings = get_image_embeddings(test, "best.pt")

	probs = []
	y_pred = []
	#pdb.set_trace()
	#test['transcripts1'] = test['transcripts']
	#test['transcripts'] = test['transcripts'].apply(lambda x: x+' [SEP] a meme')
	#test['transcripts1'] = test['transcripts1'].apply(lambda x: x+' [SEP] a misogynist meme')
	#query = test[['transcripts', 'transcripts1']].values.tolist()
	#query = np.array(query)
	#pdb.set_trace()
	for index, row in test.iterrows():
		query = [row['transcripts']+" [SEP] a meme", row['transcripts']+" [SEP] a misogynist meme"]
		prob = find_matches(model, image_embeddings[index].unsqueeze(0), query, image_filenames=test['file_name'].values, n=9)
		#prob = find_matches(model, image_embeddings[index].unsqueeze(0), query=query[index], image_filenames=test['file_name'].values, n=9)
		#prob = find_matches(model, image_embeddings, query=query, image_filenames=test['file_name'].values, n=9)
		probs.append(prob)
		y_pred.append(np.argmax(prob)) # The largest prob between both labels is taken as the prediction

	print('y_pred: ', y_pred)

	y_true = test['misogynous'].values.tolist()
	target_names = ['not misogynous', 'misogynous']
	print(classification_report(y_true, y_pred, target_names=target_names), '\n')


if __name__ == '__main__':
	main()
