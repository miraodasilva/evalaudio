import os
import sys
sys.path.append("./other_scripts/WER/LRW/lipreading")
from pypesq import pesq
from pystoi.stoi import stoi
import librosa 
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import torch
import string
import python_speech_features 
import csv
#local
from WER.grid_16k import StandardCNN as WerGrid16k
from WER.LRW.lipreading.model import Audio_Model as WerLRW
from WER.LRW.utils import read_txt_lines
from WER.LRW.config import default_model_options as WerLRWOptions
from decode import CtcDecoder
import measures


def get_file_name(path):
	return os.path.splitext(path.split("/")[-1])[0]

#Parses .csv file containing annotations
def parse_annotations_csv(file, excluded_annotations=["sil", "sp"]):
    words = []
    with open(file, 'r') as csvfile:
        annotation_reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in annotation_reader:
            if row[2] not in excluded_annotations:
                words.append(row[2])

    return words

#Calculate PESQ metric
def calculate_pesq(real,fake,sr):
	#PESQ only work with sampling rates 8000 or 16000, so we must resample
	try:
		if sr != 8000:
			real = librosa.resample(real,sr,8000)
			fake = librosa.resample(fake,sr,8000)
		return pesq(real,fake,8000)
	except:
		return 0.0

#Calculate STOI metric
def calculate_stoi(real,fake,sr):
	#Convert to scipy format - int16 - since pystoi works with this format
	real = np.round(real * (2**16)).astype(np.int16)
	fake = np.round(fake * (2**16)).astype(np.int16)
	return stoi(real,fake,sr)

def calculate_mcd(real,fake,sr=16000):
	real_mfcc = python_speech_features.base.mfcc(real,samplerate=sr)
	fake_mfcc = python_speech_features.base.mfcc(fake,samplerate=sr)
	#K = 10 / np.log(10) * np.sqrt(2)
	K = 1
	return K * np.mean(np.sqrt(np.sum((real_mfcc - fake_mfcc) ** 2, axis=1)))

class WERModel(nn.Module):
	def __init__(self,dataset):
		super(WERModel,self).__init__()
		self.dataset = dataset
		if "grid" in dataset:
			alphabet = ['_'] + list(string.ascii_uppercase) + [' ']
			self.ctc_decoder = CtcDecoder(alphabet)
			self.classifier = WerGrid16k()
			self.classifier.load_ckpt()
			self.classifier.eval()
		elif "lrw" in dataset:
			options = WerLRWOptions()
			self.classifier = WerLRW( hidden_dim=options["hidden-dim"],num_classes=options["num-classes"],relu_type = options["relu-type"],tcn_options = options["tcn_options"])
			checkpoint = torch.load('./WER/LRW/model_best.pth.tar')
			self.classifier.load_state_dict(checkpoint['state_dict'])
			self.classifier.eval()
			self.label_list = read_txt_lines("./WER/LRW/500WordsSortedList.txt")
		else:
			raise NotImplementedError("This dataset does not support WER currently.")
	def forward(self,audio_fake,audio_lengths,annotations=None,names=None):
		if "grid" in self.dataset:
			batch_size = audio_fake.size(0)
			audio_fake_lengths = [a.size(-1) for a in audio_fake]
			probs = self.classifier(audio_fake, audio_fake_lengths)
			probs = probs.contiguous().detach().cpu()
			prob_lengths = [l // 640 for l in audio_fake_lengths]
			predictions = self.ctc_decoder.decode_batch(probs.detach().cpu(),prob_lengths)
			wer = []
			for i in range(batch_size):
				wer += [measures.WER(" ".join(parse_annotations_csv(annotations[i])),predictions[i].lower())]
				#wer += [sum([np.fromstring(x,dtype=np.uint8).sum() for x in predictions[i]])]
			return wer
		elif "lrw" in self.dataset:
			batch_size = audio_fake.size(0)
			"""
			audios = []
			for audio in audio_fake:
				audio = self.preprocess(audio.squeeze().cpu().numpy())
				audio = audio[:audio.shape[0]//640*640]
				audios += [torch.from_numpy(audio).cuda().unsqueeze(0)]
			audios = torch.stack(audios)
			"""
			audio_fake = (audio_fake - torch.mean(audio_fake,dim=-1,keepdim=True))/torch.std(audio_fake,dim=-1,keepdim=True)
			audio_fake = audio_fake[:,:,:audio_fake.size(-1)//640*640]	
			logits = self.classifier(audio_fake, lengths=audio_lengths)
			_, preds = torch.max(F.softmax(logits, dim=1).data, dim=1)
			names = [get_file_name(name).split("_")[0] for name in names]
			labels = [self.label_list.index(name) for name in names]
			correct = sum([preds[i].item()==labels[i] for i in range (batch_size)])
			return [(batch_size - correct) / batch_size]
