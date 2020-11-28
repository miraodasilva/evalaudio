import argparse
import os
import torch
import torchaudio
import numpy as np
from tqdm import tqdm
#local
import eval_metrics

#Taken from torchaudio
class Scale(object):
    """Scale audio tensor from a 16-bit integer (represented as a FloatTensor)
    to a floating point number between -1.0 and 1.0.  Note the 16-bit number is
    called the "bit depth" or "precision", not to be confused with "bit rate".
    Args:
        factor (int): maximum value of input tensor. default: 16-bit depth
    """

    def __init__(self, factor=2**31):
        self.factor = factor

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor of audio of size (Samples x Channels)
        Returns:
            Tensor: Scaled by the scale factor. (default between -1.0 and 1.0)
        """
        if not tensor.dtype.is_floating_point:
            tensor = tensor.to(torch.float32)

        return tensor / self.factor

    def __repr__(self):
        return self.__class__.__name__ + '()'

#static variables
sr= 16000
annotation_folder = "./WER/grid_annotations"

print("Loading args & data & model...")
#load args
parser = argparse.ArgumentParser()
parser.add_argument("--real_folder","-rf",help="Path to real audio folder")
parser.add_argument("--fake_folder","-ff",help="Path to fake audio folder")
parser.add_argument("--dataset","-ds",help="Name of dataset, to pick WER model", choices=["grid","lrw"])
parser.add_argument("--resample_50khz",action='store_true',help="If true, resample fake audio from 50 to 16 khz")
parser.add_argument("--resample_8khz",action='store_true',help="If true, resample fake audio from 8 to 16 khz")

args = parser.parse_args()

#Setting up our device
device = torch.device("cpu")
if (torch.cuda.is_available()):
	device = torch.device("cuda:0")
else:
	raise Exception("Nao ha GPU")

#Setting up audio normalization
audio_transforms = Scale()

#validation
wer = []
pesq = []
stoi = []
mcd = []

wer_model = eval_metrics.WERModel(args.dataset)
wer_model.to(device)
wer_model.eval()

print("Evaluating... ")
mismatches = 0
for root, _, files in os.walk(args.fake_folder):
	for f in tqdm(files,desc=root):
		#only care about wav
		if not f.endswith(".wav"):
			continue
		#loading audio/annotations
		audio_fake_path = os.path.join(root,f)
		if audio_fake_path.endswith(".npz"):
			audio_fake = torch.from_numpy(np.load(audio_fake_path)["data"]).view(1,-1)
		else:
			audio_fake,_ = torchaudio.load(audio_fake_path, channels_first=False, normalization=None)
			audio_fake = audio_transforms(audio_fake).view(1,-1)

		if args.resample_50khz:
			resample = torchaudio.transforms.Resample(50000,16000)
			audio_fake = resample(audio_fake)
		elif args.resample_8khz:
			resample = torchaudio.transforms.Resample(8000,16000)
			audio_fake = resample(audio_fake)
		audio_real_path = audio_fake_path.replace(args.fake_folder,args.real_folder)
		if not os.path.exists(audio_real_path):
			print(audio_real_path)
			mismatches += 1
			print("mismatch!")
			continue 
		if audio_real_path.endswith(".npz"):
			audio_real = torch.from_numpy(np.load(audio_real_path)["data"]).view(1,-1)
		else:
			audio_real,_ = torchaudio.load(audio_real_path, channels_first=False, normalization=None)
			audio_real = audio_transforms(audio_real).view(1,-1)

		audio_fake = audio_fake.to(device)	
		audio_real = audio_real.to(device)	

		if audio_real.size(-1) != audio_fake.size(-1):
			diff = abs(audio_real.size(-1) - audio_fake.size(-1))
			#Tolerance of 1000 samples
			print("Cut!")
			assert diff < 10000
			min_len = min(audio_real.size(-1),audio_fake.size(-1))
			audio_real = audio_real[:,:min_len]
			audio_fake = audio_fake[:,:min_len]

		
		annotation = audio_fake_path.replace(args.fake_folder,annotation_folder).replace(".wav",".align") if args.dataset == "grid" else None

		#metrics
		wer += [wer_model(audio_fake.view(1,1,-1),[audio_fake.size(-1)],annotations=[annotation],names=[audio_fake_path])]
		audio_fake_np = audio_fake.detach().squeeze().cpu().numpy()
		audio_real_np = audio_real.detach().squeeze().cpu().numpy()
		pesq += [eval_metrics.calculate_pesq(audio_real_np,audio_fake_np,sr)]
		stoi += [eval_metrics.calculate_stoi(audio_real_np,audio_fake_np,sr)]
		mcd +=[eval_metrics.calculate_mcd(audio_real_np,audio_fake_np,sr)]

pesq = np.mean(np.array(pesq))
stoi = np.mean(np.array(stoi))
mcd = np.mean(np.array(mcd))
wer = np.mean(np.array(wer))

print("PESQ: %.4f, STOI: %.4f, MCD: %.4f, WER: %.4f"%(pesq,stoi,mcd,wer))
print("Found %s mismatches, ie, samples which were in the fake folder but not in the real folder."%(mismatches))



