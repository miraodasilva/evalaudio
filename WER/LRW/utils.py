import os
import cv2
import torch
import librosa
import shutil
import logging
import numpy as np


# -- Media utils
def extract_opencv(filename, bgr=False):
    video = []
    cap = cv2.VideoCapture(filename)

    while(cap.isOpened()):
        ret, frame = cap.read() # BGR
        if ret:
            video.append(frame)
        else:
            break
    cap.release()
    video = np.array(video)
    return video if bgr else video[...,::-1]

# -- load audio
def load_audio(audio_filename, specified_sr=None, int_16=True):                  
    try:                                                                         
        if audio_filename.endswith('npy'):                                       
            audio = np.load(audio_filename)                                      
        elif audio_filename.endswith('npz'):                                     
            audio = np.load(audio_filename)['data']                              
        else:                                                                    
            audio, sr = librosa.load(audio_filename, sr=None)                    
            audio = librosa.resample(audio, sr, specified_sr) if sr != specified_sr else audio
    except IOError:                                                              
        sys.exit()                                                               
    if int_16 and audio.dtype==np.float32:                                       
        audio = ((audio - 1.) * (65535./2.) + 32767.).astype(np.int16)           
        audio = np.array(np.clip(np.round(audio), -2**15, 2**15-1), dtype=np.int16)
    if not int_16 and audio.dtype==np.int16:
        audio = ((audio - 32767.) * 2 / 65535. + 1).astype(np.float32)
    return audio

# -- torch utils
def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

# -- IO utils
def read_txt_lines(filepath):
    assert os.path.isfile( filepath ), "Error when trying to read txt file, path does not exist: {}".format(filepath)
    with open( filepath ) as myfile:
        content = myfile.read().splitlines()
    return content

# -- logging utils
def get_logger(log_path):
    logger = logging.getLogger("mylog")
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logger.addHandler(console)
    return logger
