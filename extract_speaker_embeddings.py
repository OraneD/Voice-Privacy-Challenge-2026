# extract_embeddings.py
import argparse, pickle, shutil, warnings
from pathlib import Path

import numpy as np
import torch
import torchaudio
from tqdm import tqdm

from utils import read_kaldi_format, scan_checkpoint
from evaluation.privacy.asv.speechbrain_vectors import SpeechBrainVectors
from evaluation.privacy.asv.utils import normalize_wave

p = argparse.ArgumentParser()
p.add_argument('--wav_scp', required=True)
p.add_argument('--model_dir', required=True)   # e.g. exp/asv_anon_B3
p.add_argument('--out_pkl', required=True)
p.add_argument('--wavlm', default='exp/asv_ssl/WavLM-Large.pt')
p.add_argument('--save_every', type=int, default=2000) # Wont save anything on VoxAnon (1000utt)
args = p.parse_args()

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# same resolution as run_evaluation.py
model_dir = Path(args.model_dir)
model_dir = scan_checkpoint(model_dir, 'CKPT+') or model_dir
model_dir = Path(model_dir)
print(f'Using model: {model_dir}')
assert (model_dir / 'embedding_model.ckpt').is_file(), f'no embedding_model.ckpt in {model_dir}'
if not (model_dir / 'WavLM-Large.pt').is_file():
    shutil.copy2(args.wavlm, model_dir)
    print(f'Copied WavLM-Large.pt to {model_dir}')

extractor = SpeechBrainVectors(vec_type='ecapa_ssl', model_path=model_dir, device=device)

wav_scp = read_kaldi_format(Path(args.wav_scp))

out_pkl = Path(args.out_pkl)
out_pkl.parent.mkdir(parents=True, exist_ok=True)
embs = {}
if out_pkl.exists():
    with open(out_pkl, 'rb') as f:
        embs = pickle.load(f)
    print(f'Resuming: {len(embs)} utterances already done')

def save():
    tmp = out_pkl.with_suffix('.tmp')
    with open(tmp, 'wb') as f:
        pickle.dump(embs, f)
    tmp.replace(out_pkl)

todo = [u for u in wav_scp if u not in embs]
for i, utt in enumerate(tqdm(todo), 1):
    wav_path = wav_scp[utt]
    if isinstance(wav_path, list):   # kaldi command format
        wav_path = wav_path[1]
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=UserWarning, module='torchaudio')
        signal, fs = torchaudio.load(wav_path)
    norm_wave = normalize_wave(signal, fs, device=device)
    with torch.no_grad():
        vec = extractor.extract_vector(audio=norm_wave, sr=fs, wav_path=wav_path)
    embs[utt] = vec.detach().cpu().squeeze().numpy().astype(np.float32)
    if i % args.save_every == 0:
        save()
save()
print(f'Done: {len(embs)} embeddings -> {out_pkl}')