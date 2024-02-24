# now that we have the samples for the MOS eval, we need to make sure that the reviewers don't know what model they originate from

import os
import librosa
import random
from scipy.io.wavfile import write

if __name__ == "__main__":
    # load all files
    path = "/itet-stor/elucas/net_scratch/generative_inversion/evaluation/human_eval/"
    files = os.listdir(path)
    files = [f for f in files if f.endswith(".wav")]
    clips = []
    for f in files:
        audio = librosa.load(path + f, sr=22050)
        clips.append((f, audio))
    # shuffle the clips
    random.shuffle(clips)
    # save the shuffled clips
    path = "/itet-stor/elucas/net_scratch/generative_inversion/evaluation/human_eval/anonymized/"
    for i, c in enumerate(clips):
        write(path + str(i) + ".wav", 22050, c[1][0])
        print("saved clip " + str(i))
    # save mapping
    with open(path + "mapping.txt", "w") as f:
        for i, c in enumerate(clips):
            f.write(str(i) + " " + c[0] + "\n")
