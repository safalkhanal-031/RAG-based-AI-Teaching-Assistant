# Converts the videos to mp3

import os
import subprocess

files = os.listdir("RagSampleVideos")
for file in files:
    tutorial_number = file.split(" [")[0].split("#")[1]
    file_name = file.split(" ｜")[0]
    print(tutorial_number, file_name)
    subprocess.run(["ffmpeg", '-i', f"RagSampleVideos/{file}", f"Audios/{tutorial_number}_{file_name}.mp3"])