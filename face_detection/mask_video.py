import json
import os
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm


def mask_and_produce(frames_dir, cluster_mapping, frame_face_info_json):
    # cluster_mapping: {"0": ["0", "1"], "1": ["2"]}
    frame_folder = frames_dir
    frame_list = os.listdir(frame_folder)
    num_frame = len(frame_list)

    with open(frame_face_info_json) as f:
        frame_face_info = json.load(f)
    frame_face_info_df = pd.DataFrame(data=frame_face_info)

    masked_frames = {}
    for key in cluster_mapping.keys():
        masked_frames[key] = []

    # Mask
    for i in tqdm(range(1, num_frame + 1)):
        frame_orig = cv2.imread(f"{frame_folder}/frame_{str(i).zfill(4)}.jpg")
        faces_info = list(frame_face_info_df[frame_face_info_df['frame_number'] == str(i)]['faces'])[0]
        for key in cluster_mapping.keys():
            frame_key_specific = frame_orig.copy()
            for face in faces_info:
                if face['prediction'] not in cluster_mapping[key]:
                    x1 = int(face['coordinates']['x1'])
                    x2 = int(face['coordinates']['x2'])
                    y1 = int(face['coordinates']['y1'])
                    y2 = int(face['coordinates']['y2'])
                    frame_key_specific[y1:y2,x1:x2] = 0
            masked_frames[key].append(frame_key_specific)

    for key in masked_frames.keys():
        img_list = masked_frames[key]
        fc = cv2.VideoWriter_fourcc(*'MP4V')
        video = cv2.VideoWriter(f"F:/School/FYP/face_detection/masked_vid_{key}.mp4", fc, 30, (1920, 1080))
        for image in img_list:
            video.write(image)


def main(frames_dir, cluster_mapping, frame_face_info_json):
    mask_and_produce(frames_dir, cluster_mapping, frame_face_info_json)

if __name__ == '__main__':
    frames_dir = "F:\\School\\FYP\\face_detection\\temp"
    cluster_mapping = {"0": ["0"], "1": ["1"]}
    frame_face_info_json = "F:\\School\\FYP\\face_detection\\output.json"
    main(frames_dir, cluster_mapping, frame_face_info_json)
