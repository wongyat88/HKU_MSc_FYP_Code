# Part 0 - Frame extraction
import shutil
import subprocess
import os
import random

# Part I - Face detection
# import os
# os.environ['HF_HOME'] = "L:/huggingface_cache"
# os.environ['TRANSFORMERS_CACHE'] = "L:/huggingface_cache/models"
# os.environ['HF_DATASETS_CACHE'] = "L:/huggingface_cache/datasets"
# from huggingface_hub import hf_hub_download
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from skimage.transform import resize
import argparse
import json

# Part II - Clustering & Prediction
# from deepface import DeepFace
from tqdm import tqdm

# Part III - Output
from collections import Counter


def extract_frames_from_video(video_path, frame_store_path, method="ffmpeg"):
    os.makedirs(frame_store_path, exist_ok=True)
    if method == "ffmpeg":  # ffmpeg (fast)
        frame_store_format = f"{frame_store_path}/frame_%04d.jpg"
        subprocess.run(["ffmpeg", "-i", video_path, "-q:v", "2", frame_store_format])
    else:  # cv2 (slow)
        video = cv2.VideoCapture(video_path)
        frame_count = 1
        interval = 1
        while True:
            ret, frame = video.read()
            if frame is None:
                break
            else:
                plt.imsave(
                    f"{frame_store_path}/frame_{str(frame_count).zfill(4)}.jpg", frame
                )
                video.set(cv2.CAP_PROP_POS_FRAMES, frame_count + interval)  # Stream
                frame_count += interval
    print(
        f"Frame extraction finished. Total frames: {len(os.listdir(frame_store_path))}"
    )


def detect(
    frame_skip,
    conf_threshold,
    crop_as_square,
    force_resize,
    resize_to,
    video_path,
    batch,
    frame_store_path,
    n_speakers,
):
    interval = frame_skip
    conf_threshold = conf_threshold
    crop_as_square = crop_as_square
    force_resize = force_resize
    resize_to = resize_to

    # trained_model_dir = hf_hub_download(repo_id="arnabdhar/YOLOv8-Face-Detection", filename="model.pt")
    trained_model_dir_v2 = (
        os.path.dirname(os.path.realpath(__file__)) + "/yolov11n-face.pt"
    )
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    fps = int(fps)
    # interval = int(fps)

    # output_dir = img_out_path
    # os.makedirs(output_dir, exist_ok=True)

    model = YOLO(trained_model_dir_v2)

    frame_count = 1
    faces_crop_list = []
    faces_frame_coord_pred = []
    continue_stream = True
    while continue_stream:
        start_frame = frame_count  # Record the frame number
        count = 0
        # Batch inference
        imgs_to_pred = []
        for _ in tqdm(range(0, batch)):
            try:
                frame = cv2.imread(
                    f"{frame_store_path}/frame_{str(frame_count).zfill(4)}.jpg"
                )
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # BGR --> RGB
                imgs_to_pred.append(frame)
                frame_count += interval
            except:
                continue_stream = False  # Break the streaming
                break  # Break the for loop

        # Verbose
        print(f"Processed second: {frame_count/fps}")

        if len(imgs_to_pred) > 0:
            # Basic information - Dimension
            y_length = imgs_to_pred[0].shape[0]
            x_length = imgs_to_pred[0].shape[1]

            # Prediction
            predictions = model(imgs_to_pred, verbose=False, max_det=n_speakers)
            for img_to_pred, prediction in zip(imgs_to_pred, predictions):
                predicted_b_boxes = prediction.boxes.xyxy.cpu().numpy().astype("int")
                predicted_b_boxes_confidence = list(prediction.boxes.conf.cpu().numpy())
                # print(predicted_b_boxes)
                num_b_boxes = predicted_b_boxes.shape[0]
                faces_for_frame = 0
                # For each bounding box
                for i in range(0, num_b_boxes):
                    # Bounding box = Face
                    if predicted_b_boxes_confidence[i] >= conf_threshold:
                        x1 = predicted_b_boxes[i, 0]
                        x2 = predicted_b_boxes[i, 2]
                        y1 = predicted_b_boxes[i, 1]
                        y2 = predicted_b_boxes[i, 3]
                        x_diff = abs(x1 - x2)
                        y_diff = abs(y1 - y2)
                        if crop_as_square:
                            # Crop as square with bigger bounding box
                            long_edge = max(x_diff, y_diff)
                            # Make y edge longer
                            if y_diff < long_edge:
                                add_to_each_side = int((long_edge - y_diff + 1) / 2)
                                distance_to_upper_bound = y_length - (
                                    y2 + add_to_each_side
                                )
                                distance_to_lower_bound = y1 - add_to_each_side
                                print(
                                    "distance to bound",
                                    distance_to_upper_bound,
                                    distance_to_lower_bound,
                                )
                                if (
                                    distance_to_upper_bound >= 0
                                    and distance_to_lower_bound >= 0
                                ):
                                    y1 = y1 - add_to_each_side
                                    y2 = y2 + add_to_each_side
                                elif (
                                    distance_to_upper_bound >= 0
                                    and distance_to_lower_bound < 0
                                ):
                                    if abs(distance_to_upper_bound) >= abs(
                                        distance_to_lower_bound
                                    ):
                                        y1 = 0
                                        y2 = (
                                            y2
                                            + add_to_each_side
                                            + abs(distance_to_lower_bound)
                                        )
                                    else:
                                        print("Bounding box is too big, skipping...")
                                        continue
                                elif (
                                    distance_to_upper_bound < 0
                                    and distance_to_lower_bound >= 0
                                ):
                                    if abs(distance_to_lower_bound) >= abs(
                                        distance_to_upper_bound
                                    ):
                                        y1 = (
                                            y1
                                            - add_to_each_side
                                            - abs(distance_to_upper_bound)
                                        )
                                        y2 = y_length
                                    else:
                                        print("Bounding box is too big, skipping...")
                                        continue
                                else:
                                    print("Bounding box is too big, skipping...")
                                    continue
                            # Make x edge longer
                            if x_diff < long_edge:
                                add_to_each_side = int((long_edge - x_diff + 1) / 2)
                                distance_to_upper_bound = x_length - (
                                    x2 + add_to_each_side
                                )
                                distance_to_lower_bound = x1 - add_to_each_side
                                if (
                                    distance_to_upper_bound >= 0
                                    and distance_to_lower_bound >= 0
                                ):
                                    x1 = x1 - add_to_each_side
                                    x2 = x2 + add_to_each_side
                                elif (
                                    distance_to_upper_bound >= 0
                                    and distance_to_lower_bound < 0
                                ):
                                    if abs(distance_to_upper_bound) >= abs(
                                        distance_to_lower_bound
                                    ):
                                        x1 = 0
                                        x2 = (
                                            x2
                                            + add_to_each_side
                                            + abs(distance_to_lower_bound)
                                        )
                                    else:
                                        print("Bounding box is too big, skipping...")
                                        continue
                                elif (
                                    distance_to_upper_bound < 0
                                    and distance_to_lower_bound >= 0
                                ):
                                    if abs(distance_to_lower_bound) >= abs(
                                        distance_to_upper_bound
                                    ):
                                        x1 = (
                                            x1
                                            - add_to_each_side
                                            - abs(distance_to_upper_bound)
                                        )
                                        x2 = x_length
                                    else:
                                        print("Bounding box is too big, skipping...")
                                        continue
                                else:
                                    print("Bounding box is too big, skipping...")
                                    continue

                        cropped_img = img_to_pred[y1:y2, x1:x2]

                        if force_resize:
                            new_size = (resize_to, resize_to, cropped_img.shape[2])
                            cropped_img = resize(
                                cropped_img, new_size, preserve_range=True
                            )
                            cropped_img = cropped_img.astype("uint8")
                        else:
                            pass

                        # Save as file
                        # output_dir = "L:/MSCCS Project/HKU_MSc_FYP/face_detection/1-detect_out"
                        # img_save_name = f"frame_{start_frame + count}_face_{i}.jpg"
                        # plt.imsave(f"{output_dir}/{img_save_name}", np.squeeze(cropped_img))

                        # Save as array
                        # cropped_img = np.reshape(cropped_img, (1, resize_to, resize_to, cropped_img.shape[-1]))
                        # if faces_crop_arr is None:
                        #     faces_crop_arr = cropped_img
                        # else:
                        #     faces_crop_arr = np.append(faces_crop_arr, cropped_img, axis=0)

                        # Save as list
                        cropped_img = np.squeeze(cropped_img)
                        # plt.imshow(cropped_img)
                        # plt.show()
                        faces_crop_list.append(cropped_img)

                        faces_frame_coord_pred.append(
                            [start_frame + count, [x1, y1, x2, y2]]
                        )
                        faces_for_frame += 1
                # print(f"Detected faces for frame {start_frame + count}: {faces_for_frame}")
                count += 1

    return faces_frame_coord_pred, faces_crop_list


def predict(faces_crop, n_speakers):
    def train_pca(embeddings):
        from sklearn.decomposition import PCA

        embeddings = embeddings.astype("float32")
        pca_model = PCA(n_components=2)
        pca_model.fit(embeddings)
        pca_components = pca_model.transform(embeddings)
        # plt.figure(dpi=1000)
        # plt.scatter(list(pca_components[:,0]), list(pca_components[:,1]), marker='x', s=0.5)
        # plt.show()
        return pca_components

    def train_dbscan(components):
        from sklearn.cluster import DBSCAN

        clustering = DBSCAN(eps=0.5, min_samples=30).fit(components)
        y_pred = clustering.labels_  # model.predict(components)
        # plt.figure(dpi=1000)
        # plt.scatter(list(components[:,0]), list(components[:,1]), marker='x', s=0.5, c=y_pred)
        # plt.show()
        return y_pred

    # facenet_pytorch
    from facenet_pytorch import InceptionResnetV1
    from torchvision.transforms import ToTensor

    resnet = InceptionResnetV1(pretrained="vggface2", device="cuda").eval()
    tf_img = lambda i: ToTensor()(i).unsqueeze(0)
    embed = lambda input: resnet(input)

    embeddings = None
    for i in tqdm(range(0, len(faces_crop))):
        # embedding_objs = DeepFace.represent(img_path = faces_crop[i], enforce_detection=False, model_name="Facenet")
        # embedding = embedding_objs[0]['embedding']
        t = tf_img(faces_crop[i]).to("cuda")
        embedding = embed(t).squeeze().cpu().tolist()
        if embeddings is None:
            embeddings = np.array([embedding])
        else:
            embeddings = np.append(embeddings, np.array([embedding]), axis=0)

    pca_components = train_pca(embeddings)
    prediction = train_dbscan(pca_components)

    return prediction

    # for i in range(0, len(embeddings_l)):
    #     km_pred = prediction[i]
    #     img = np.squeeze(faces_crop[i, :, :, :])
    #     plt.title(km_pred)
    #     plt.imshow(img)
    #     plt.show()


def output_as_json(faces_frame_coord_pred, prediction, json_output, n_speakers):
    # faces_frame_coord_pred: [frame_number, [x1, y1, x2, y2]]
    # prediction: [class]
    c = Counter(prediction).most_common(n_speakers)
    target_prediction = [i[0] for i in c]
    output = []
    frame_list = list(set([i[0] for i in faces_frame_coord_pred]))
    for frame_num in frame_list:
        frame_info = {"frame_number": str(frame_num), "faces": []}
        for idx in range(0, len(faces_frame_coord_pred)):
            if faces_frame_coord_pred[idx][0] == frame_num:
                # Only record the most commonly seen faces
                # if prediction[idx] in target_prediction:
                frame_info["faces"].append(
                    {
                        "prediction": str(prediction[idx]),
                        "coordinates": {
                            "x1": str(faces_frame_coord_pred[idx][1][0]),
                            "y1": str(faces_frame_coord_pred[idx][1][1]),
                            "x2": str(faces_frame_coord_pred[idx][1][2]),
                            "y2": str(faces_frame_coord_pred[idx][1][3]),
                        },
                    }
                )
        if len(frame_info["faces"]) > 0:
            output.append(frame_info)
    with open(json_output, "w") as f:
        json.dump(output, f, indent=4)
    print(f"JSON stored at {json_output} successfully")


def save_speaker_samples(prediction, faces_frame_coord_pred, faces_crop, output_dir):
    """
    Saves 2 random sample images for each predicted speaker.

    Args:
        prediction (list): List of predicted speaker IDs for each face.
        faces_frame_coord_pred (list): List containing [frame_number, [coords]] for each face.
        faces_crop (list): List of cropped face images (numpy arrays).
        output_dir (str): The directory where the 'images' folder will be created.
    """
    images_save_path = os.path.join(output_dir, "images")
    if os.path.exists(images_save_path):
        shutil.rmtree(images_save_path)  # Deletes the directory and all its contents
    os.makedirs(images_save_path, exist_ok=True)
    print(f"Saving speaker sample images to: {images_save_path}")

    unique_speakers = sorted(
        list(set(p for p in prediction if p != -1))
    )  # Exclude noise label -1 if present

    for speaker_id in unique_speakers:
        # Find all indices for the current speaker
        speaker_indices = [i for i, p in enumerate(prediction) if p == speaker_id]

        # Randomly select up to 2 indices
        num_samples = min(len(speaker_indices), 2)
        selected_indices = random.sample(speaker_indices, num_samples)

        # Save the selected images
        for i, idx in enumerate(selected_indices):
            frame_number = faces_frame_coord_pred[idx][0]
            face_image = faces_crop[idx]
            # Ensure the image is in BGR format for cv2.imwrite
            if face_image.shape[2] == 3:  # Check if it's a color image
                face_image_bgr = cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR)
            else:  # Grayscale or other format
                face_image_bgr = (
                    face_image  # Assume it can be saved directly or handle as needed
                )

            filename = f"{speaker_id}_{frame_number}.png"
            save_path = os.path.join(images_save_path, filename)
            try:
                cv2.imwrite(save_path, face_image_bgr)
                # print(f"Saved sample for speaker {speaker_id}: {filename}")
            except Exception as e:
                print(f"Error saving image {save_path}: {e}")
    print("Finished saving speaker samples.")


def main(
    frame_skip,
    conf_threshold,
    crop_as_square,
    force_resize,
    resize_to,
    video_path,
    batch,
    json_output,
    n_speakers,
    frame_store_path,
):
    # Part 0 - Frame extraction with ffmpeg
    extract_frames_from_video(video_path, frame_store_path)

    # Part I - Face detection
    faces_frame_coord_pred, faces_crop = detect(
        frame_skip,
        conf_threshold,
        crop_as_square,
        force_resize,
        resize_to,
        video_path,
        batch,
        frame_store_path,
        n_speakers,
    )

    # Part II - Clustering & Detection
    prediction = predict(faces_crop, n_speakers)

    # Part III - Output as JSON
    output_as_json(faces_frame_coord_pred, prediction, json_output, n_speakers)

    # Part IV - Save speaker samples
    output_dir = os.path.dirname(json_output)
    save_speaker_samples(
        prediction,
        faces_frame_coord_pred,
        faces_crop,
        # os.path.dirname(os.path.abspath(__file__)),
        output_dir,
    )

    # Clean up temporary frame folder
    if os.path.exists(frame_store_path):
        try:
            shutil.rmtree(frame_store_path)
            print(f"Cleaned up temporary frame folder: {frame_store_path}")
        except OSError as e:
            print(f"Error removing temporary folder {frame_store_path}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--frame_skip",
        type=int,
        default=1,
        help="Number of frame skipped between processing",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.7,
        help="Minimum confidence level to be considered as valid detection",
    )
    parser.add_argument(
        "--vid_input_path",
        type=str,
        help="Full path where frame to be predicted is stored",
    )
    parser.add_argument(
        "--json_output_path",
        type=str,
        help="Full path where the output json file is stored",
    )
    parser.add_argument(
        "--yolo_batch_size",
        type=int,
        help="Batch size used in face detection with YoloV8",
    )
    parser.add_argument("--n_speakers", type=int, help="Number of speakers")
    args = parser.parse_args()

    print("Frame skip", args.frame_skip)
    print("Confidence level", args.conf)
    print("YoloV8 detection batch size", args.yolo_batch_size)
    print("INPUT VIDEO", args.vid_input_path)
    print("OUTPUT JSON", args.json_output_path)
    print("Number of speakers", args.n_speakers)

    # Create a temp folder on this root folder path, if not exist, if exist, clear the folder
    script_dir = os.path.dirname(os.path.realpath(__file__))
    frame_store_path = os.path.join(script_dir, "temp")

    if os.path.exists(frame_store_path):
        shutil.rmtree(frame_store_path)
        print(f"Cleared existing temporary frame folder: {frame_store_path}")

    os.makedirs(frame_store_path, exist_ok=True)
    print(f"Created temporary frame folder: {frame_store_path}")

    frame_skip = args.frame_skip
    conf_threshold = args.conf
    yolo_batch_size = args.yolo_batch_size
    vid_input_path = args.vid_input_path
    json_output = args.json_output_path
    n_speakers = args.n_speakers

    # Fixed parameters
    crop_as_square = True
    force_resize = True
    resize_to = 128

    # ### Debug ###
    # frame_store_path = "L:/MSCCS Project/HKU_MSc_FYP/face_detection/frames"
    # frame_skip = 1
    # conf_threshold = 0.7
    # vid_input_path = "L:/MSCCS Project/HKU_MSc_FYP/Full interview： Donald Trump details his plans for Day 1 and beyond in the White House [b607aDHUu2I].mp4"
    # yolo_batch_size = 512
    # json_output = "L:/MSCCS Project/HKU_MSc_FYP/face_detection/faces_detection_out.json"
    # n_speakers = 2

    main(
        frame_skip,
        conf_threshold,
        crop_as_square,
        force_resize,
        resize_to,
        vid_input_path,
        yolo_batch_size,
        json_output,
        n_speakers,
        frame_store_path,
    )
