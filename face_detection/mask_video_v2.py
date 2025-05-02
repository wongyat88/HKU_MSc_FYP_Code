import argparse
import json
import os
import cv2
import pandas as pd
from tqdm import tqdm


def mask_and_produce(frames_dir, cluster_mapping, frame_face_info_json, output_dir):
    """
    For each key in cluster_mapping (e.g. {'0': ['0'], '1': ['1']}),
    produce a video where only the faces whose 'prediction' is in
    cluster_mapping[key] are visible; everything else is black.
    """
    # load face detection JSON
    with open(frame_face_info_json) as f:
        frame_face_info = json.load(f)
    # index by frame_number for quick lookup
    frame_info_map = {
        entry["frame_number"]: entry["faces"] for entry in frame_face_info
    }

    # ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # prepare a VideoWriter for each speaker key
    writers = {}
    sample_frame = cv2.imread(os.path.join(frames_dir, os.listdir(frames_dir)[0]))
    h, w = sample_frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = 30  # adjust if needed

    for key in cluster_mapping:
        out_path = os.path.join(output_dir, f"masked_vid_{key}.mp4")
        writers[key] = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    # process each frame in order
    frame_files = sorted(os.listdir(frames_dir))
    for frame_file in tqdm(frame_files, desc="Masking frames"):
        frame_number = os.path.splitext(frame_file)[0].split("_")[-1].lstrip("0") or "0"
        orig = cv2.imread(os.path.join(frames_dir, frame_file))
        faces = frame_info_map.get(frame_number, [])

        for key, preds_to_keep in cluster_mapping.items():
            # start from a black image
            masked = orig.copy()
            masked[:] = 0

            # paste in each target face region
            for face in faces:
                if face["prediction"] in preds_to_keep:
                    x1 = int(face["coordinates"]["x1"])
                    y1 = int(face["coordinates"]["y1"])
                    x2 = int(face["coordinates"]["x2"])
                    y2 = int(face["coordinates"]["y2"])
                    # copy face patch from orig
                    masked[y1:y2, x1:x2] = orig[y1:y2, x1:x2]

            # # * Add 10% padding to each side of the face bounding box (Total 20% padding)
            # # get frame height/width
            # h_frame, w_frame = orig.shape[:2]

            # for face in faces:
            #     if face["prediction"] in preds_to_keep:
            #         # original coords
            #         x1 = int(face["coordinates"]["x1"])
            #         y1 = int(face["coordinates"]["y1"])
            #         x2 = int(face["coordinates"]["x2"])
            #         y2 = int(face["coordinates"]["y2"])

            #         # compute 10% of width/height to pad each side
            #         dw = int((x2 - x1) * 0.1)
            #         dh = int((y2 - y1) * 0.1)

            #         # expanded coords, clamped to image bounds
            #         x1e = max(0, x1 - dw)
            #         y1e = max(0, y1 - dh)
            #         x2e = min(w_frame, x2 + dw)
            #         y2e = min(h_frame, y2 + dh)

            #         # copy that slightly larger patch
            #         masked[y1e:y2e, x1e:x2e] = orig[y1e:y2e, x1e:x2e]

            writers[key].write(masked)

    # release all writers
    for w in writers.values():
        w.release()


def main():
    # Testing
    # frames_dir = r"F:\School\FYP2\backend_frontend_ui\face_detection\temp"
    frame_face_info_json = r"F:\School\FYP2\backend_frontend_ui\backend\public\phase5\detection_output.json"
    output_dir = r"F:\School\FYP2\backend_frontend_ui\backend\public\phase5"

    # map each speaker‐ID to the list of prediction labels you want to keep
    # here: speaker “0” → only pred “0”; speaker “1” → only pred “1”
    cluster_mapping = {
        "0": ["0"],
        "1": ["1"],
    }

    parser = argparse.ArgumentParser()
    # parser.add_argument("--frames_dir", type=str, default=frames_dir, help="Directory containing frames")
    parser.add_argument(
        "--frame_face_info_json",
        type=str,
        default=frame_face_info_json,
        help="Path to face detection JSON",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=output_dir,
        help="Output directory for masked videos",
    )
    parser.add_argument(
        "--cluster_mapping",
        type=json.loads,
        default=json.dumps(cluster_mapping),
        help="JSON string for cluster mapping",
    )

    args = parser.parse_args()

    print(args.cluster_mapping)
    print(type(args.cluster_mapping))

    # Get this file location
    current_dir = os.path.dirname(os.path.abspath(__file__))
    frames_dir = os.path.join(current_dir, "temp")  # Hard coded
    frame_face_info_json = args.frame_face_info_json
    output_dir = args.output_dir
    cluster_mapping = args.cluster_mapping

    mask_and_produce(frames_dir, cluster_mapping, frame_face_info_json, output_dir)

    print("Masking completed. Videos saved to:", output_dir)


if __name__ == "__main__":
    main()
