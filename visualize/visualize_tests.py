import numpy as np
import torch
import cv2
import sys
import oflibpytorch as of
from tqdm import tqdm
import os
import matplotlib.pyplot as plt


def extract_background_frames(video_path, folder, frequency):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    # Get frames per second (fps) of the video
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Frames per second: {fps}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Frame count: {frame_count}")

    extracted_frames = 0

    if not os.path.isdir(f"{folder}/background_frames"):
        os.mkdir(f"{folder}/background_frames")

    while True:
        # Read the next frame
        ret, frame = cap.read()

        if not ret:
            break

        # Extract the frame if it's at the right position
        if extracted_frames % int(fps / frequency) == 0:
            frame_filename = f"{folder}/background_frames/frame_{str(extracted_frames).zfill(4)}.png"
            cv2.imwrite(frame_filename, frame)
            # print(f"Saved frame {extracted_frames} as {frame_filename}")

        extracted_frames += 1

    cap.release()
    print("Finished extracting frames")


# Function to generate arrow plots
def generate_upsampled_arrow_plot(flow_vecs, gt, index, side, video_frame_list, folder):
    video_frame_path = os.path.join(f"{folder}/background_frames/", video_frame_list[index])
    video_frame = cv2.imread(video_frame_path)
    w = 128
    h = 128
    shape = video_frame.shape
    x = shape[1] / 2 - w / 2
    y = shape[0] / 2 - h / 2
    video_frame = video_frame[int(y):int(y + h), int(x):int(x + w)]
    upsampled_flow_vecs_x = cv2.resize(flow_vecs[0, :, :], (480, 480), interpolation=cv2.INTER_NEAREST)
    upsampled_flow_vecs_y = cv2.resize(flow_vecs[1, :, :], (480, 480), interpolation=cv2.INTER_NEAREST)
    upsampled_flow_vecs = np.stack((upsampled_flow_vecs_x, upsampled_flow_vecs_y), axis=0)
    upsampled_video_frame = cv2.resize(video_frame, (480, 480), interpolation=cv2.INTER_NEAREST)
    upsampled_flow_vecs_torch = torch.tensor(upsampled_flow_vecs)
    flow = of.Flow(upsampled_flow_vecs_torch)
    flow_img_0 = flow.visualise_arrows(20, return_tensor=False, img=upsampled_video_frame, colour=(0, 0, 250),
                                       thickness=2)
    upsampled_flow_vecs_x = cv2.resize(gt[0, :, :], (480, 480), interpolation=cv2.INTER_NEAREST)
    upsampled_flow_vecs_y = cv2.resize(gt[1, :, :], (480, 480), interpolation=cv2.INTER_NEAREST)
    upsampled_flow_vecs = np.stack((upsampled_flow_vecs_x, upsampled_flow_vecs_y), axis=0)
    upsampled_flow_vecs_torch = torch.tensor(upsampled_flow_vecs)
    flow = of.Flow(upsampled_flow_vecs_torch)
    flow_img = flow.visualise_arrows(20, return_tensor=False, img=flow_img_0, colour=(250, 250, 0), thickness=2)
    cv2.imwrite(f"{folder}/frames/frame_{str(index).zfill(4)}_{side}.png", np.squeeze(flow_img, axis=0))


def plot_landing_video_wo_reset(flight, type):
    background_video_path = f'/media/odvorak/Expansion/2Phase/ALEDv1.1extension/PALD/{flight}/{type}.mp4'
    flight = 'testD'
    folder = f'/media/odvorak/Expansion/2Phase/TDF_tuning/Results/{flight}/'
    extract_background_frames(background_video_path, folder, 10)

    video_frame_list = sorted(os.listdir(f"{folder}/background_frames"))

    root = "/media/odvorak/Expansion/2Phase/TDF_tuning/Results"

    prediction_path = os.path.join(root, flight, "prediction.npy")
    mask_path = os.path.join(root, flight, "mask.npy")
    gt_path = os.path.join(root, flight, "gt.npy")

    pred_array = np.load(prediction_path)
    mask_array = np.load(mask_path)
    gt_array = np.load(gt_path)

    N = pred_array.shape[0]
    if not os.path.isdir(f"{folder}/frames/"):
        os.mkdir(f"{folder}/frames/")
    if not os.path.isdir(f"{folder}/error_frames/"):
        os.mkdir(f"{folder}/error_frames/")

    error_list = []
    for j in tqdm(range(0, N - 2, 1)):
        flow_vecs = torch.tensor(pred_array[j, :, :, :])
        gt_vecs = torch.tensor(gt_array[j, :, :, :])
        error = torch.sqrt((flow_vecs[0, :, :] - gt_vecs[0, :, :]) ** 2 + (flow_vecs[1, :, :] - gt_vecs[1, :, :]) ** 2)
        mean = error.mean()
        error_list.append(mean)
    print(np.median(error_list))

    # Create a video from the frames
    frame_height, frame_width = (480, 480)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(f'{folder}/prediction.mp4', fourcc, 10, (frame_width, frame_height))

    for image in sorted(os.listdir(f"{folder}/frames/")):
        frame = cv2.imread(os.path.join(f"{folder}/frames/", image))
        out.write(frame)

    out.release()
    print("Video created successfully")


if __name__ == "__main__":
    list_flights = ["N9", "N10", "N11", "N12", "D9", "D10", "D11", "D12", "V5", "V6"]
    for flight in list_flights:
        plot_landing_video_wo_reset_multi(flight, "flight")