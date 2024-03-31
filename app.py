import argparse
import torch
import torch.nn.functional as F
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import cv2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import json
import os

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

mtcnn = MTCNN(select_largest=False, post_process=False, device=DEVICE).to(DEVICE).eval()

model = InceptionResnetV1(
    pretrained="vggface2", classify=True, num_classes=1, device=DEVICE
)

checkpoint = torch.load("model.pth", map_location=torch.device("cpu"))
model.load_state_dict(checkpoint["model_state_dict"])
model.to(DEVICE)
model.eval()


def predict(input_image):
    """Predict the label of the input_image"""
    # input_image = Image.open(input_image_path).convert("RGB")
    if isinstance(input_image, str):
        # If the input is a string, open the image file
        input_image = Image.open(input_image)
    input_image = input_image.convert("RGB")
    face = mtcnn(input_image)
    if face is None:
        raise Exception("No face detected")
    face = face.unsqueeze(0)  # add the batch dimension
    face = F.interpolate(face, size=(256, 256), mode="bilinear", align_corners=False)

    # convert the face into a numpy array to be able to plot it
    prev_face = face.squeeze(0).permute(1, 2, 0).cpu().detach().int().numpy()
    prev_face = prev_face.astype("uint8")

    face = face.to(DEVICE)
    face = face.to(torch.float32)
    face = face / 255.0
    face_image_to_plot = face.squeeze(0).permute(1, 2, 0).cpu().detach().int().numpy()

    target_layers = [model.block8.branch1[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)
    targets = [ClassifierOutputTarget(0)]

    grayscale_cam = cam(input_tensor=face, targets=targets, eigen_smooth=True)
    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(face_image_to_plot, grayscale_cam, use_rgb=True)
    face_with_mask = cv2.addWeighted(prev_face, 1, visualization, 0.5, 0)

    with torch.no_grad():
        output = torch.sigmoid(model(face).squeeze(0))
        prediction = "real" if output.item() < 0.5 else "fake"

        real_prediction = 1 - output.item()
        fake_prediction = output.item()

        confidences = {"real": real_prediction, "fake": fake_prediction}
    return confidences, face_with_mask


def process_video(video_path):
    """Process a video frame by frame"""
    # Open the video file
    video = cv2.VideoCapture(video_path)

    # Get the frame rate of the video
    fps = video.get(cv2.CAP_PROP_FPS)

    # Initialize counters for the total predictions and the number of frames
    total_predictions = {"real": 0, "fake": 0}
    num_frames = 0
    frame_counter = 0

    while True:
        # Read the next frame
        ret, frame = video.read()

        # If the frame is empty, we have reached the end of the video
        if not ret:
            break

        # Process every nth frame, where n is the frame rate of the video
        if frame_counter % round(fps / 4) == 0:
            # Convert the frame to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Convert the frame to a PIL Image
            input_image = Image.fromarray(frame)

            # Process the frame
            predictions, face_with_mask = predict(input_image)

            # Add the predictions to the total
            total_predictions["real"] += predictions["real"]
            total_predictions["fake"] += predictions["fake"]

            # Increment the number of processed frames
            num_frames += 1

            # Save the face_with_mask image for this frame

        # Increment the frame counter
        frame_counter += 1
    cv2.imwrite("output/face_with_mask.jpg", face_with_mask)
    # Release the video file
    video.release()

    # Calculate the average predictions
    avg_predictions = {
        key: total / num_frames for key, total in total_predictions.items()
    }

    return avg_predictions


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process an image or video")
    parser.add_argument(
        "file_path", type=str, help="The path to the image or video file"
    )

    args = parser.parse_args()

    # Get the file extension
    _, ext = os.path.splitext(args.file_path)

    # Process the file based on its extension
    if ext.lower() in [".jpg", ".jpeg", ".png"]:
        # This is an image file
        predictions, face_with_mask = predict(args.file_path)

        # Save the face_with_mask image
        cv2.imwrite("output/face_with_mask.jpg", face_with_mask)
    else:
        # This is a video file
        predictions = process_video(args.file_path)

    print(json.dumps(predictions))
    # print(predictions)
# pip install -r requirements.txt
