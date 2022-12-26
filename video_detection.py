import torch
import cv2
import numpy as np

from yolov7 import create_yolov7_model
from yolov7.trainer import filter_eval_predictions

import mediapipe as mp
from resnet import ResidualBlock, Resnet

# to get proper performance without dll duplicates define new environment
# import albumentations as A


COLORS = np.random.uniform(0, 255, size=(80, 3))
cropping = True

yolo_model = create_yolov7_model(architecture="yolov7", num_classes=1, pretrained=False)
checkpoint = torch.load("best_model.pt", map_location="cpu")
yolo_model.load_state_dict(checkpoint["model_state_dict"])
yolo_model.eval()

mpHands = mp.solutions.hands
hands = mpHands.Hands()

checkpoint = torch.load("model.pth", map_location="cuda")
model = checkpoint["model"]
model.load_state_dict(checkpoint["state_dict"])

int_to_labels = {0: 'palm', 1: 'l', 2: 'fist', 3: 'moved', 4: 'thumb', 5: 'index', 6: 'ok', 7: 'c', 8: 'down'}
# transforms = A.Compose([A.LongestMaxSize(max_size=640, p=1.0), A.PadIfNeeded(640, 640, border_mode=0,
#                                                                             value=(114, 114, 114),)])

def get_boxes(frame):
    frame = cv2.cvtColor(np.asarray(frame), cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (640, 640))  # does not keep aspect ratio: better resize by longest side and pad shortest
    # frame = transforms(image=frame)["image"]
    image_tensor = torch.FloatTensor(frame / 255.).permute(2, 0, 1).unsqueeze(0).to("cpu")

    with torch.no_grad():
        out = yolo_model(image_tensor)
    output = yolo_model.postprocess(out, conf_thres=0.001, max_detections=10, multiple_labels_per_box=True)
    output = filter_eval_predictions(output, confidence_threshold=0.2, nms_threshold=0.65)
    output = output[0].cpu().detach().numpy()
    boxes = output[:, :4] / 640
    boxes = boxes.tolist()

    return boxes


def draw_boxes(boxes, image, text="no_text"):

    image = cv2.cvtColor(np.asarray(image), cv2.COLOR_BGR2RGB)
    for i, box in enumerate(boxes):
        color = COLORS[i]
        if len(box) > 0:
            xmin, ymin, xmax, ymax = box
            h, w, c = image.shape

            xmin, xmax = xmin * w, xmax * w
            ymin, ymax = ymin * h, ymax * h

            cv2.rectangle(
                image,
                (int(xmin), int(ymin)),
                (int(xmax), int(ymax)),
                color, 2
            )
            cv2.putText(image, text, (int(xmin), int(ymin-5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2,
                        lineType=cv2.LINE_AA)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def get_hand_boxes(image, results):
    h, w, c = image.shape

    hand_landmarks = results.multi_hand_landmarks
    if hand_landmarks:
        hand_boxes = []
        for handLMs in hand_landmarks:
            x_max = 0
            y_max = 0
            x_min = w
            y_min = h

            for lm in handLMs.landmark:

                x, y = int(lm.x * w), int(lm.y * h)
                if x > x_max:
                    x_max = x
                if x < x_min:
                    x_min = x
                if y > y_max:
                    y_max = y
                if y < y_min:
                    y_min = y

        hand_box = [x_min, y_min, x_max, y_max]
        hand_boxes.append(hand_box)
        return hand_boxes


def crop_hands(image, boxes, threshoded=False):
    if boxes is None:
        return None
    hands = []
    h, w, c = image.shape
    for box in boxes:
        x_min, y_min, x_max, y_max = box
        canvas = np.zeros((y_max - y_min, w, 3))

        hand_image = image[y_min:y_max, x_min:x_max, :]

        canvas[:, x_min:x_max, :] = hand_image
        canvas = canvas.astype(np.uint8)
        canvas = cv2.cvtColor(canvas, cv2.COLOR_RGB2GRAY)
        if threshoded:
            _, canvas = cv2.threshold(canvas, 100, 255, cv2.THRESH_BINARY)
        hands.append(canvas)
    return hands


def get_hand_label(model, image):
    if len(image.shape) == 2:
        frame = np.repeat(image[..., np.newaxis], repeats=3, axis=-1)
    else:
        frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    frame = cv2.resize(frame, (256, 96))
    frame = frame / 127.5 - 1
    frame = frame[:, :, 0][None]
    tensor = torch.FloatTensor(frame).unsqueeze(0).to("cuda")

    model.eval()

    with torch.no_grad():
        logits = model(tensor)

    return int_to_labels.get(torch.max(logits, 1)[1].item())


def get_landmarks(results):
    hand_landmarks = results.multi_hand_landmarks
    if not hand_landmarks:
        return None

    landmarks = []
    h, w, c = image.shape

    for handLMs in hand_landmarks:
        for lm in handLMs.landmark:
            landmarks.append((int(lm.x * w), int(lm.y * h)))

    return landmarks


def increase_brightness(img, value=50):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2RGB)
    return img


def get_convex_crop(image, landmarks, brighten=False, value=50):  # rgb image
    if brighten:
        image = increase_brightness(image, value=value)

    convexHull = cv2.convexHull(np.array(landmarks))

    stencil = np.zeros_like(image, dtype=np.uint8)
    cv2.fillPoly(stencil, [convexHull], [255, 255, 255])
    new_image = cv2.bitwise_and(image, stencil)

    return new_image


cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

if not cap.isOpened():
    print('Cannot connect camera')

while cap.isOpened():
    ret, frame = cap.read()

    if ret:

        boxes = get_boxes(frame)
        text = f"Show sign"

        if len(boxes) > 0:  # do not response on hand gesture in case of no face detected

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(image)
            # hand_boxes = get_hand_boxes(image, results)
            landmarks = get_landmarks(results)
            # if hand_boxes is not None:
            if landmarks is not None:
                if cropping:
                    # images = crop_hands(image, hand_boxes, False)
                    # text = get_hand_label(model, images[0])
                    new_image = get_convex_crop(image, landmarks, brighten=True)
                    text = get_hand_label(model, new_image)
                else:
                    text = get_hand_label(model, frame)

        frame = draw_boxes(boxes, frame, text)

        cv2.imshow('image', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    else:
        break

cap.release()
cv2.destroyAllWindows()
