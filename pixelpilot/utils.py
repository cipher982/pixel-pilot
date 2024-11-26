import base64
import io
import os
import time
from functools import wraps
from typing import Dict
from typing import List
from typing import Tuple

import cv2
import easyocr
import numpy as np
import supervision as sv
import torch
from matplotlib import pyplot as plt
from PIL import Image
from torchvision.ops import box_convert
from torchvision.transforms import ToPILImage

from pixelpilot.logger import setup_logger

os.environ["TOKENIZERS_PARALLELISM"] = "false"

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


logger = setup_logger(__name__)

reader = easyocr.Reader(["en"])


def log_runtime(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        duration = time.perf_counter() - start
        logger.info(f"{func.__name__} took {duration:.2f} seconds")
        return result

    return wrapper


def get_yolo_model(model_path):
    from ultralytics import YOLO

    # Load the model.
    model = YOLO(model_path)
    return model


@log_runtime
@torch.inference_mode()
def get_parsed_content_icon(filtered_boxes, ocr_bbox, image_source, caption_model_processor, prompt=None):
    to_pil = ToPILImage()
    if ocr_bbox:
        non_ocr_boxes = filtered_boxes[len(ocr_bbox) :]
    else:
        non_ocr_boxes = filtered_boxes
    croped_pil_image = []
    for i, coord in enumerate(non_ocr_boxes):
        xmin, xmax = int(coord[0] * image_source.shape[1]), int(coord[2] * image_source.shape[1])
        ymin, ymax = int(coord[1] * image_source.shape[0]), int(coord[3] * image_source.shape[0])
        cropped_image = image_source[ymin:ymax, xmin:xmax, :]
        croped_pil_image.append(to_pil(cropped_image))

    model, processor = caption_model_processor["model"], caption_model_processor["processor"]
    if not prompt:
        if "florence" in model.config.name_or_path:
            prompt = "<CAPTION>"
        else:
            prompt = "The image shows"

    batch_size = 100
    generated_texts = []
    device = model.device

    batch_count = len(croped_pil_image) // batch_size
    logger.info(f"Parsing {len(croped_pil_image)} sized image in {batch_count} batches...")
    for i in range(0, len(croped_pil_image), batch_size):
        batch = croped_pil_image[i : i + batch_size]

        # Use original working code for now
        inputs = processor(images=batch, text=[prompt] * len(batch), return_tensors="pt").to(device=device)

        if "florence" in model.config.name_or_path:
            generated_ids = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                num_beams=3,
                do_sample=False,
            )
        else:
            generated_ids = model.generate(
                **inputs,
                max_length=100,
                num_beams=5,
                no_repeat_ngram_size=2,
                early_stopping=True,
                num_return_sequences=1,
            )  # temperature=0.01, do_sample=True,
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
        generated_text = [gen.strip() for gen in generated_text]
        generated_texts.extend(generated_text)

    logger.info(f"DEBUG: parsed content: {generated_texts}")

    return generated_texts


@log_runtime
def remove_overlap(boxes, iou_threshold, ocr_bbox=None):
    assert ocr_bbox is None or isinstance(ocr_bbox, List)

    def box_area(box):
        return (box[2] - box[0]) * (box[3] - box[1])

    def intersection_area(box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        return max(0, x2 - x1) * max(0, y2 - y1)

    def IoU(box1, box2):
        intersection = intersection_area(box1, box2)
        union = box_area(box1) + box_area(box2) - intersection + 1e-6
        if box_area(box1) > 0 and box_area(box2) > 0:
            ratio1 = intersection / box_area(box1)
            ratio2 = intersection / box_area(box2)
        else:
            ratio1, ratio2 = 0, 0
        return max(intersection / union, ratio1, ratio2)

    boxes = boxes.tolist()
    filtered_boxes = []
    if ocr_bbox:
        filtered_boxes.extend(ocr_bbox)
    # print('ocr_bbox!!!', ocr_bbox)
    for i, box1 in enumerate(boxes):
        # if not any(IoU(box1, box2) > iou_threshold and box_area(box1) > box_area(box2)
        # for j, box2 in enumerate(boxes) if i != j):
        is_valid_box = True
        for j, box2 in enumerate(boxes):
            if i != j and IoU(box1, box2) > iou_threshold and box_area(box1) > box_area(box2):
                is_valid_box = False
                break
        if is_valid_box:
            # add the following 2 lines to include ocr bbox
            if ocr_bbox:
                if not any(IoU(box1, box3) > iou_threshold for k, box3 in enumerate(ocr_bbox)):
                    filtered_boxes.append(box1)
            else:
                filtered_boxes.append(box1)
    return torch.tensor(filtered_boxes)


@log_runtime
def annotate(
    image_source: np.ndarray,
    boxes: torch.Tensor,
    phrases: List[str],
    text_scale: float,
    text_padding: int = 5,
    text_thickness: int = 2,
    thickness: int = 3,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    h, w, _ = image_source.shape
    boxes = boxes * torch.Tensor([w, h, w, h])
    xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
    xywh = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xywh").numpy()
    detections = sv.Detections(xyxy=xyxy)  # type: ignore

    labels = [f"{i}" for i in range(boxes.shape[0])]

    from pixelpilot.util.box_annotator import BoxAnnotator

    box_annotator = BoxAnnotator(
        text_scale=text_scale, text_padding=text_padding, text_thickness=text_thickness, thickness=thickness
    )
    annotated_frame = image_source.copy()
    annotated_frame = box_annotator.annotate(
        scene=annotated_frame, detections=detections, labels=labels, image_size=(w, h)
    )

    label_coordinates = {f"{phrase}": v for phrase, v in zip(phrases, xywh)}
    return annotated_frame, label_coordinates


@log_runtime
def predict(model, image, caption, box_threshold, text_threshold):
    """Use huggingface model to replace the original model"""
    model, processor = model["model"], model["processor"]
    device = model.device

    inputs = processor(images=image, text=caption, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=box_threshold,  # 0.4,
        text_threshold=text_threshold,  # 0.3,
        target_sizes=[image.size[::-1]],
    )[0]
    boxes, logits, phrases = results["boxes"], results["scores"], results["labels"]
    return boxes, logits, phrases


@log_runtime
def predict_yolo(model, image_np, box_threshold):
    """Use YOLO model to predict boxes on numpy array image"""
    result = model.predict(
        source=image_np,
        conf=box_threshold,
    )
    boxes = result[0].boxes.xyxy
    conf = result[0].boxes.conf
    phrases = [str(i) for i in range(len(boxes))]

    return boxes, conf, phrases


@log_runtime
def get_som_labeled_img(
    image,
    model=None,
    box_threshold=0.01,
    output_coord_in_ratio=False,
    ocr_bbox=None,
    text_scale=0.4,
    text_padding=5,
    draw_bbox_config=None,
    caption_model_processor=None,
    ocr_text=[],
    iou_threshold=0.9,
    prompt=None,
):
    # Convert to numpy array if PIL Image
    if isinstance(image, Image.Image):
        image = image.convert("RGB")
        image_np = np.array(image)
    else:
        image_np = image

    # Process image and get boxes
    image_source = process_image_for_detection(image_np)
    h, w = image_source.shape[:2]
    xyxy, logits, phrases = get_detection_boxes(image_source, model, box_threshold)

    # Process OCR boxes
    filtered_boxes = process_ocr_boxes(xyxy, h, w, ocr_bbox, iou_threshold)

    # Get content labels (semantic processing only if Florence is enabled)
    parsed_content_merged = [f"Text Box ID {i}: {txt}" for i, txt in enumerate(ocr_text)]
    if caption_model_processor is not None:
        parsed_content_icon = get_parsed_content_icon(
            filtered_boxes, ocr_bbox, image_source, caption_model_processor, prompt=prompt
        )
        icon_start = len(ocr_text)
        parsed_content_icon_ls = [
            f"Icon Box ID {str(i+icon_start)}: {txt}" for i, txt in enumerate(parsed_content_icon)
        ]
        parsed_content_merged.extend(parsed_content_icon_ls)

    # Prepare final output
    filtered_boxes = box_convert(boxes=filtered_boxes, in_fmt="xyxy", out_fmt="cxcywh")
    phrases = [i for i in range(len(filtered_boxes))]
    # Draw boxes and get coordinates
    annotated_frame, label_coordinates = draw_boxes_and_labels(
        image_source,
        filtered_boxes,
        phrases,
        draw_bbox_config or {"text_scale": text_scale, "text_padding": text_padding},
    )

    # Format output
    encoded_image = encode_image_output(annotated_frame)
    if output_coord_in_ratio:
        label_coordinates = {k: [v[0] / w, v[1] / h, v[2] / w, v[3] / h] for k, v in label_coordinates.items()}

    return encoded_image, label_coordinates, parsed_content_merged


@log_runtime
def process_image_for_detection(image_np):
    """Process image for detection, ensuring RGB format."""
    if len(image_np.shape) == 3:
        if image_np.shape[2] == 4:  # RGBA
            image_source = image_np[:, :, :3]  # Drop alpha channel
        elif image_np.shape[2] == 3:  # RGB or BGR
            if image_np.dtype == np.uint8:  # Assuming BGR if uint8
                image_source = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
            else:
                image_source = image_np
        else:
            raise ValueError(f"Unexpected number of channels: {image_np.shape[2]}")
    else:
        raise ValueError("Image must be RGB/BGR/RGBA with 3 or 4 channels")
    return image_source


@log_runtime
def get_detection_boxes(image_source, model, box_threshold):
    """Get detection boxes from YOLO model."""
    h, w = image_source.shape[:2]
    xyxy, logits, phrases = predict_yolo(
        model=model,
        image_np=image_source,
        box_threshold=box_threshold,
    )
    xyxy = xyxy / torch.Tensor([w, h, w, h]).to(xyxy.device)
    phrases = [str(i) for i in range(len(phrases))]
    return xyxy, logits, phrases


@log_runtime
def process_ocr_boxes(xyxy, h, w, ocr_bbox, iou_threshold):
    """Process and normalize OCR boxes."""
    if ocr_bbox:
        logger.info(f"ocr_bbox: {ocr_bbox}")
        ocr_bbox = torch.tensor(ocr_bbox) / torch.Tensor([w, h, w, h])
        ocr_bbox = ocr_bbox.tolist()
    else:
        logger.info("ocr_bbox is None")
        ocr_bbox = None
    return remove_overlap(boxes=xyxy, iou_threshold=iou_threshold, ocr_bbox=ocr_bbox)


@log_runtime
def get_content_labels(
    use_local_semantics, filtered_boxes, ocr_bbox, image_source, caption_model_processor, ocr_text, prompt
):
    """Get content labels based on semantics setting."""
    if use_local_semantics and caption_model_processor:
        parsed_content_icon = get_parsed_content_icon(
            filtered_boxes, ocr_bbox, image_source, caption_model_processor, prompt=prompt
        )
        ocr_text = [f"Text Box ID {i}: {txt}" for i, txt in enumerate(ocr_text)]
        icon_start = len(ocr_text)
        parsed_content_icon_ls = []
        for i, txt in enumerate(parsed_content_icon):
            parsed_content_icon_ls.append(f"Icon Box ID {str(i+icon_start)}: {txt}")
        return ocr_text + parsed_content_icon_ls
    else:
        return [f"Text Box ID {i}: {txt}" for i, txt in enumerate(ocr_text)]


@log_runtime
def draw_boxes_and_labels(image_source, filtered_boxes, phrases, draw_config):
    """Draw boxes and labels on the image.

    Returns:
        tuple: (annotated_image, label_coordinates)
            - annotated_image: np.ndarray of the image with drawn boxes and labels
            - label_coordinates: dict mapping phrase IDs to their coordinates in xywh format
    """
    return annotate(
        image_source=image_source,
        boxes=filtered_boxes,
        phrases=phrases,
        text_scale=draw_config.get("text_scale", 0.4),
        text_padding=draw_config.get("text_padding", 5),
    )


@log_runtime
def encode_image_output(annotated_frame):
    """Encode the annotated image to base64."""
    pil_img = Image.fromarray(annotated_frame)
    buffered = io.BytesIO()
    pil_img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("ascii")


def get_xywh(input):
    x, y, w, h = input[0][0], input[0][1], input[2][0] - input[0][0], input[2][1] - input[0][1]
    x, y, w, h = int(x), int(y), int(w), int(h)
    return x, y, w, h


def get_xyxy(input):
    x, y, xp, yp = input[0][0], input[0][1], input[2][0], input[2][1]
    x, y, xp, yp = int(x), int(y), int(xp), int(yp)
    return x, y, xp, yp


def get_xywh_yolo(input):
    x, y, w, h = input[0], input[1], input[2] - input[0], input[3] - input[1]
    x, y, w, h = int(x), int(y), int(w), int(h)
    return x, y, w, h


@log_runtime
def check_ocr_box(
    image_np, display_img=True, output_bb_format="xywh", goal_filtering=None, easyocr_args=None, use_paddleocr=False
):
    if use_paddleocr:
        # result = paddle_ocr.ocr(image_path, cls=False)[0]
        # coord = [item[0] for item in result]
        # text = [item[1][0] for item in result]
        raise NotImplementedError("PaddleOCR not implemented")
    else:  # EasyOCR
        if easyocr_args is None:
            easyocr_args = {}
        result = reader.readtext(image_np, **easyocr_args)
        # print('goal filtering pred:', result[-5:])
        coord = [item[0] for item in result]
        text = [item[1] for item in result]
    # read the image using cv2
    if display_img:
        opencv_img = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        bb = []
        for item in coord:
            x, y, a, b = get_xywh(item)
            # print(x, y, a, b)
            bb.append((x, y, a, b))
            cv2.rectangle(opencv_img, (x, y), (x + a, y + b), (0, 255, 0), 2)

        # Display the image
        plt.imshow(opencv_img)
    else:
        if output_bb_format == "xywh":
            bb = [get_xywh(item) for item in coord]
        elif output_bb_format == "xyxy":
            bb = [get_xyxy(item) for item in coord]
        # print('bounding box!!!', bb)
    return (text, bb), goal_filtering
