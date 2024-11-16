from typing import List
from typing import NamedTuple
from typing import Optional
from typing import Tuple
from typing import Union

import cv2
import numpy as np
from supervision.detection.core import Detections
from supervision.draw.color import Color
from supervision.draw.color import ColorPalette


class BoxAnnotator:
    """
    A class for drawing bounding boxes on an image using detections provided.

    Attributes:
        color (Union[Color, ColorPalette]): The color to draw the bounding box,
            can be a single color or a color palette
        thickness (int): The thickness of the bounding box lines, default is 2
        text_color (Color): The color of the text on the bounding box, default is white
        text_scale (float): The scale of the text on the bounding box, default is 0.5
        text_thickness (int): The thickness of the text on the bounding box,
            default is 1
        text_padding (int): The padding around the text on the bounding box,
            default is 5

    """

    def __init__(
        self,
        color: Union[Color, ColorPalette] = ColorPalette.DEFAULT,
        thickness: int = 3,  # 1 for seeclick 2 for mind2web and 3 for demo
        text_color: Color = Color.BLACK,
        text_scale: float = 0.5,  # 0.8 for mobile/web, 0.3 for desktop # 0.4 for mind2web
        text_thickness: int = 2,  # 1, # 2 for demo
        text_padding: int = 10,
        avoid_overlap: bool = True,
    ):
        self.color: Union[Color, ColorPalette] = color
        self.thickness: int = thickness
        self.text_color: Color = text_color
        self.text_scale: float = text_scale
        self.text_thickness: int = text_thickness
        self.text_padding: int = text_padding
        self.avoid_overlap: bool = avoid_overlap

    def annotate(
        self,
        scene: np.ndarray,
        detections: Detections,
        labels: Optional[List[str]] = None,
        skip_label: bool = False,
        image_size: Optional[Tuple[int, int]] = None,
    ) -> np.ndarray:
        """
        Draws bounding boxes on the frame using the detections provided.

        Args:
            scene (np.ndarray): The image on which the bounding boxes will be drawn
            detections (Detections): The detections for which the
                bounding boxes will be drawn
            labels (Optional[List[str]]): An optional list of labels
                corresponding to each detection. If `labels` are not provided,
                corresponding `class_id` will be used as label.
            skip_label (bool): Is set to `True`, skips bounding box label annotation.
        Returns:
            np.ndarray: The image with the bounding boxes drawn on it

        Example:
            ```python
            import supervision as sv

            classes = ['person', ...]
            image = ...
            detections = sv.Detections(...)

            box_annotator = sv.BoxAnnotator()
            labels = [
                f"{classes[class_id]} {confidence:0.2f}"
                for _, _, confidence, class_id, _ in detections
            ]
            annotated_frame = box_annotator.annotate(
                scene=image.copy(),
                detections=detections,
                labels=labels
            )
            ```
        """
        font = cv2.FONT_HERSHEY_SIMPLEX
        for i in range(len(detections)):
            x1, y1, x2, y2 = detections.xyxy[i].astype(int)
            class_id = detections.class_id[i] if detections.class_id is not None else None
            idx = class_id if class_id is not None else i
            color = self.color.by_idx(idx) if isinstance(self.color, ColorPalette) else self.color
            cv2.rectangle(
                img=scene,
                pt1=(x1, y1),
                pt2=(x2, y2),
                color=color.as_bgr(),
                thickness=self.thickness,
            )
            if skip_label:
                continue

            text = f"{class_id}" if (labels is None or len(detections) != len(labels)) else labels[i]

            text_width, text_height = cv2.getTextSize(
                text=text,
                fontFace=font,
                fontScale=self.text_scale,
                thickness=self.text_thickness,
            )[0]

            if not self.avoid_overlap:
                text_x = x1 + self.text_padding
                text_y = y1 - self.text_padding

                text_background_x1 = x1
                text_background_y1 = y1 - 2 * self.text_padding - text_height

                text_background_x2 = x1 + 2 * self.text_padding + text_width
                text_background_y2 = y1
                # text_x = x1 - self.text_padding - text_width
                # text_y = y1 + self.text_padding + text_height
                # text_background_x1 = x1 - 2 * self.text_padding - text_width
                # text_background_y1 = y1
                # text_background_x2 = x1
                # text_background_y2 = y1 + 2 * self.text_padding + text_height
            else:
                text_x, text_y, text_background_x1, text_background_y1, text_background_x2, text_background_y2 = (
                    get_optimal_label_pos(
                        self.text_padding, text_width, text_height, x1, y1, x2, y2, detections, image_size
                    )
                )

            cv2.rectangle(
                img=scene,
                pt1=(text_background_x1, text_background_y1),
                pt2=(text_background_x2, text_background_y2),
                color=color.as_bgr(),
                thickness=cv2.FILLED,
            )
            # import pdb; pdb.set_trace()
            box_color = color.as_rgb()
            luminance = 0.299 * box_color[0] + 0.587 * box_color[1] + 0.114 * box_color[2]
            text_color = (0, 0, 0) if luminance > 160 else (255, 255, 255)
            cv2.putText(
                img=scene,
                text=text,
                org=(text_x, text_y),
                fontFace=font,
                fontScale=self.text_scale,
                # color=self.text_color.as_rgb(),
                color=text_color,
                thickness=self.text_thickness,
                lineType=cv2.LINE_AA,
            )
        return scene


def box_area(box):
    return (box[2] - box[0]) * (box[3] - box[1])


def intersection_area(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    return max(0, x2 - x1) * max(0, y2 - y1)


def IoU(box1, box2, return_max=True):
    intersection = intersection_area(box1, box2)
    union = box_area(box1) + box_area(box2) - intersection
    if box_area(box1) > 0 and box_area(box2) > 0:
        ratio1 = intersection / box_area(box1)
        ratio2 = intersection / box_area(box2)
    else:
        ratio1, ratio2 = 0, 0
    if return_max:
        return max(intersection / union, ratio1, ratio2)
    else:
        return intersection / union


class LabelPosition(NamedTuple):
    """Represents coordinates for label text and background."""

    text_x: int
    text_y: int
    bg_x1: int
    bg_y1: int
    bg_x2: int
    bg_y2: int


def get_optimal_label_pos(
    text_padding: int,
    text_width: int,
    text_height: int,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    detections: Detections,
    image_size: Tuple[int, int],
) -> Tuple[int, int, int, int, int, int]:
    """
    Find optimal position for label text that avoids overlaps with detections
    and image boundaries.

    Tries positions in this order:
    1. Top left of bounding box
    2. Outer left of bounding box
    3. Outer right of bounding box
    4. Top right of bounding box

    If all positions overlap, returns the last position (top right).

    Args:
        text_padding: Padding around the text
        text_width: Width of the text
        text_height: Height of the text
        x1, y1, x2, y2: Coordinates of the bounding box
        detections: All detection boxes to check for overlaps
        image_size: (width, height) of the image

    Returns:
        Tuple of (
            text_x, text_y,
            background_x1, background_y1, background_x2, background_y2
        )
    """
    positions = [
        # Top left
        LabelPosition(
            text_x=x1 + text_padding,
            text_y=y1 - text_padding,
            bg_x1=x1,
            bg_y1=y1 - 2 * text_padding - text_height,
            bg_x2=x1 + 2 * text_padding + text_width,
            bg_y2=y1,
        ),
        # Outer left
        LabelPosition(
            text_x=x1 - text_padding - text_width,
            text_y=y1 + text_padding + text_height,
            bg_x1=x1 - 2 * text_padding - text_width,
            bg_y1=y1,
            bg_x2=x1,
            bg_y2=y1 + 2 * text_padding + text_height,
        ),
        # Outer right
        LabelPosition(
            text_x=x2 + text_padding,
            text_y=y1 + text_padding + text_height,
            bg_x1=x2,
            bg_y1=y1,
            bg_x2=x2 + 2 * text_padding + text_width,
            bg_y2=y1 + 2 * text_padding + text_height,
        ),
        # Top right
        LabelPosition(
            text_x=x2 - text_padding - text_width,
            text_y=y1 - text_padding,
            bg_x1=x2 - 2 * text_padding - text_width,
            bg_y1=y1 - 2 * text_padding - text_height,
            bg_x2=x2,
            bg_y2=y1,
        ),
    ]

    def is_valid_position(pos: LabelPosition) -> bool:
        # Check image boundaries
        if pos.bg_x1 < 0 or pos.bg_x2 > image_size[0] or pos.bg_y1 < 0 or pos.bg_y2 > image_size[1]:
            return False

        # Check overlaps with all detections
        bg_box = [pos.bg_x1, pos.bg_y1, pos.bg_x2, pos.bg_y2]
        for detection in detections.xyxy:
            if IoU(bg_box, detection.astype(int)) > 0.3:
                return False
        return True

    # Return first valid position, or last position if none are valid
    for pos in positions:
        if is_valid_position(pos):
            return (pos.text_x, pos.text_y, pos.bg_x1, pos.bg_y1, pos.bg_x2, pos.bg_y2)

    last_pos = positions[-1]
    return (last_pos.text_x, last_pos.text_y, last_pos.bg_x1, last_pos.bg_y1, last_pos.bg_x2, last_pos.bg_y2)
