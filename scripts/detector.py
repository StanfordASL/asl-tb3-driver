#!/usr/bin/env python3
import time
import pdb

import numpy as np
import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, String
import cv2
import os

import torch
from torchvision import transforms
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from torchvision.utils import draw_bounding_boxes

COCO_LABELS = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]


class MobileNetDetector(Node):
    def __init__(self):
        super().__init__("mobilenet_detector")

        # Setup ROS Parameters
        self.declare_parameter("threshold", 0.5)
        self.declare_parameter("target_classes", ["stop sign", "traffic light"])
        self.declare_parameter("republish_img", True)

        # Check CUDA
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        model_path = os.path.expanduser("~/section_assets/finetuned_ssd_model.pkl")
        if os.path.exists(model_path):
            self.get_logger().info("Using uploaded finetuned model")
            loaded_model = ssdlite320_mobilenet_v3_large(pretrained=False)
            state_dict = torch.load(model_path, map_location=self.device)
            loaded_model.load_state_dict(state_dict)
            self.model = loaded_model.to(self.device)
        else:
            self.get_logger().info("Using default SSD model")
            self.model = ssdlite320_mobilenet_v3_large(pretrained=True)

            # Set the model to eval mode and move to cuda
            self.model = self.model.to(self.device)
        
        self.model.eval()

        # Setup image preprocessing
        self.preprocess = transforms.Compose([transforms.ToTensor()])
        self.bridge = CvBridge()

        # Publishers
        self.detection_bool_pub = self.create_publisher(Bool, "/detector_bool", 10)
        self.detection_class_pub = self.create_publisher(String, "/detector_class", 10)
        if self.republish_img:
            self.highlight_pub = self.create_publisher(Image, "/detector_image", 10)

        # Subscriber
        self.image_sub = self.create_subscription(
            Image, "/image", self.image_callback, 10
        )

    @property
    def republish_img(self) -> bool:
        return self.get_parameter("republish_img").value

    @property
    def threshold(self) -> float:
        return self.get_parameter("threshold").value

    @property
    def target_classes(self) -> str:
        return self.get_parameter("target_classes").value

    def image_callback(self, img_msg):
        img_cv = self.bridge.imgmsg_to_cv2(img_msg, "rgb8")
        # self.get_logger().info(f"Running inference on: {self.device}")

        # Run detection model on image
        start_time = time.perf_counter()
        img_cpu = self.preprocess(img_cv)
        img_cuda = img_cpu.unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(img_cuda)[0]
        inference_time = time.perf_counter() - start_time

        # filter with threshold
        valid_mask = output["scores"] > self.threshold
        boxes = output["boxes"][valid_mask]
        scores = output["scores"][valid_mask]
        labels = output["labels"][valid_mask].cpu().numpy()

        colors = []
        viz_texts = []
        detection_bool = Bool()
        detection_class = String()

        for i, label_idx in enumerate(labels):
            label = COCO_LABELS[label_idx]
            if label in self.target_classes:
                colors.append((0, 255, 0))
                detection_bool.data = True
                detection_class.data = label
            else:
                colors.append((0, 0, 255))
            viz_texts.append(f"{label} [p={scores[i]:.3f}]")

        # Publish detection status
        self.detection_bool_pub.publish(detection_bool)
        if detection_bool.data:
            self.detection_class_pub.publish(detection_class)

        # Visualize if enabled
        if self.republish_img:
            img_viz = draw_bounding_boxes(
                torch.from_numpy(img_cv.transpose(2, 0, 1)),
                boxes,
                viz_texts,
                colors
            )
            img_viz = img_viz.cpu().numpy().transpose(1, 2, 0)
            
            # Add inference time
            color = [0, 255, 0]
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img_viz, f"[{int(1e3 * inference_time)} ms]", 
                       (25, 30), font, 0.5, color)
            
            highlight_msg = self.bridge.cv2_to_imgmsg(img_viz, encoding="rgb8")
            self.highlight_pub.publish(highlight_msg)

if __name__ == "__main__":
    rclpy.init()
    node = MobileNetDetector()
    rclpy.spin(node)
    rclpy.shutdown()

