#!/usr/bin/env python3
import time
import pdb

import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import String, Bool
import cv2

import torch
from torchvision import transforms
from torchvision.models.mobilenetv3 import MobileNet_V3_Small_Weights, MobileNet_V3_Large_Weights


class MobileNetDetector(Node):
    def __init__(self):
        super().__init__("mobilenet_detector")

        # Setup ROS Parameters
        self.declare_parameter("threshold", 0.8)
        self.declare_parameter("target_class", "street sign")
        self.declare_parameter("republish_img", False)
        self.declare_parameter(
            "classes_path", "~/tb_ws/src/asl_tb3_drivers/configs/imagenet_classes.txt"
        )

        self.publish_highlight = self.get_parameter("republish_img").value
        self.target_class = self.get_parameter("target_class").value
        self.threshold = self.get_parameter("threshold").value
        self.classes_path = self.get_parameter("classes_path").value

        # Load classes
        with open(self.classes_path, "r") as f:
            self.categories = [s.strip() for s in f.readlines()]

        try:
            self.target_class_index = self.categories.index(self.target_class)
        except ValueError as e:
            print("Class specified with target_class is not an imagenet 1k class!")
            raise e

        # load the model from torch hub
        self.model = torch.hub.load(
            "pytorch/vision:v0.10.0",
            "mobilenet_v3_large",
            weights=MobileNet_V3_Large_Weights.IMAGENET1K_V1,
        )

        # Check CUDA
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Set the model to eval mode and move to cuda
        self.model = self.model.to(self.device)
        self.model.eval()

        # Setup image preprocessing
        self.preprocess = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(256, antialias=True),
                transforms.CenterCrop(224),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        # CvBridge
        self.bridge = CvBridge()

        # Image subscriber
        self.image_sub = self.create_subscription(
            Image, "/image", self.image_callback, 10
        )

        self.detection_cat_pub = self.create_publisher(String, "/detector_top3", 10)
        self.detection_bool_pub = self.create_publisher(Bool, "/detector_bool", 10)

        if self.publish_highlight:
            self.highlight_pub = self.create_publisher(Image, "/detector_image", 10)

    def image_callback(self, img_msg):
        img = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")

        # Run model on image
        top3_p, top3_id, target_prob, inf_time = self.classify(img.copy())

        detection_bool = target_prob >= self.threshold

        # Publish detection message
        bool_msg = Bool()
        bool_msg.data = detection_bool
        self.detection_bool_pub.publish(bool_msg)

        # Top-K categories
        top3_msg = String()
        for i in range(3):
            top3_msg.data += (
                f"{self.categories[top3_id[i]]} [p={top3_p[i].item():0.2f}] |"
            )
        self.detection_cat_pub.publish(top3_msg)

        if self.publish_highlight:
            color = [0, 255, 0] if detection_bool else [0, 0, 255]
            img_highlight = cv2.copyMakeBorder(
                img, 10, 10, 10, 10, cv2.BORDER_CONSTANT, None, value=color
            )
            font = cv2.FONT_HERSHEY_SIMPLEX
            hl_text = f"{self.target_class} [p={target_prob:0.2f}] [{int(1e3 * inf_time)} ms]"
            cv2.putText(img_highlight, hl_text, (25, 30), font, 0.5, color)
            highlight_msg = self.bridge.cv2_to_imgmsg(
                img_highlight, encoding="bgr8"
            )
            self.highlight_pub.publish(highlight_msg)

    def classify(self, img_cv):
        start_time = time.perf_counter()
        img = self.preprocess(img_cv)
        img = img.unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(img)
        inference_time = time.perf_counter() - start_time

        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        top3_prob, top3_catid = torch.topk(probabilities, 3)

        target_prob = probabilities[self.target_class_index].item()

        return top3_prob, top3_catid, target_prob, inference_time

if __name__ == "__main__":
    rclpy.init()
    node = MobileNetDetector()
    rclpy.spin(node)
    rclpy.shutdown()


    MobileNetDetector()
