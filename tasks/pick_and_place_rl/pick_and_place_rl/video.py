"""Record the Gazebo observer camera to an mp4, paced by simulation time."""

import threading

import cv2
import numpy as np
from sensor_msgs.msg import Image


class VideoRecorder:
    """Samples `/observer/image_raw` at a fixed *sim-time* rate.

    The sim may run slower than real time, so frames are picked on a sim-time grid;
    the resulting video plays back at true simulated speed.
    """

    def __init__(self, scene, path, topic="/observer/image_raw", fps=20):
        self.scene = scene
        self.path = path
        self.fps = fps
        self.caption = ""
        self._writer = None
        self._lock = threading.Lock()
        self._next_t = None
        self.frames = 0
        self._sub = scene.node.create_subscription(Image, topic, self._on_image, 5)
        self._active = False

    def start(self):
        self._active = True
        self._next_t = None

    def stop(self):
        self._active = False
        with self._lock:
            if self._writer is not None:
                self._writer.release()
                self._writer = None

    def _on_image(self, msg: Image):
        if not self._active:
            return
        t = self.scene.now()
        if self._next_t is not None and t < self._next_t:
            return
        self._next_t = t + 1.0 / self.fps if self._next_t is None else self._next_t + 1.0 / self.fps
        if t > self._next_t:  # fell behind: resync
            self._next_t = t + 1.0 / self.fps
        img = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, -1)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if msg.encoding == "rgb8" else img
        img = np.ascontiguousarray(img)
        if self.caption:
            cv2.rectangle(img, (0, 0), (img.shape[1], 34), (0, 0, 0), -1)
            cv2.putText(img, self.caption, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        with self._lock:
            if self._writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                self._writer = cv2.VideoWriter(self.path, fourcc, self.fps, (msg.width, msg.height))
            self._writer.write(img)
            self.frames += 1
