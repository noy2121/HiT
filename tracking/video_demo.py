import os
import sys
import time
import argparse
import numpy as np
import cv2 as cv
import onnx
import onnxruntime
env_path = os.path.join(os.path.dirname(__file__), '..')
if env_path not in sys.path:
    sys.path.append(env_path)
os.environ["CUDA_VISIBLE_DEVICES"]= "0"

import logging
def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
from lib.test.tracker.basetracker import BaseTracker
import torch
from lib.test.tracker.vittrack_utils import sample_target
# for debug
import os
import lib.models.HiT.levit_utils as utils
from lib.models.HiT import build_hit
from lib.test.tracker.vittrack_utils import Preprocessor
from lib.utils.box_ops import clip_box

logger = logging.getLogger(__name__)


class HiT:
    def __init__(self, model_path):
        onnx_model= onnx.load(model_path)
        onnx.checker.check_model(onnx_model)
        with torch.no_grad():
            net = onnxruntime.InferenceSession(model_path,providers=['CUDAExecutionProvider'])
        self.net = net
        self.preprocessor = Preprocessor()
        self.state = None
        # for debug
        # self.debug = False
        self.frame_id = 0
        # for save boxes from all queries
        self.z_dict1 = {}

    def initialize(self, image, info: dict):
        # forward the template once
        # info['init_bbox']: list [x0,y0,w,h] example: [367.0, 101.0, 41.0, 16.0]
        z_patch_arr, _ = sample_target(image, info['init_bbox'], 2.0, 128)
        self.template = self.preprocessor.process(z_patch_arr)
        self.state = info['init_bbox']
        self.frame_id = 0

    def track(self, image, info: dict = None):
        H, W, _ = image.shape
        self.frame_id += 1
        x_patch_arr, resize_factor = sample_target(image, self.state, 4.0, 256)  # (x1, y1, w, h)
        search = self.preprocessor.process(x_patch_arr)
        ort_inputs = {'search': to_numpy(search).astype(np.float32),
                      'template': to_numpy(self.template).astype(np.float32)
                      }
        with torch.no_grad():
            ort_outs = self.net.run(None, ort_inputs)

        pred_boxes = torch.from_numpy(ort_outs[0]).view(-1, 4)
        # Baseline: Take the mean of all pred boxes as the final result
        pred_box = (pred_boxes.mean(dim=0) * 256 / resize_factor).tolist()  # (cx, cy, w, h) [0,1]
        # get the final box result
        self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)
        # self.state: list [x0,y0,w,h,] example: [365.4537048339844, 102.24719142913818, 47.13159942626953, 15.523386001586914]
        return {"target_bbox": self.state}

    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * 256 / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

    def map_box_back_batch(self, pred_box: torch.Tensor, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box.unbind(-1) # (N,4) --> (N,)
        half_side = 0.5 * 256 / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return torch.stack([cx_real - 0.5 * w, cy_real - 0.5 * h, w, h], dim=-1)


def get_tracker_class():
    return HiT


def get_hit_tracker_class():
    return HiT


def save_predictions(output_boxes, output_dir):
    """Save predictions. one txt file, one bbox per line."""
    predictions_file = os.path.join(output_dir, 'predictions.txt')
    with open(predictions_file, 'w') as f:
        for frame_id, bbox in enumerate(output_boxes):
            # frame_id, x, y, w, h
            f.write(f"{frame_id+1},{bbox[0]:.3f},{bbox[1]:.3f},{bbox[2]:.3f},{bbox[3]:.3f}\n")
    return predictions_file

        
def run_video(model_path, videofilepath, output_dir=None):
    """Run the tracker with the vieofile.
    args:
        net_path: Path to ONNX model
        videofilepath: Path to video file
        output_dir: Directory to save results
        logger: Logger instance
    """
    optional_box = None
    # net = NetWithBackbone(net_path=net_path, use_gpu=True)
    tracker = HiT(model_path)

    # create dataset
    assert os.path.isfile(videofilepath), "Invalid param {}".format(videofilepath)
    ", videofilepath must be a valid videofile"

    output_boxes = []

    cap = cv.VideoCapture(videofilepath)
    display_name = 'Display: ' +'mobiletrack'
    cv.namedWindow(display_name, cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO)
    cv.resizeWindow(display_name, 960, 720)
    success, frame = cap.read()
    cv.imshow(display_name, frame)

    def _build_init_info(box):
        return {'init_bbox': box}

    if success is not True:
        if logger:
            logger.error(f"Read frame from {videofilepath} failed.")
        else:
            print(f"Read frame from {videofilepath} failed.")
            sys.exit(-1)
    if optional_box is not None:
        assert isinstance(optional_box, (list, tuple))
        assert len(optional_box) == 4, "valid box's foramt is [x,y,w,h]"
        tracker.initialize(frame, _build_init_info(optional_box))
        output_boxes.append(optional_box)
    else:
        while True:
            # cv.waitKey()
            frame_disp = frame.copy()

            cv.putText(frame_disp, 'Select target ROI and press ENTER', (20, 30), cv.FONT_HERSHEY_COMPLEX_SMALL,
                        1.5, (0, 0, 0), 1)

            x, y, w, h = cv.selectROI(display_name, frame_disp, fromCenter=False)
            init_state = [x, y, w, h]
            tracker.initialize(frame, _build_init_info(init_state))
            output_boxes.append(init_state)
            break
    
    torch.cuda.synchronize()
    start_all = time.time()
    tracker_time_total = 0
    frame_count = 0
    while True:
        ret, frame = cap.read()

        if frame is None:
            break

        frame_disp = frame.copy()

        # Draw box
        torch.cuda.synchronize()
        start_t = time.time()
        out = tracker.track(frame)
        torch.cuda.synchronize()
        end_t = time.time()
        tracker_time_total += (end_t - start_t)
        frame_count += 1
        state = [int(s) for s in out['target_bbox']]
        output_boxes.append(state)

        cv.rectangle(frame_disp, (state[0], state[1]), (state[2] + state[0], state[3] + state[1]),
                        (0, 255, 0), 5)

        font_color = (0, 0, 0)
        cv.putText(frame_disp, 'Tracking!', (20, 30), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                    font_color, 1)
        cv.putText(frame_disp, 'Press r to reset', (20, 55), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                    font_color, 1)
        cv.putText(frame_disp, 'Press q to quit', (20, 80), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                    font_color, 1)

        # Display the resulting frame
        cv.imshow(display_name, frame_disp)
        key = cv.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('r'):
            ret, frame = cap.read()
            frame_disp = frame.copy()

            cv.putText(frame_disp, 'Select target ROI and press ENTER', (20, 30), cv.FONT_HERSHEY_COMPLEX_SMALL, 1.5,
                        (0, 0, 0), 1)

            cv.imshow(display_name, frame_disp)
            x, y, w, h = cv.selectROI(display_name, frame_disp, fromCenter=False)
            init_state = [x, y, w, h]
            tracker.initialize(frame, _build_init_info(init_state))
            output_boxes.append(init_state)

    # When everything done, release the capture
    cap.release()
    cv.destroyAllWindows()
    torch.cuda.synchronize()
    end_all = time.time()
    total_time = end_all - start_all

    logger.info("--- Performance Metrics ---")
    logger.info(f"Total frames: {frame_count}")
    logger.info(f"Total time: {total_time:.2f} sec")
    logger.info(f"Tracker-only time: {tracker_time_total:.2f} sec")
    logger.info(f"FPS (total): {frame_count / total_time:.2f}")
    logger.info(f"FPS (tracker-only): {frame_count / tracker_time_total:.2f}")

    # Save predictions if output directory is provided
    if output_dir:
        predictions_file = save_predictions(output_boxes, output_dir)
        logger.info(f"Predictions saved to: {predictions_file}")
        return predictions_file

    return None


def main():
    parser = argparse.ArgumentParser(description='Run the tracker on your webcam.')
    parser.add_argument('tracker_path', type=str, help='Name of tracking method.')
    parser.add_argument('videofile', type=str, help='path to a video file.')
    parser.add_argument('--optional_box', type=float, default=None, nargs="+", help='optional_box with format x y w h.')
    parser.add_argument('--debug', type=int, default=0, help='Debug level.')
    parser.add_argument('--save_results', dest='save_results', action='store_true', help='Save bounding boxes')
    parser.set_defaults(save_results=False)

    args = parser.parse_args()

    run_video(args.tracker_path, args.videofile)


if __name__ == '__main__':
    main()
