import sys
sys.path.append('../EfficientLoFTR')
import cv2
import numpy as np
import torch
from copy import deepcopy
from src.loftr import LoFTR, full_default_cfg, opt_default_cfg, reparameter
from src.utils.plotting import make_matching_figure
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from ultralytics import YOLO
# from utils import *
import time
from ransac import *
import math
params = [0,1600/2,800/2]

g8p = EightPointAlgorithmGeneralGeometry()
ransac = RANSAC_8PA()

def cam_from_img_vectorized(params, points):
    PI = 4 * math.atan(1)
    c1, c2 = params[1], params[2]
    theta = (points[:, 0] - c1) * PI / c1
    phi = (points[:, 1] - c2) * PI / (2 * c2)
    u = np.cos(theta) * np.cos(phi)
    v = np.sin(phi)
    w = np.sin(theta) * np.cos(phi)
    return np.column_stack((u, v, w))


class SphericalImageMatcher:
    def __init__(self, model_type='opt', precision='mp', img_width=1600, img_height=800):
        self.IMG_WIDTH = img_width
        self.IMG_HEIGHT = img_height
        self.model_type = model_type
        self.precision = precision
        self.matcher = self._initialize_matcher()
        self.yolo_model =  YOLO("/home/megumi/work/sahil/repos/E-LoFTR_SFM/yolo_weights/best.pt")

    def _initialize_matcher(self):
        if self.model_type == 'full':
            _default_cfg = deepcopy(full_default_cfg)
        elif self.model_type == 'opt':
            _default_cfg = deepcopy(opt_default_cfg)
        
        if self.precision == 'mp':
            _default_cfg['mp'] = True
        elif self.precision == 'fp16':
            _default_cfg['half'] = True

        matcher = LoFTR(config=_default_cfg)
        matcher.load_state_dict(torch.load("../EfficientLoFTR/weights/eloftr_outdoor.ckpt")['state_dict'])
        matcher = reparameter(matcher)
        if self.precision == 'fp16':
            matcher = matcher.half()
        return matcher.eval().cuda()

    def yolo_post_processing(self,preds,mkpts,mkpts1,mconf):
        bboxes=  np.array([])
        try:
            bboxes = np.array([
                [int(boxes.xyxy[0][0].cpu()), int(boxes.xyxy[0][1].cpu()), 
                    int(boxes.xyxy[0][2].cpu()), int(boxes.xyxy[0][3].cpu())]
                for result in preds
                for boxes in [result.boxes]
            ])
        except:
            pass

        if len(bboxes) > 0:
            keypoints = mkpts
            x_within = (keypoints[:, 0:1] >= bboxes[:, 0]) & (keypoints[:, 0:1] <= bboxes[:, 2])
            y_within = (keypoints[:, 1:2] >= bboxes[:, 1]) & (keypoints[:, 1:2] <= bboxes[:, 3])
            inside_bbox = np.any(x_within & y_within, axis=1)
    
            mkpts = mkpts[~inside_bbox]
            mkpts1 = mkpts1[~inside_bbox]
            mconf = mconf[~inside_bbox]

        return mkpts,mkpts1,mconf


    def post_process_kpts(self,img0_cubemap,img1_cubemap,mkpts0,mkpts1,mconf,pred_conf=0.2*100):
        preds_0 = self.yolo_model(img0_cubemap, verbose=False)
        preds_1 = self.yolo_model(img1_cubemap, verbose=False)
        mkpts0,mkpts1,mconf = self.yolo_post_processing(preds_0,mkpts0,mkpts1,mconf)
        mkpts1,mkpts0,mconf = self.yolo_post_processing(preds_1,mkpts1,mkpts0,mconf)
        ind = mconf>pred_conf
        mkpts0=mkpts0[ind]
        mkpts1=mkpts1[ind]
        mconf=mconf[ind]
        return mkpts0,mkpts1,mconf


    def match(self,img_0,img_1):
        img_0 = np.squeeze(np.array(img_0))
        img_1 = np.squeeze(np.array(img_1))
        img0_raw = cv2.cvtColor(img_0, cv2.COLOR_BGR2GRAY) if len(img_0.shape) == 3 else img_0
        img1_raw = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY) if len(img_1.shape) == 3 else img_1

        img0_raw = cv2.resize(img0_raw, (img0_raw.shape[1]//32*32, img0_raw.shape[0]//32*32))
        img1_raw = cv2.resize(img1_raw, (img1_raw.shape[1]//32*32, img1_raw.shape[0]//32*32))

        if self.precision == 'fp16':
            img0 = torch.from_numpy(img0_raw)[None][None].half().cuda() / 255.
            img1 = torch.from_numpy(img1_raw)[None][None].half().cuda() / 255.
        else:
            img0 = torch.from_numpy(img0_raw)[None][None].cuda() / 255.
            img1 = torch.from_numpy(img1_raw)[None][None].cuda() / 255.


        batch = {
            'image0': img0,
            'image1': img1,
        }
        with torch.no_grad():
            if self.precision == 'mp':
                with torch.autocast(enabled=True, device_type='cuda'):
                    self.matcher(batch)
            else:
                self.matcher(batch)
            mkpts0 = batch['mkpts0_f'].cpu().numpy()
            mkpts1 = batch['mkpts1_f'].cpu().numpy()
            mconf = batch['mconf'].cpu().numpy()
        # print (img0.shape)
        mkpts0,mkpts1,mconf = self.post_process_kpts(img_0,img_1,mkpts0,mkpts1,mconf)
        return {"keypoints0":mkpts0,"keypoints1":mkpts1,"scores":mconf}


    def match_batch(self, img_pairs, batch_size=4):
        results = []

        img0_batch = []
        img1_batch = []
        img0_batch_raw = []
        img1_batch_raw = []

        for img_0, img_1 in img_pairs:
            img0_raw = cv2.cvtColor(img_0.numpy(), cv2.COLOR_BGR2GRAY) if len(img_0.shape) == 3 else img_0.numpy()
            img1_raw = cv2.cvtColor(img_1.numpy(), cv2.COLOR_BGR2GRAY) if len(img_1.shape) == 3 else img_1.numpy()

            img0_raw = cv2.resize(img0_raw, (img0_raw.shape[1] // 32 * 32, img0_raw.shape[0] // 32 * 32))
            img1_raw = cv2.resize(img1_raw, (img1_raw.shape[1] // 32 * 32, img1_raw.shape[0] // 32 * 32))

            if self.precision == 'fp16':
                img0 = torch.from_numpy(img0_raw)[None][None].half().cuda() / 255.
                img1 = torch.from_numpy(img1_raw)[None][None].half().cuda() / 255.
            else:
                img0 = torch.from_numpy(img0_raw)[None][None].cuda() / 255.
                img1 = torch.from_numpy(img1_raw)[None][None].cuda() / 255.

            img0_batch_raw.append(img0)
            img1_batch_raw.append(img1)
            img0_batch.append(img_0)
            img1_batch.append(img_1)

            if len(img0_batch) == batch_size:
                results.extend(self._process_batch(img0_batch_raw, img1_batch_raw, img0_batch, img1_batch))
                img0_batch = []
                img1_batch = []
                img0_batch_raw = []
                img1_batch_raw = []

        if len(img0_batch) > 0:
            results.extend(self._process_batch(img0_batch_raw, img1_batch_raw, img0_batch, img1_batch))

        return results

    def _process_batch(self, img0_batch_raw, img1_batch_raw, img0_batch, img1_batch):
        batch_results = []
        img0_tensor = torch.cat(img0_batch_raw, dim=0)
        img1_tensor = torch.cat(img1_batch_raw, dim=0)
        batch = {
            'image0': img0_tensor,
            'image1': img1_tensor,
        }

        with torch.no_grad():
            if self.precision == 'mp':
                with torch.autocast(enabled=True, device_type='cuda'):
                    t1 = time.time()
                    self.matcher(batch)
                    # print (time.time()-t1)
            else:
                self.matcher(batch)

            mkpts0_batch = batch['mkpts0_f'].cpu().numpy()
            mkpts1_batch = batch['mkpts1_f'].cpu().numpy()
            mconf_batch = batch['mconf'].cpu().numpy()
            b_ids = batch['b_ids'].cpu().numpy()

        for i in range(len(img0_batch)):
            mask = b_ids == i
            mkpts0 = mkpts0_batch[mask]
            mkpts1 = mkpts1_batch[mask]
            mconf = mconf_batch[mask]
            points0_spherical = cam_from_img_vectorized(params,mkpts0)
            points1_spherical = cam_from_img_vectorized(params,mkpts1)
            # print ()
            inliers = ransac.get_inliers(points0_spherical.T,points1_spherical.T)
            ransac.reset()
            mkpts0 = mkpts0[inliers]
            mkpts1 = mkpts1[inliers]
            mconf = mconf[inliers]
            img0 = img0_batch[i]
            img1 = img1_batch[i]

            # mkpts0, mkpts1, mconf = self.post_process_kpts(img0, img1, mkpts0, mkpts1, mconf)
            batch_results.append({
                'keypoints0': mkpts0,
                'keypoints1': mkpts1,
                'scores': mconf
            })

        return batch_results