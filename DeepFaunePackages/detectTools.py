# Copyright CNRS 2023

# simon.chamaille@cefe.cnrs.fr; vincent.miele@univ-lyon1.fr

# This software is a computer program whose purpose is to identify
# animal species in camera trap images.

#This software is governed by the CeCILL  license under French law and
# abiding by the rules of distribution of free software.  You can  use, 
# modify and/ or redistribute the software under the terms of the CeCILL
# license as circulated by CEA, CNRS and INRIA at the following URL
# "http://www.cecill.info". 

# As a counterpart to the access to the source code and  rights to copy,
# modify and redistribute granted by the license, users are provided only
# with a limited warranty  and the software's author,  the holder of the
# economic rights,  and the successive licensors  have only  limited
# liability. 

# In this respect, the user's attention is drawn to the risks associated
# with loading,  using,  modifying and/or developing or reproducing the
# software by the user in light of its specific status of free software,
# that may mean  that it is complicated to manipulate,  and  that  also
# therefore means  that it is reserved for developers  and  experienced
# professionals having in-depth computer knowledge. Users are therefore
# encouraged to load and test the software's suitability as regards their
# requirements in conditions enabling the security of their systems and/or 
# data to be ensured and,  more generally, to use and operate it in the 
# same conditions as regards security. 

# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

YOLO_WIDTH = 960 # image width
YOLO_THRES = 0.6
YOLOHUMAN_THRES = 0.4 # boxes with human above this threshold are saved
YOLOCOUNT_THRES = 0.6
model = 'deepfaune-yolov8s_960.pt'

####################################################################################
### BEST BOX DETECTION 
####################################################################################
class Detector:
    def __init__(self):
        self.yolo = YOLO(model)
    """
    :param imagecv: openCV image in BGR
    :param threshold : above threshold, keep the best box given
    :return: cropped image, possibly None
    :rtype: PIL image
    """
    def bestBoxDetection(self, filename_or_imagecv, threshold=YOLO_THRES):
        try:
            results = self.yolo(filename_or_imagecv, verbose=False, imgsz=YOLO_WIDTH)
        except FileNotFoundError:
            return None, 0, np.zeros(4), 0, None
        except Exception as err:
            print(f"Unexpected {err=}, {type(err)=}")
            #raise
        # orig_img a numpy array (cv2) in BGR
        imagecv = results[0].cpu().orig_img
        detection = results[0].cpu().numpy().boxes
        if not len(detection.cls) or detection.conf[0] < threshold:
            return None, 0, np.zeros(4), 0, None
        ## best box
        category = detection.cls[0] + 1
        count = sum(detection.conf>YOLOCOUNT_THRES) # only if best box > YOLOTHRES
        box = detection.xyxy[0] # xmin, ymin, xmax, ymax
        croppedimagecv = cropSquareCV(imagecv, box.copy())
        croppedimage = Image.fromarray(croppedimagecv[:,:,(2,1,0)]) # converted to PIL BGR image
        if croppedimage is None: # FileNotFoundError
            category = 0
        ## count
        count = sum(detection.conf>YOLOCOUNT_THRES) # only if best box > YOLOTHRES
        ## human boxes
        ishuman = (detection.cls==1) & (detection.conf>=YOLOHUMAN_THRES)
        if any(ishuman==True):
            humanboxes = detection.xyxy[ishuman,]
            return croppedimage, category, box, count, humanboxes
        else:
            return croppedimage, category, box, count, None

####################################################################################
### BEST BOX DETECTION WITH JSON
####################################################################################
from load_api_results import load_api_results
import json
import contextlib
import os
from pandas import concat
from numpy import argmax
import sys

MDV5_THRES = 0.5

class DetectorJSON:
    """
    We assume JSON categories are 1=animal, 2=person, 3=vehicle and the empty category 0=empty

    :param jsonfilename: JSON file containing the bondoing boxes coordinates, such as generated by megadetectorv5
    :param threshold : above threshold, keep the best box
    """
    def __init__(self, jsonfilename):
        # getting results in a dataframe
        with contextlib.redirect_stdout(open(os.devnull, 'w')):
            try:
                self.df_json, _ = load_api_results(jsonfilename)
                # removing lines with Failure event
                if 'failure' in self.df_json.keys():
                    self.df_json = self.df_json[self.df_json['failure'].isnull()]
                    self.df_json.reset_index(drop=True, inplace = True)
                    self.df_json.drop('failure', axis=1, inplace=True)
            except json.decoder.JSONDecodeError:
                self.df_json = []
        self.k = 0 # current image index
        self.kbox = 0 # current box index
        self.imagecv = None

    """
    :return: cropped image, possibly None
    :rtype: PIL image
    """
    def nextBestBoxDetection(self, threshold=MDV5_THRES):
        if self.k >= len(self.df_json):
            raise IndexError # no next box
        if len(self.df_json['detections'][self.k]): # is non empty
            # Focus on the most confident bounding box coordinates
            self.kbox = argmax([box['conf'] for box in self.df_json['detections'][self.k]])
            if self.df_json['detections'][self.k][self.kbox]['conf']>threshold:
                category = int(self.df_json['detections'][self.k][self.kbox]['category'])
            else:
                category = 0
        else: # is empty
            category = 0
        # is an animal detected ?
        if category != 1:
            croppedimage = None
        # if yes, cropping the bounding box
        else:
            self.nextImread()
            croppedimage = self.cropCurrentBox()
            if croppedimage is None: # FileNotFoundError
                category = 0
        self.k += 1
        return croppedimage, category

    """
    :return: cropped image, possibly None
    :rtype: PIL image
    """
    def nextBoxDetection(self, threshold=MDV5_THRES):
        if self.k >= len(self.df_json):
            raise IndexError # no next box
        # is an animal detected ?
        if len(self.df_json['detections'][self.k]):
            if self.kbox == 0:
                self.nextImread() 
            # is box above threshold ?
            if self.df_json['detections'][self.k][self.kbox]['conf']>threshold:
                category = int(self.df_json['detections'][self.k][self.kbox]['category'])
                croppedimage = self.cropCurrentBox()
            else: # considered as empty
                category = 0
                croppedimage = None
            self.kbox += 1
            if self.kbox >= len(self.df_json['detections'][self.k]):
                self.k += 1
                self.kbox = 0
        else: # is empty
            category = 0
            croppedimage = None
            self.k += 1
            self.kbox = 0
        return croppedimage, category

    """
    :return: image from file
    :rtype: openCV image
    """
    def nextImread(self):
        try:
            self.imagecv = cv2.imdecode(np.fromfile(str(self.df_json["file"][self.k]), dtype=np.uint8),  cv2.IMREAD_UNCHANGED)
        except FileNotFoundError as e:
            print(e, file=sys.stderr)
            self.imagecv = None
    
    
    """
    :return: cropped image, possibly None
    :rtype: PIL image
    """
    def cropCurrentBox(self):
        if self.imagecv is None:
            return None
        image = Image.fromarray(cv2.cvtColor(self.imagecv, cv2.COLOR_BGR2RGB))
        box_norm = self.df_json['detections'][self.k][self.kbox]["bbox"]
        xmin = int(box_norm[0] * image.width)
        ymin = int(box_norm[1] * image.height)
        xmax = xmin + int(box_norm[2] * image.width)
        ymax = ymin + int(box_norm[3] * image.height)
        box = [xmin, ymin, xmax, ymax]
        croppedimage = cropSquare(image, box)
        return croppedimage
        
    def getNbFiles(self):
        return self.df_json.shape[0]
    
    def getFilenames(self):
        return list(self.df_json["file"].to_numpy())
    
    def getCurrentFilename(self):
        if self.k >= len(self.df_json):
            raise IndexError
        return self.df_json['file'][self.k]
    
    def resetDetection(self):
        self.k = 0
        self.kbox = 0
    
    def merge(self, detector):
        self.df_json = concat([self.df_json, detector.df_json], ignore_index=True)
        self.resetDetection()

  
####################################################################################
### TOOLS
####################################################################################      
'''
:return: cropped PIL image, as squared as possible (rectangle if close to the borders)
'''
def cropSquare(image, box):
    x1, y1, x2, y2 = box
    xsize = (x2-x1)
    ysize = (y2-y1)
    if xsize>ysize:
        y1 = y1-int((xsize-ysize)/2)
        y2 = y2+int((xsize-ysize)/2)
    if ysize>xsize:
        x1 = x1-int((ysize-xsize)/2)
        x2 = x2+int((ysize-xsize)/2)
    croppedimage = image.crop((max(0,x1), max(0,y1), min(x2,image.width), min(y2,image.height)))
    return croppedimage

'''
:return: cropped cv2 image, as squared as possible (rectangle if close to the borders)
'''
def cropSquareCV(imagecv, box):
    x1, y1, x2, y2 = box
    xsize = (x2-x1)
    ysize = (y2-y1)
    if xsize>ysize:
        y1 = y1-int((xsize-ysize)/2)
        y2 = y2+int((xsize-ysize)/2)
    if ysize>xsize:
        x1 = x1-int((ysize-xsize)/2)
        x2 = x2+int((ysize-xsize)/2)
    height, width, _ = imagecv.shape
    croppedimagecv = imagecv[max(0,int(y1)):min(int(y2),height),max(0,int(x1)):min(int(x2),width)]
    return croppedimagecv
