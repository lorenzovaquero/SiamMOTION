"""videoLoader.py: Manipulation of multiple video formats and devices"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import tempfile
import shutil
import cv2
import glob
import numpy as np
import logging
import errno

logger = logging.getLogger(__name__)

CURRENT_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(CURRENT_DIR, '..'))

from misc.region_utils import region_to_aabb
from misc.url_utils import is_url

__author__ = "Lorenzo Vaquero Otal"
__credits__ = ["Lorenzo Vaquero Otal"]
__date__ = "2023"
__version__ = "1.0.0"
__maintainer__ = "Lorenzo Vaquero Otal"
__contact__ = "lorenzo.vaquero.otal@usc.es"
__status__ = "Prototype"


class Video(object):
    """Allows the manipulation of multiple video formats and devices"""

    def __init__(self, path, is_scheduled=True, webcam_number=None):
        logger.debug('Opening video')

        self.groundtruth_line = None
        self.current_groundtruth_number = None
        self.current_frame = None
        self.current_frame_number = None
        self.current_groundtruth = None

        self.webcam_number = None
        self.path = None
        self.is_scheduled = None
        self.name = None
        self.frames = None
        self.frames_path = None
        self.frame_number = None
        self.has_groundtruth = None
        self.is_vot_video = None
        self.__remove_folder = None
        self.__frame_files = None
        self.__videocapture = None
        self.groundtruth_file = None
        self.groundtruth_path = None

        logger.debug('Opening video capturer')

        self.__videocapture = cv2.VideoCapture(self.webcam_number)

        if webcam_number is not None:
            self.webcam_number = webcam_number
            self.path = None
            self.is_scheduled = False
            self.__open_webcam()

        elif is_url(path):
            self.webcam_number = path
            self.path = path
            self.is_scheduled = False
            self.__open_webcam()

        else:
            self.path = path
            self.is_scheduled = is_scheduled
            self.webcam_number = None
            self.__get_video_paths()
            self.__open()

        logger.debug('Video opened')

    def __get_video_paths(self):
        if self.path == '' or self.path is None:
            logger.error('Video does not exist')
            raise OSError(errno.ENOENT, os.strerror(errno.ENOENT), self.path)

        self.path = os.path.abspath(self.path)
        self.name = os.path.basename(self.path)

        if os.path.isdir(self.path):
            logger.debug('VOT video detected')

            self.is_vot_video = True
            if os.path.isfile(os.path.join(self.path, 'groundtruth.txt')):
                self.groundtruth_path = os.path.join(self.path, 'groundtruth.txt')
                self.has_groundtruth = True
            elif os.path.isfile(os.path.join(self.path, 'groundtruth_rect.txt')):
                self.groundtruth_path = os.path.join(self.path, 'groundtruth_rect.txt')
                self.has_groundtruth = True
            else:
                self.has_groundtruth = False

            if self.has_groundtruth:
                logger.debug('Groundtruth file found')

            else:
                logger.debug('Groundtruth file not found')

            if os.path.isdir(os.path.join(self.path, 'imgs')):
                logger.debug('Frame files found in an alternate folder')
                self.frames_path = os.path.join(self.path, 'imgs')

            elif os.path.isdir(os.path.join(self.path, 'img')):
                logger.debug('Frame files found in an alternate folder')

                self.frames_path = os.path.join(self.path, 'img')

            elif os.path.isdir(os.path.join(self.path, 'color')):
                logger.debug('Frame files found in an alternate folder')

                self.frames_path = os.path.join(self.path, 'color')

            else:
                self.frames_path = self.path

        elif os.path.isfile(self.path):
            logger.debug('mp4 video found')

            self.is_vot_video = False

            if os.path.isfile(os.path.join(os.path.dirname(self.path), 'groundtruth.txt')):
                self.groundtruth_path = os.path.join(os.path.dirname(self.path), 'groundtruth.txt')
                self.has_groundtruth = True
            else:
                self.has_groundtruth = False

            if self.has_groundtruth:
                logger.debug('Groundtruth file found')
            else:
                logger.debug('Groundtruth file not found')

            if self.is_scheduled:
                logger.debug('Created a temp folder to extract the video frames')

                self.__remove_folder = True
                self.frames_path = tempfile.mkdtemp()

        else:
            logger.error('Video does not exist')
            raise OSError(errno.ENOENT, os.strerror(errno.ENOENT), self.path)

    def __open(self):
        self.__remove_folder = False

        if self.has_groundtruth:
            logger.debug('Opening groundtruth file')

            logger.debug('Loading first groundtruth')
            self.groundtruth_file = open(self.groundtruth_path, 'r')
            self.groundtruth_line = self.groundtruth_file.readline()

            if ',' in self.groundtruth_line:
                region = [float(i) for i in self.groundtruth_line.strip().split(",")]
            elif '\t' in self.groundtruth_line:
                region = [float(i) for i in self.groundtruth_line.strip().split("\t")]
            else:
                region = [float(i) for i in self.groundtruth_line.strip().split(" ")]

            self.current_groundtruth = region_to_aabb(region)
            self.current_groundtruth_number = 1

        else:
            self.current_groundtruth = None

        self.current_frame_number = 1
        if self.is_scheduled:
            logger.debug('Loading video frames')

            if self.is_vot_video:
                self.frames = get_video_frames(self.frames_path)
            else:
                raise NotImplementedError("Not implemented yet")

            logger.debug('Loading first frame')

            self.frame_number = len(self.frames)
            self.current_frame = self.frames[self.current_frame_number - 1]

        else:
            if self.is_vot_video:
                logger.debug('Creating array of frame names')

                self.__frame_files = sorted(glob.glob(os.path.join(self.frames_path, "*.jpg")))
                self.frame_number = len(self.__frame_files)
                logger.debug('%d frames found' % self.frame_number)

                logger.debug('Loading first frame')

                if self.current_frame_number < self.frame_number:
                    self.current_frame = cv2.imread(self.__frame_files[self.current_frame_number - 1])
                else:
                    self.current_frame = None
            else:
                logger.debug('Opening video capturer')

                self.__videocapture = cv2.VideoCapture(self.path)
                try:
                    # check if we are using OpenCV 3
                    if cv2.__version__.startswith("3."):
                        self.frame_number = int(self.__videocapture.get(cv2.CAP_PROP_FRAME_COUNT))

                    # otherwise, we are using OpenCV 2.4
                    else:
                        self.frame_number = int(self.__videocapture.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))

                    logger.debug('%d frames found' % self.frame_number)

                    # otherwise, ignore the frame number
                except:
                    self.frame_number = -1
                    logger.debug('Couldn\'t read the number of frames')

                logger.debug('Loading first frame')

                success, self.current_frame = self.__videocapture.read()
                if not success:
                    logger.error('Video is empty')

                    raise OSError('Video is empty')

    def next_groundtruth(self):
        logger.debug('Loading next groundtruth')

        if self.has_groundtruth:
            self.groundtruth_line = self.groundtruth_file.readline()
            if self.groundtruth_line:
                region = [float(i) for i in self.groundtruth_line.strip().split(",")]
                self.current_groundtruth = region_to_aabb(region)

                self.current_groundtruth_number = self.current_groundtruth_number + 1
                logger.debug('Groundtruth %d loaded' % self.current_groundtruth_number)

            else:
                self.current_groundtruth = None
                logger.debug('Reached end of groundtruth file')
        else:
            self.current_groundtruth = None
            logger.debug('Video doesn\'t have a groundtruth file')

        return self.current_groundtruth

    def next_frame(self):
        logger.debug('Loading next frame')

        if self.is_scheduled:
            if self.current_frame_number < self.frame_number:
                self.current_frame_number = self.current_frame_number + 1
                self.current_frame = self.frames[self.current_frame_number - 1]
                logger.debug('Frame %d loaded' % self.current_frame_number)

            else:
                self.current_frame = None
                logger.debug('Reached end of video')

        else:
            if self.is_vot_video:
                if self.current_frame_number < self.frame_number:
                    self.current_frame_number = self.current_frame_number + 1
                    self.current_frame = cv2.imread(self.__frame_files[self.current_frame_number - 1])
                    logger.debug('Frame %d loaded' % self.current_frame_number)

                else:
                    self.current_frame = None
                    logger.debug('Reached end of video')

            else:
                success, self.current_frame = self.__videocapture.read()
                if success:
                    self.current_frame_number = self.current_frame_number + 1
                    logger.debug('Frame %d loaded' % self.current_frame_number)

                else:
                    self.current_frame = None
                    logger.debug('Reached end of video')

        return self.current_frame

    def close(self):
        logger.debug('Closing video')
        if self.__remove_folder:
            logger.debug('Removing temp frames folder')
            shutil.rmtree(self.frames_path)

        if self.has_groundtruth:
            logger.debug('Closing groundtruth file')
            self.groundtruth_file.close()

        if self.is_scheduled is False and self.is_vot_video is False and self.__videocapture is not None:
            logger.debug('Releasing video capture')
            self.__videocapture.release()

        logger.debug('Video closed')

    def __open_webcam(self):
        self.path = ''
        self.frames_path = ''
        self.name = ('Camera_device_' + str(self.webcam_number)).replace('/', '_')
        self.frame_number = -1
        self.has_groundtruth = False
        self.is_vot_video = False
        self.__remove_folder = False

        logger.debug('Opening video capturer')

        self.__videocapture = cv2.VideoCapture(self.webcam_number)

        logger.debug('Loading first frame')

        success, self.current_frame = self.__videocapture.read()
        if not success:
            logger.debug('Couldn\'t access camera device')

            raise OSError('Couldn\'t access camera device')

        self.current_frame_number = 1


def get_video_info(base_path, video_name, image_folder=False):
    """Retrieves the info needed for tracking a video"""
    if image_folder is True:
        if os.path.isdir(os.path.join(base_path, video_name, 'imgs')):
            video_path = os.path.join(base_path, video_name, 'imgs')

        elif os.path.isdir(os.path.join(base_path, video_name, 'img')):
            video_path = os.path.join(base_path, video_name, 'img')

        elif os.path.isdir(os.path.join(base_path, video_name, 'color')):
            video_path = os.path.join(base_path, video_name, 'color')

        else:
            video_path = os.path.join(base_path, video_name, 'imgs')

    else:
        video_path = os.path.join(base_path, video_name)

    groundtruth = open(os.path.join(base_path, video_name, 'groundtruth.txt'), 'r')
    line = groundtruth.readline()
    region = [float(i) for i in line.strip().split(",")]
    cx, cy, w, h = region_to_aabb(region)
    position = [cy, cx]
    target_size = [h, w]

    frames = get_video_frames(video_path)

    groundtruth.close()

    return frames, np.array(position), np.array(target_size)


def get_video_frames(path):
    logger.debug('Loading video frames')
    frame_list = []
    file_list = glob.glob(os.path.join(path, "*.jpg"))

    i = 0

    file_list.sort()
    for frame_file in file_list:
        i = i + 1
        logger.debug('Loading frame %d of %d' % (i, len(file_list)))
        # We load the image
        frame_list.append(cv2.imread(frame_file).astype(np.uint8))

        logger.debug('%d frames loaded' % i)

    return frame_list
