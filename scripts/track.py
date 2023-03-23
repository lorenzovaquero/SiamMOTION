#!/usr/bin/env python

"""track.py: Tracks an object during a video, given its first-frame location and size"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import cv2
import shutil
import tempfile
import argparse
import numpy as np
import glob
import json
from datetime import datetime
import logging

logger = logging.getLogger()

import tensorflow as tf

CURRENT_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(CURRENT_DIR, '..'))

from parameters.Parameters import Parameters
from tracking.Video import Video
from display.select_region import select_region, select_region_live
from misc.region_utils import region_to_aabb
from display.display_NEW import cv2_detailed_display_multi, pause_display
from misc.FramesPerSecond import FramesPerSecond
import inference.TrackerFactory as TrackerFactory
from logger.log_utils import setup_log, LOG_FORMAT_PRINT
from tracking.AxisAlignedBB import AxisAlignedBB
from misc.url_utils import is_url
from misc.choose_gpu import choose_gpu

__author__ = "Lorenzo Vaquero Otal"
__credits__ = ["Lorenzo Vaquero Otal"]
__date__ = "2023"
__version__ = "1.0.0"
__maintainer__ = "Lorenzo Vaquero Otal"
__contact__ = "lorenzo.vaquero.otal@usc.es"
__status__ = "Prototype"


def main(input_video, parameter_file=None, output_file=None, output_folder=None, output_video=None,
         is_detailed=False, no_logs=False, is_scheduled=False, is_verbose=False, no_display=False,
         webcam_number=None, is_frame_by_frame=False, use_dataset=False):
    #####################################################################
    # INITIAL VARIABLES SETUP
    #####################################################################
    log_level = 'debug' if is_verbose else 'info'
    setup_log(logger, level=log_level, format_style=LOG_FORMAT_PRINT)

    logger.info('Running tracking')

    parameters = Parameters(parameter_file)
    gpu_id = choose_gpu(gpu_id=parameters.gpuId)
    similarity_op = parameters.Tracker.similarity.upper()  # El similarity operation estara definido en los parametros, no en los argumentos

    if (input_video == '' or input_video is None) and webcam_number is None:
        logger.error('Missing input video.')
        sys.exit(2)

    if input_video != '' and input_video is not None:
        if is_url(input_video):
            absolute_input_video = input_video
        else:
            absolute_input_video = os.path.abspath(input_video)
    else:
        absolute_input_video = ''

    # We open the video
    video = Video(absolute_input_video, is_scheduled, webcam_number)

    # We get the first bounding-box (or make the user select it)
    if video.has_groundtruth:
        target = video.current_groundtruth
    else:
        logger.debug('Making the user select the target to track')
        logger.info('Select the target to track')

        if video.webcam_number is not None:
            l, t, w, h = select_region_live(video)
            video.current_frame_number = 0  # In order to ignore the frames elapsed during target selection
        else:
            l, t, w, h = select_region(cv2.cvtColor(video.current_frame, cv2.COLOR_BGR2RGB))  # Due to the file being loaded with cv2, we have to shuffle the color channels from BGR to RGB

        if l == 0 and t == 0 and w == 0 and h == 0:
            logger.error('Missing groundtruth.')
            sys.exit(2)

        target = region_to_aabb((l, t, w, h))

    logger.debug('Target selected')

    # We set the variables and create the necessary files for a correct output
    if output_file == '' or output_file is None:
        absolute_output_file = None
        output_file_writer = None
    else:
        logger.debug('Creating output file')

        absolute_output_file = os.path.abspath(output_file)
        output_file_writer = open(output_file, "w")

        logger.debug('Writing groundtruth to the output file')

        output_file_writer.write("%d,%d,%d,%d" % (target.left, target.top, target.width, target.height))

    if output_folder == '' or output_folder is None:
        absolute_output_folder = None
    else:
        logger.debug('Creating output folder')

        absolute_output_folder = os.path.abspath(output_folder)
        try:
            os.makedirs(absolute_output_folder)
        except OSError:
            logger.warning('Output folder already exists')

    if output_video == '' or output_video is None:
        absolute_output_video = None
        absolute_output_video_frames_folder = None
    else:
        logger.debug('Creating temp folder for output video')
        absolute_output_video = os.path.abspath(output_video)
        absolute_output_video_frames_folder = tempfile.mkdtemp()

    actual_datetime = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    actual_datetime = actual_datetime + '_{}'.format(similarity_op)
    if not no_logs:
        logger.debug('Creating tracking logs folder')
        tracking_logs_folder = os.path.join(os.path.abspath(parameters.Tracker.trackingLogsFolder), video.name + '-' + actual_datetime)
        try:
            os.makedirs(tracking_logs_folder)
        except OSError:
            logger.warning('Logs folder already exists')
    else:
        tracking_logs_folder = None

    with tf.device('/device:GPU:%d' % gpu_id):
        #####################################################################
        # MULTI-TARGET TRACKING
        #####################################################################
        print("Initial target: {}".format(target.tensor))

        tracker_class = TrackerFactory.get_tracker(flavour=parameters.Tracker.similarity)
        tracker = tracker_class(parameters=parameters.Tracker, loader_as_feed_dict=not use_dataset)

        if similarity_op == 'FC':
            display_num_scales = tracker.parameters.FCInference.numScales
        else:
            display_num_scales = 1

        if use_dataset:
            frame_extensions = ['jpg', 'png', 'JPG', 'PNG']
            for frame_extension in frame_extensions:
                frame_path_list = glob.glob(os.path.join(absolute_input_video, "*." + frame_extension))
                frame_path_list = list(sorted(frame_path_list))
                if len(frame_path_list) > 0:
                    break

            if len(frame_path_list) == 0:
                raise OSError('Path "{}" doesn\'t contain frame files.'.format(absolute_input_video))

        else:
            frame_path_list = None

        tracker.build(logs_folder=tracking_logs_folder, frame_path_list=frame_path_list)

        fps = FramesPerSecond()

        if use_dataset:
            current_frame = None
        else:
            current_frame = video.current_frame

        tracker.add_target(current_frame, target.tensor)

        if use_dataset:
            video.next_frame()

        fps.start()

        #####################################################################
        # MAIN TRACKING LOOP
        #####################################################################

        while video.current_frame is not None:
            try:
                targets, target_confidences = tracker.track_next_frame(video.current_frame)
                if targets is None:  # Si no hay targets que seguir
                    targets = []
                    target_confidences = []

            except tf.errors.OutOfRangeError as e:  # Thrown at the end of the dataset iterator.
                logger.info('Reached end of dataset')
                break

            if is_detailed:
                target_string = 'Targets (frame {}):'.format(video.current_frame_number)
                for i, t in enumerate(targets):
                    target_string += '\n  {} - {} ({})'.format(int(tracker.current_target_identifier[i]), t, target_confidences[i])
                print(target_string)

            if output_file != '' and output_file is not None:
                logger.debug('Writing groundtruth to the output file')
                raise NotImplementedError

            current_fps = fps.tick()

            if not no_display or (output_video != '' and output_video is not None):
                logger.debug('Tracking at %f fps' % current_fps)

                if is_detailed:
                    return_value, _ = cv2_detailed_display_multi(
                        frame_image=tracker.current_frame,
                        target_in_frame=[AxisAlignedBB(t[1], t[0], t[3], t[2]) for t in tracker.current_target_in_preprocessed_frame],
                        searcharea_crop_size=tracker.current_searcharea_crop_target_in_frame[:, 2:4],
                        frame_features=tracker.current_frame_features,
                        target_in_frame_features=None,  # [AxisAlignedBB(t[1], t[0], t[3], t[2]) for t in tracker.current_target_in_frame_features],
                        searcharea_crop_size_in_features=None,  # tracker.current_searcharea_crop_target_in_frame_features[:, 2:4],
                        num_scales=display_num_scales,
                        best_scales=None,
                        searcharea_image=tracker.current_searcharea_image,
                        target_in_searcharea=[AxisAlignedBB(t[1], t[0], t[3], t[2]) for t in tracker.current_target_in_searcharea_image],
                        searcharea_features=tracker.current_searcharea_features,
                        target_in_searcharea_features=None,  # [AxisAlignedBB(t[1], t[0], t[3], t[2]) for t in tracker.current_target_in_searcharea_features],
                        exemplar_image=tracker.exemplar_image,
                        exemplar_features=tracker.exemplar_features,
                        scoremap=np.max(tracker.current_scoremap, axis=-1),
                        penalized_scoremap=np.max(tracker.current_penalized_scoremap, axis=-1),
                        target_confidence=target_confidences,
                        penalization_window=tracker.penalization_window,
                        global_penalization_window=tracker.global_penalization_window,
                        target_in_score=None,
                        target_in_score_for_tracking=None,
                        frame_valid_size_features=tracker.current_frame_size_represented_inside_features,
                        frame_valid_size_score=None,
                        frame_features_valid_size_score=None,
                        searcharea_features_valid_size_score=tracker.current_searcharea_features_size_represented_inside_score,
                        target_ids=tracker.current_target_identifier,
                        show_all_scales=False,

                        video_name=video.name, frame_number=video.current_frame_number,
                        save_folder=absolute_output_video_frames_folder, only_save=no_display,
                        pixels_loaded_as_opencv=tracker.parameters.loadAsOpenCV)

                else:
                    return_value, _ = cv2_detailed_display_multi(
                        frame_image=tracker.current_frame,
                        target_in_frame=[AxisAlignedBB(t[1], t[0], t[3], t[2]) for t in targets],
                        target_ids=tracker.current_target_identifier,

                        video_name=video.name, frame_number=video.current_frame_number,
                        save_folder=absolute_output_video_frames_folder, only_save=no_display,
                        pixels_loaded_as_opencv=tracker.parameters.loadAsOpenCV)

                if return_value == 27:  # Escape key
                    break

                elif return_value == ord(' '):  # Space bar
                    pause_display(matplotlib=False)

                elif return_value == ord('+'):  # Plus key
                    if video.webcam_number >= 0:
                        l, t, w, h = select_region_live(video)

                    else:
                        l, t, w, h = select_region(cv2.cvtColor(video.current_frame, cv2.COLOR_BGR2RGB))

                    if not (l == 0 and t == 0 and w == 0 and h == 0):
                        new_target = region_to_aabb((l, t, w, h)).tensor
                        print("Adding target (frame {}): {}".format(video.current_frame_number, new_target))
                        tracker.add_target(video.current_frame, new_target)

                elif return_value == ord('-'):  # Minus key
                    print("Removing target (frame {}): {}".format(video.current_frame_number, tracker.current_target_identifier[0]))
                    tracker.remove_target()

                elif return_value == ord('d'):  # d key (from "duplicate")
                    for target in targets:
                        print("Adding target (frame {}): {}".format(video.current_frame_number, target))
                        tracker.add_target(video.current_frame, target)

                elif is_frame_by_frame:
                    pause_display(matplotlib=False)

            if not use_dataset:
                video.next_frame()

        average_fps = fps.stop()
        fastest_fps = fps.fastest_tick
        slowest_fps = fps.slowest_tick
        tracked_frames = fps.total_ticks
        tracker.close()

    #####################################################################
    # CLOSING
    #####################################################################
    video.close()

    if absolute_output_file is not None:
        output_file_writer.close()

    logger.info('%d video frames tracked at an average of %f fps (%f fps fastest, %f fps slowest)' % (
        tracked_frames, average_fps, fastest_fps, slowest_fps))

    return


def is_int(value):
    ivalue = int(value)
    if ivalue < 0:
        raise argparse.ArgumentTypeError("%s is an invalid positive int value" % value)
    return ivalue


def to_int(value):
    try:
        return int(value)
    except ValueError:
        return value


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Tracks a target inside a video and displays the results')
    parser.add_argument('-p', metavar='file', help='JSON file with tracking parameters')
    parser.add_argument('-o', metavar='file', help='creates a groundtruth text file with the tracking results')
    parser.add_argument('-f', metavar='folder', help='saves the video frames containing the bounding box to the folder')
    parser.add_argument('-v', metavar='file', help='creates a mp4 video containing the bounding box of the tracking result')
    parser.add_argument('--detailed', action='store_true', help='shows detailed video information during tracking')
    parser.add_argument('--verbose', action='store_true', help='prints detailed information about the program execution')
    parser.add_argument('--nodisplay', action='store_true', help='doesn\'t show the video during tracking')
    parser.add_argument('--nologs', action='store_true', help='doesn\'t store execution logs')
    parser.add_argument('--frame-by-frame', action='store_true', help='pauses the display after each frame')

    group_preload = parser.add_mutually_exclusive_group(required=False)
    group_preload.add_argument('--scheduled', action='store_true', help='loads in memory the entire video, before starting the tracking')
    group_preload.add_argument('--dataset', action='store_true', help='loads the video using tf.data.Dataset')

    group_video = parser.add_mutually_exclusive_group(required=True)
    group_video.add_argument('video', help='video to analyze (in mp4, avi, VOT\'s format, OTB\'s format or an IP camera stream)', nargs='?')
    group_video.add_argument('-w', metavar='deviceNumber', help='loads the video in real time from the deviceNumber camera', type=to_int)

    args = parser.parse_args()

    if args.w is not None:
        if args.w >= 0 and args.scheduled:
            print('error: argument --scheduled: not allowed with argument -w')
            sys.exit(1)

    main(input_video=args.video, parameter_file=args.p, output_file=args.o, output_folder=args.f,
         output_video=args.v, is_detailed=args.detailed, no_logs=args.nologs,
         is_scheduled=args.scheduled, is_verbose=args.verbose, no_display=args.nodisplay, webcam_number=args.w,
         is_frame_by_frame=args.frame_by_frame, use_dataset=args.dataset)
