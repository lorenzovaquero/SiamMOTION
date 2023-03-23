"""display_multi.py: Displays a multiple-target video and it's tracking information frame-by-frame"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import cv2
import numpy as np
from matplotlib.backend_bases import KeyEvent, CloseEvent
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from .display_utils import horizontal_stack, vertical_stack, create_value_image, drawrect, features_to_image

__author__ = "Lorenzo Vaquero Otal"
__credits__ = ["Lorenzo Vaquero Otal"]
__date__ = "2023"
__version__ = "1.0.0"
__maintainer__ = "Lorenzo Vaquero Otal"
__contact__ = "lorenzo.vaquero.otal@usc.es"
__status__ = "Prototype"


COLOR_LIST = [(0, 255, 255), (255, 0, 255), (255, 255, 0), (255, 0, 0), (0, 255, 0), (255, 0, 0), (0, 255, 255),
              (255, 20, 147), (147, 20, 255), (139, 69, 19), (19, 69, 139), (144, 128, 112), (212, 255, 127),
              (0, 140, 255), (220, 245, 245)]

MATPLOTLIB_KEY = -1

def display_mot_benchmark(frame_image, dt_target=None, dt_ids=None, gt_target=None, gt_ids=None,
                          target_reports=None, target_reinitialization=None, ignore_accuracy_after_failure=None,
                          video_name='', frame_number=None, save_folder=None,
                          only_save=False, pixels_loaded_as_opencv=False):
    """It has less features, but appears to be faster than MatPlotLib"""
    return_value = -1

    if pixels_loaded_as_opencv:
        current_frame_image = frame_image.astype(np.uint8)  # In order to not draw (with cv2.rectangle) over the original image
    else:
        current_frame_image = (frame_image * 255.0).astype(np.uint8)[..., ::-1]  # In order to not draw (with cv2.rectangle) over the original image

    if dt_target is not None:
        for i, current_dt_target in enumerate(dt_target):
            current_dt_id = str(int(dt_ids[i]))
            current_color_dt = COLOR_LIST[int(current_dt_id) % len(COLOR_LIST)]

            # Bounding-box
            cv2.rectangle(current_frame_image, tuple(current_dt_target.left_top_point.astype(int)),
                          tuple(current_dt_target.right_bottom_point.astype(int)),
                          current_color_dt, thickness=2)

    if gt_target is not None:
        for i, current_gt_target in enumerate(gt_target):
            current_gt_id = str(int(gt_ids[i]))
            current_color_gt = COLOR_LIST[int(current_gt_id) % len(COLOR_LIST)]

            # Bounding-box
            drawrect(current_frame_image, tuple(current_gt_target.left_top_point.astype(int)),
                     tuple(current_gt_target.right_bottom_point.astype(int)),
                     current_color_gt, thickness=1, style='dotted')

            # Object-center
            cv2.rectangle(current_frame_image, tuple(current_gt_target.center.astype(int)),
                          tuple(current_gt_target.center.astype(int)),
                          current_color_gt, thickness=1)


    values_image = __create_mot_stats_image(target_reports, target_ids=dt_ids,
                                            target_reinitialization=target_reinitialization,
                                            ignore_accuracy_after_failure=ignore_accuracy_after_failure)
    current_frame_image = horizontal_stack(current_frame_image, values_image)


    current_frame_image = cv2.copyMakeBorder(current_frame_image, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=tuple([255, 255, 255]))

    if save_folder is not None:
        cv2.imwrite(os.path.join(save_folder, "%08d.jpg" % frame_number), current_frame_image)

    if not only_save:
        cv2.imshow("Tracking " + video_name, current_frame_image)
        return_value = cv2.waitKey(1)

        if cv2.getWindowProperty("Tracking " + video_name, cv2.WND_PROP_VISIBLE) == 0:  # If the "x" is pressed
            return_value = 27

    return return_value, current_frame_image

def __create_mot_stats_image(target_reports, target_ids, target_reinitialization, ignore_accuracy_after_failure):
    values = []
    label_width = 75
    value_width = 400

    values.append(create_value_image(" ID |", "IoU  |Ign| Status || Fails| Acc", label_width=label_width, value_width=value_width))

    for current_target_id in target_ids:
        current_target_id = int(current_target_id)
        current_color = COLOR_LIST[int(current_target_id) % len(COLOR_LIST)]

        current_target_report = target_reports[current_target_id]
        current_target_iou = current_target_report['iou'][-1]
        current_target_reinit = current_target_report['reinitialization'][-1]
        current_target_ignore = current_target_report['ignore_accuracy'][-1]

        current_target_failures = sum(current_target_report['reinitialization'])
        current_target_acc = [iou for i, iou in enumerate(current_target_report['iou']) if not current_target_report['ignore_accuracy'][i]]
        if len(current_target_acc) > 0:
            current_target_acc = sum(current_target_acc) / len(current_target_acc)
            current_target_acc = "%4.2f" % current_target_acc
        else:
            current_target_acc = " NA "

        current_target_reinit_counter = target_reinitialization[current_target_id]
        if current_target_reinit_counter == 0:
            if current_target_ignore:
                current_target_status = " Starting"
            else:
                current_target_status = " Normal "

        elif current_target_reinit == 1:
            current_target_status = " Failure "

        elif current_target_reinit_counter + 1 > ignore_accuracy_after_failure:
            current_target_status = " Skipping"

        elif current_target_reinit_counter + 1 == ignore_accuracy_after_failure:
            current_target_status = "  Reinit "

        else:
            current_target_status = " Starting"

        values.append(create_value_image("%3d |" % int(current_target_id),
                                         "%4.2f | %s |%s||%4d | %s" % (current_target_iou,
                                                                        "T" if current_target_ignore else "F",
                                                                        current_target_status,
                                                                        current_target_failures,
                                                                        current_target_acc),
                                         txt_color=current_color,
                                         label_width=label_width, value_width=value_width))

    image = values[0]
    for value in values[1::]:
        image = vertical_stack(image, value)

    return image

def cv2_detailed_display_multi(frame_image, target_in_frame=None, searcharea_crop_size=None,
                                 frame_features=None, target_in_frame_features=None,
                                 target_in_frame_features_for_tracking=None,
                                 searcharea_crop_size_in_features=None,
                                 num_scales=1, best_scales=None,
                                 searcharea_image=None, target_in_searcharea=None,
                                 searcharea_features=None, target_in_searcharea_features=None,
                                 exemplar_image=None, exemplar_features=None,
                                 scoremap=None, penalized_scoremap=None, penalization_window=None,
                                 global_penalization_window=None,
                                 target_confidence=None,
                                 target_in_score=None, target_in_score_for_tracking=None,
                                 frame_valid_size_features=None,
                                 frame_valid_size_score=None,
                                 frame_features_valid_size_score=None,
                                 target_ids=None,
                                 searcharea_features_valid_size_score=None,
                                 show_all_scales=False,
                                 video_name='', frame_number=None, save_folder=None,
                                 only_save=False, pixels_loaded_as_opencv=False):
    """It has less features, but appears to be faster than MatPlotLib"""
    return_value = -1

    if best_scales is None and target_in_frame is not None:
        best_scales = [num_scales // 2] * len(target_in_frame)

    if pixels_loaded_as_opencv:
        current_frame_image = frame_image.astype(np.uint8)  # In order to not draw (with cv2.rectangle) over the original image
    else:
        current_frame_image = (frame_image * 255.0).astype(np.uint8)[..., ::-1]  # In order to not draw (with cv2.rectangle) over the original image
        searcharea_image = (searcharea_image * 255.0).astype(np.uint8)[..., ::-1]
        exemplar_image = (exemplar_image * 255.0).astype(np.uint8)[..., ::-1]

    current_frame_image_axis_size = (np.array(current_frame_image.shape[0:2]) - 1) / 2

    if frame_valid_size_features is not None:  # Indica el tamano de las features
        drawrect(current_frame_image,
                 tuple(current_frame_image_axis_size - (frame_valid_size_features - 1) / 2)[::-1],
                 tuple(current_frame_image_axis_size + (frame_valid_size_features - 1) / 2)[::-1],
                 (9, 59, 89), thickness=2, style='dotted')

    if frame_valid_size_score is not None:  # Indica el tamano del score
        drawrect(current_frame_image,
                 tuple(current_frame_image_axis_size - (frame_valid_size_score - 1) / 2)[::-1],
                 tuple(current_frame_image_axis_size + (frame_valid_size_score - 1) / 2)[::-1],
                 (19, 89, 89), thickness=2, style='dotted')


    if target_in_frame is not None:
        for i, current_target_in_frame in enumerate(target_in_frame):
            current_color = __choose_color(i, target_ids=target_ids)
            # current_color = COLOR_LIST[i % len(COLOR_LIST)]

            # Bounding-box
            cv2.rectangle(current_frame_image, tuple(current_target_in_frame.left_top_point.astype(int)),
                          tuple(current_target_in_frame.right_bottom_point.astype(int)),
                          current_color, thickness=2)

            # Object-center
            cv2.rectangle(current_frame_image, tuple(current_target_in_frame.center.astype(int)),
                          tuple(current_target_in_frame.center.astype(int)),
                          current_color, thickness=1)

            # Object searcharea
            if searcharea_crop_size is not None:
                drawrect(current_frame_image, tuple(current_target_in_frame.left_top_point.astype(int) - searcharea_crop_size[i].astype(int)),
                          tuple(current_target_in_frame.right_bottom_point.astype(int) + searcharea_crop_size[i].astype(int)),
                         current_color, thickness=1, style='dotted')


    if frame_features is not None:
        if type(frame_features) in [list, tuple] and isinstance(frame_features[0], np.ndarray):
            frame_features = frame_features[0]

        current_features_axis_size = (np.array(frame_features.shape[1:3]) - 1) / 2
        current_frame_features = features_to_image(frame_features[0])

        if frame_features_valid_size_score is not None:
            drawrect(current_frame_features,
                     tuple(current_features_axis_size - (frame_features_valid_size_score - 1) / 2)[::-1],
                     tuple(current_features_axis_size + (frame_features_valid_size_score - 1) / 2)[::-1],
                     (19, 89, 89), thickness=1, gap=10)

        if target_in_frame_features is not None:
            for i, current_target_in_frame_features in enumerate(target_in_frame_features):
                current_color = __choose_color(i, target_ids=target_ids)
                # current_color = COLOR_LIST[i % len(COLOR_LIST)]

                # Bounding-box
                cv2.rectangle(current_frame_features, tuple(current_target_in_frame_features.left_top_point.astype(int)),
                              tuple(current_target_in_frame_features.right_bottom_point.astype(int)),
                              current_color, thickness=1)

                # Object-center
                cv2.rectangle(current_frame_features, tuple(current_target_in_frame_features.center.astype(int)),
                              tuple(current_target_in_frame_features.center.astype(int)),
                              current_color, thickness=1)

                # Object searcharea
                if searcharea_crop_size_in_features is not None:
                    drawrect(current_frame_features, tuple(current_target_in_frame_features.center.astype(int) - searcharea_crop_size_in_features[i].astype(int)),
                              tuple(current_target_in_frame_features.center.astype(int) + searcharea_crop_size_in_features[i].astype(int)),
                             current_color, thickness=1, style='dotted')

        current_frame_image = vertical_stack(current_frame_image, current_frame_features)

    if global_penalization_window is not None:
        if len(global_penalization_window.shape) > 3:
            for i in range(global_penalization_window.shape[0]):
                current_global_penalization_window = __transform_matrix_range(np.repeat(global_penalization_window[i, :, :, np.newaxis], 3, axis=2), 255, 0).astype(np.uint8)
                current_global_penalization_window = cv2.applyColorMap(current_global_penalization_window, cv2.COLORMAP_JET)

                current_frame_image = vertical_stack(current_frame_image, current_global_penalization_window)

        else:
            current_global_penalization_window = __transform_matrix_range(np.repeat(global_penalization_window[:, :, np.newaxis], 3, axis=2), 255, 0).astype(np.uint8)
            current_global_penalization_window = cv2.applyColorMap(current_global_penalization_window, cv2.COLORMAP_JET)

            current_frame_image = vertical_stack(current_frame_image, current_global_penalization_window)


    crops = np.full([1, 1, 3], 255, np.uint8)
    for i in iter(range(len(target_in_frame if target_in_frame is not None else []))):
        current_color = __choose_color(i, target_ids=target_ids)
        # current_color = COLOR_LIST[i % len(COLOR_LIST)]

        if show_all_scales:
            print_scale_range = range(num_scales)
        else:
            print_scale_range = [best_scales[i]]

        for j in print_scale_range:
            current_target_scale = i * num_scales + j

            if searcharea_image is not None:
                current_searcharea_image = searcharea_image[current_target_scale].astype(np.uint8)

                if target_in_searcharea is not None:
                    # Bounding-box
                    cv2.rectangle(current_searcharea_image,
                                  tuple(target_in_searcharea[i].left_top_point.astype(int)),
                                  tuple(target_in_searcharea[i].right_bottom_point.astype(int)),
                                  current_color, thickness=1)

                    # Object-center
                    cv2.rectangle(current_searcharea_image, tuple(target_in_searcharea[i].center.astype(int)),
                                  tuple(target_in_searcharea[i].center.astype(int)),
                                  current_color, thickness=1)

            else:
                current_searcharea_image = np.full([1, 1, 3], 255, np.uint8)

            if searcharea_features is not None:
                if len(searcharea_features.shape) == 4 and searcharea_features.shape[0] == 1:
                    current_searcharea_features = features_to_image(searcharea_features[0])  
                else:
                    current_searcharea_features = features_to_image(searcharea_features[current_target_scale])

                current_searcharea_features_axis_size = (np.array(current_searcharea_features.shape[0:2]) - 1) / 2

                if searcharea_features_valid_size_score is not None:
                    drawrect(current_searcharea_features,
                             tuple(current_searcharea_features_axis_size - (searcharea_features_valid_size_score - 1) / 2)[::-1],
                             tuple(current_searcharea_features_axis_size + (searcharea_features_valid_size_score - 1) / 2)[::-1],
                             (19, 89, 89), thickness=1, gap=5)

                if target_in_searcharea_features is not None:
                    # Bounding-box
                    cv2.rectangle(current_searcharea_features,
                                  tuple(target_in_searcharea_features[i].left_top_point.astype(int)),
                                  tuple(target_in_searcharea_features[i].right_bottom_point.astype(int)),
                                  current_color, thickness=1)

                    # Object-center
                    cv2.rectangle(current_searcharea_features, tuple(target_in_searcharea_features[i].center.astype(int)),
                                  tuple(target_in_searcharea_features[i].center.astype(int)),
                                  current_color, thickness=1)

                current_searcharea_image = horizontal_stack(current_searcharea_image, current_searcharea_features)


            if scoremap is not None:
                current_heatmap = __transform_matrix_range(np.repeat(scoremap[i, :, :, np.newaxis], 3, axis=2), 255, 0).astype(np.uint8)
                current_heatmap = cv2.applyColorMap(current_heatmap, cv2.COLORMAP_JET)


                if target_in_score is not None:
                    # Bounding-box
                    cv2.rectangle(current_heatmap,
                                  tuple(target_in_score[i].left_top_point.astype(int)),
                                  tuple(target_in_score[i].right_bottom_point.astype(int)),
                                  current_color, thickness=1)

                    # Object-center
                    cv2.rectangle(current_heatmap, tuple(target_in_score[i].center.astype(int)),
                                  tuple(target_in_score[i].center.astype(int)),
                                  current_color, thickness=1)

                current_searcharea_image = horizontal_stack(current_searcharea_image, current_heatmap)


            if penalized_scoremap is not None:

                current_penalized_heatmap = __transform_matrix_range(np.repeat(penalized_scoremap[i, :, :, np.newaxis], 3, axis=2), 255, 0).astype(np.uint8)
                current_penalized_heatmap = cv2.applyColorMap(current_penalized_heatmap, cv2.COLORMAP_JET)

                if target_in_score is not None:
                    # Bounding-box
                    cv2.rectangle(current_penalized_heatmap,
                                  tuple(target_in_score[i].left_top_point.astype(int)),
                                  tuple(target_in_score[i].right_bottom_point.astype(int)),
                                  current_color, thickness=1)

                    # Object-center
                    cv2.rectangle(current_penalized_heatmap, tuple(target_in_score[i].center.astype(int)),
                                  tuple(target_in_score[i].center.astype(int)),
                                  current_color, thickness=1)

                current_searcharea_image = horizontal_stack(current_searcharea_image, current_penalized_heatmap)


            if penalization_window is not None:
                if len(penalization_window.shape) < 3:
                    current_penalization_window = __transform_matrix_range(np.repeat(penalization_window[:, :, np.newaxis], 3, axis=2), 255, 0).astype(np.uint8)
                else:
                    current_penalization_window = __transform_matrix_range(np.repeat(penalization_window[i, :, :, np.newaxis], 3, axis=2), 255, 0).astype(np.uint8)

                current_penalization_window = cv2.applyColorMap(current_penalization_window, cv2.COLORMAP_JET)

                if target_in_score_for_tracking is not None:
                    # Bounding-box
                    cv2.rectangle(current_penalization_window,
                                  tuple(target_in_score_for_tracking[i].left_top_point.astype(int)),
                                  tuple(target_in_score_for_tracking[i].right_bottom_point.astype(int)),
                                  current_color, thickness=1)

                    # Object-center
                    cv2.rectangle(current_penalization_window, tuple(target_in_score_for_tracking[i].center.astype(int)),
                                  tuple(target_in_score_for_tracking[i].center.astype(int)),
                                  current_color, thickness=1)

                current_searcharea_image = horizontal_stack(current_searcharea_image, current_penalization_window)


            if exemplar_image is not None:
                current_exemplar_image = exemplar_image[i].astype(np.uint8)

            else:
                current_exemplar_image = np.full([1, 1, 3], 255, np.uint8)

            if exemplar_features is not None:
                current_exemplar_features = features_to_image(exemplar_features[i])

                current_exemplar_image = horizontal_stack(current_exemplar_features, current_exemplar_image)

            if target_confidence is not None:
                confidence_image = create_value_image("", " {:.1%}".format(target_confidence[i]),
                                                      label_width=1, value_width=40, value_scale=0.5, height=17)

                current_exemplar_image = horizontal_stack(current_exemplar_image, confidence_image)


            current_searcharea_image = horizontal_stack(current_searcharea_image, current_exemplar_image)

            current_searcharea_image = cv2.copyMakeBorder(current_searcharea_image, 1, 1, 1, 1, cv2.BORDER_CONSTANT,
                                                          value=tuple([255, 255, 255]))
            if show_all_scales and j == best_scales[i]:
                cv2.rectangle(current_searcharea_image, tuple([0, 0]),
                              tuple([current_searcharea_image.shape[1]-1, current_searcharea_image.shape[0]-1]),
                              (0, 0, 255), thickness=1)

            if show_all_scales and j == 0:
                current_searcharea_image = cv2.copyMakeBorder(current_searcharea_image, 1, 0, 0, 0, cv2.BORDER_CONSTANT,
                                                              value=np.array([0, 0, 0]))
            if show_all_scales and j == (len(print_scale_range) - 1):
                current_searcharea_image = cv2.copyMakeBorder(current_searcharea_image, 0, 1, 0, 0, cv2.BORDER_CONSTANT,
                                                              value=np.array([0, 0, 0]))


            crops = vertical_stack(crops, current_searcharea_image)


    current_frame_image = horizontal_stack(current_frame_image, crops)

    current_frame_image = cv2.copyMakeBorder(current_frame_image, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=tuple([255, 255, 255]))

    if save_folder is not None:
        cv2.imwrite(os.path.join(save_folder, "%08d.jpg" % frame_number), current_frame_image)

    if not only_save:
        cv2.imshow("Tracking " + video_name, current_frame_image)
        return_value = cv2.waitKey(1)

        if cv2.getWindowProperty("Tracking " + video_name, cv2.WND_PROP_VISIBLE) == 0:  # If the "x" is pressed
            return_value = 27

    return return_value, current_frame_image


def __choose_color(i, target_ids=None):
    num = 0
    if target_ids is not None and len(target_ids) > 0:
        num = int(float(target_ids[i]))
    else:
        num = i

    current_color = COLOR_LIST[num % len(COLOR_LIST)]
    return current_color



def __transform_matrix_range(matrix, new_max, new_min):
    matrix += -(np.min(matrix))

    # To avoid nan
    if np.max(matrix) == 0 and np.min(matrix) == 0:
        matrix = np.full_like(matrix, new_min)
    elif np.max(matrix) == 0:
        matrix /= np.nextafter(0, 1) / (new_max - new_min)
    else:
        matrix /= np.max(matrix) / (new_max - new_min)

    matrix += new_min
    return matrix


def __register_key(event):
    """When Escape is pressed or the figure is closed"""
    global MATPLOTLIB_KEY

    if type(event) == KeyEvent:
        if event.key == 'escape':
            MATPLOTLIB_KEY = 27
        else:
            MATPLOTLIB_KEY = ord(event.key)

    elif type(event) == CloseEvent:
        MATPLOTLIB_KEY = 27

def pause_display(matplotlib=False):
    """Pauses an ongoing display"""

    if not matplotlib:
        cv2.waitKey(0)

    else:
        while not plt.waitforbuttonpress():
            pass
