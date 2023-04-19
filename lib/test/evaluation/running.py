import time
from datetime import timedelta

import numpy as np
import multiprocessing
import os
import sys
from itertools import product
from collections import OrderedDict
from lib.test.evaluation import Sequence, Tracker
import torch


def det_save_tracker_output(seq: Sequence, tracker: Tracker, output: dict, run_id=None):
    """Saves the output of the tracker."""

    if not os.path.exists(tracker.results_dir):
        os.makedirs(tracker.results_dir)

    base_results_path = os.path.join(tracker.results_dir, seq.name)
    if not os.path.exists(base_results_path):
        os.makedirs(base_results_path)

    # segmentation_path = os.path.join(tracker.segmentation_dir, seq.name)

    if type(seq.frames[0]) is dict:
        frame_names = [os.path.splitext(os.path.basename(f['color']))[0] for f in seq.frames]
    else:
        frame_names = [os.path.splitext(os.path.basename(f[0]))[0] for f in seq.frames]

    def save_bb(file, data):
        tracked_bb = np.array(data).astype(int)
        # np.savetxt(file, tracked_bb, delimiter='\t', fmt='%d')
        np.savetxt(file, tracked_bb, delimiter=',', fmt='%d')

    def save_time(file, data):
        exec_times = np.array(data).astype(float)
        np.savetxt(file, exec_times, delimiter='\t', fmt='%f')

    def save_confidence(file, data):
        confidence = np.array(data).astype(float)
        np.savetxt(file, confidence, delimiter='\t', fmt='%f')

    def _convert_dict(input_dict):
        data_dict = {}
        for elem in input_dict:
            for k, v in elem.items():
                if k in data_dict.keys():
                    data_dict[k].append(v)
                else:
                    data_dict[k] = [v, ]
        return data_dict

    for key, data in output.items():
        # If data is empty
        if not data:
            continue

        if key == 'target_bbox':
            if isinstance(data[0], (dict, OrderedDict)):
                data_dict = _convert_dict(data)

                for obj_id, d in data_dict.items():
                    # bbox_file = '{}_{}.txt'.format(base_results_path, obj_id)
                    if run_id:
                        bbox_file = '{}/{}_{}_{:03}.txt'.format(base_results_path, seq.name, obj_id,run_id)
                    else:
                        bbox_file = '{}/{}_{}_001.txt'.format(base_results_path, seq.name, obj_id)
                    save_bb(bbox_file, d)
            else:
                # Single-object mode
                # bbox_file = '{}.txt'.format(base_results_path)
                if run_id:
                    bbox_file = '{}/{}_{:03}.txt'.format(base_results_path, seq.name, run_id)
                else:
                    bbox_file = '{}/{}_001.txt'.format(base_results_path, seq.name)
                save_bb(bbox_file, data)

        elif key == 'time':
            if isinstance(data[0], dict):
                data_dict = _convert_dict(data)

                for obj_id, d in data_dict.items():
                    # timings_file = '{}_{}_time.txt'.format(base_results_path, obj_id)
                    if run_id:
                        timings_file = '{}/{}_{}_{:03}_time.value'.format(base_results_path, seq.name, obj_id, run_id)
                    else:
                        timings_file = '{}/{}_{}_001_time.value'.format(base_results_path, seq.name, obj_id)
                    save_time(timings_file, d)
            else:
                # timings_file = '{}_time.txt'.format(base_results_path)
                if run_id:
                    timings_file = '{}/{}_{:03}_time.value'.format(base_results_path, seq.name, run_id)
                else:
                    timings_file = '{}/{}_001_time.value'.format(base_results_path, seq.name)
                save_time(timings_file, data)

        elif key == 'confidence':
            if isinstance(data[0], dict):
                data_dict = _convert_dict(data)

                for obj_id, d in data_dict.items():
                    # confidence_file = '{}_{}_confidence.txt'.format(base_results_path, obj_id)
                    if run_id:
                        confidence_file = '{}/{}_{}_{:03}_confidence.value'.format(base_results_path, seq.name, obj_id, run_id)
                    else:
                        confidence_file = '{}/{}_{}_001_confidence.value'.format(base_results_path, seq.name, obj_id)
                    save_time(confidence_file, d)
            else:
                # confidence_file = '{}_confidence.txt'.format(base_results_path)
                if run_id:
                    confidence_file = '{}/{}_{:03}_confidence.value'.format(base_results_path, seq.name, run_id)
                else:
                    confidence_file = '{}/{}_001_confidence.value'.format(base_results_path, seq.name)
                save_time(confidence_file, data)


def _save_tracker_output(seq: Sequence, tracker: Tracker, output: dict):
    """Saves the output of the tracker."""

    if not os.path.exists(tracker.results_dir):
        print("create tracking result dir:", tracker.results_dir)
        os.makedirs(tracker.results_dir)
    if seq.dataset in ['trackingnet', 'got10k']:
        if not os.path.exists(os.path.join(tracker.results_dir, seq.dataset)):
            os.makedirs(os.path.join(tracker.results_dir, seq.dataset))
    '''2021.1.5 create new folder for these two datasets'''
    if seq.dataset in ['trackingnet', 'got10k']:
        base_results_path = os.path.join(tracker.results_dir, seq.dataset, seq.name)
    else:
        base_results_path = os.path.join(tracker.results_dir, seq.name)

    def save_bb(file, data):
        tracked_bb = np.array(data).astype(int)
        np.savetxt(file, tracked_bb, delimiter='\t', fmt='%d')

    def save_time(file, data):
        exec_times = np.array(data).astype(float)
        np.savetxt(file, exec_times, delimiter='\t', fmt='%f')

    def save_score(file, data):
        scores = np.array(data).astype(float)
        np.savetxt(file, scores, delimiter='\t', fmt='%.2f')

    def _convert_dict(input_dict):
        data_dict = {}
        for elem in input_dict:
            for k, v in elem.items():
                if k in data_dict.keys():
                    data_dict[k].append(v)
                else:
                    data_dict[k] = [v, ]
        return data_dict

    for key, data in output.items():
        # If data is empty
        if not data:
            continue

        if key == 'target_bbox':
            if isinstance(data[0], (dict, OrderedDict)):
                data_dict = _convert_dict(data)

                for obj_id, d in data_dict.items():
                    bbox_file = '{}_{}.txt'.format(base_results_path, obj_id)
                    save_bb(bbox_file, d)
            else:
                # Single-object mode
                bbox_file = '{}.txt'.format(base_results_path)
                save_bb(bbox_file, data)

        if key == 'all_boxes':
            if isinstance(data[0], (dict, OrderedDict)):
                data_dict = _convert_dict(data)

                for obj_id, d in data_dict.items():
                    bbox_file = '{}_{}_all_boxes.txt'.format(base_results_path, obj_id)
                    save_bb(bbox_file, d)
            else:
                # Single-object mode
                bbox_file = '{}_all_boxes.txt'.format(base_results_path)
                save_bb(bbox_file, data)

        if key == 'all_scores':
            if isinstance(data[0], (dict, OrderedDict)):
                data_dict = _convert_dict(data)

                for obj_id, d in data_dict.items():
                    bbox_file = '{}_{}_all_scores.txt'.format(base_results_path, obj_id)
                    save_score(bbox_file, d)
            else:
                # Single-object mode
                print("saving scores...")
                bbox_file = '{}_all_scores.txt'.format(base_results_path)
                save_score(bbox_file, data)

        elif key == 'time':
            if isinstance(data[0], dict):
                data_dict = _convert_dict(data)

                for obj_id, d in data_dict.items():
                    timings_file = '{}_{}_time.txt'.format(base_results_path, obj_id)
                    save_time(timings_file, d)
            else:
                timings_file = '{}_time.txt'.format(base_results_path)
                save_time(timings_file, data)


def run_sequence(seq: Sequence, tracker: Tracker, debug=False, num_gpu=8):
    """Runs a tracker on a sequence."""
    '''2021.1.2 Add multiple gpu support'''
    try:
        worker_name = multiprocessing.current_process().name
        worker_id = int(worker_name[worker_name.find('-') + 1:]) - 1
        gpu_id = worker_id % num_gpu
        torch.cuda.set_device(gpu_id)
    except:
        pass

    def det_results_exist():
        if seq.object_ids is None:
            bbox_file = '{}/{}/{}_001.txt'.format(tracker.results_dir, seq.name, seq.name)
            return os.path.isfile(bbox_file)
        else:
            bbox_files = ['{}/{}/{}_{}_001.txt'.format(tracker.results_dir, seq.name, seq.name, obj_id) for obj_id in seq.object_ids]
            missing = [not os.path.isfile(f) for f in bbox_files]
            return sum(missing) == 0

    def _results_exist():
        if seq.object_ids is None:
            if seq.dataset in ['trackingnet', 'got10k']:
                base_results_path = os.path.join(tracker.results_dir, seq.dataset, seq.name)
                bbox_file = '{}.txt'.format(base_results_path)
            else:
                bbox_file = '{}/{}.txt'.format(tracker.results_dir, seq.name)
            return os.path.isfile(bbox_file)
        else:
            bbox_files = ['{}/{}_{}.txt'.format(tracker.results_dir, seq.name, obj_id) for obj_id in seq.object_ids]
            missing = [not os.path.isfile(f) for f in bbox_files]
            return sum(missing) == 0

    if det_results_exist() and not debug:
        print('FPS: {}'.format(-1))
        return

    print('Tracker: {} {} {} ,  Sequence: {}'.format(tracker.name, tracker.parameter_name, tracker.run_id, seq.name))

    if debug:
        output = tracker.run_sequence(seq, debug=debug)
    else:
        # try:
        output = tracker.run_sequence(seq, debug=debug)
        # except Exception as e:
        #     print(e)
        #     return

    sys.stdout.flush()

    if isinstance(output['time'][0], (dict, OrderedDict)):
        exec_time = sum([sum(times.values()) for times in output['time']])
        num_frames = len(output['time'])
    else:
        exec_time = sum(output['time'])
        num_frames = len(output['time'])

    print('FPS: {}'.format(num_frames / exec_time))

    if not debug:
        det_save_tracker_output(seq, tracker, output)


def run_dataset(dataset, trackers, debug=False, threads=0, num_gpus=8):
    """Runs a list of trackers on a dataset.
    args:
        dataset: List of Sequence instances, forming a dataset.
        trackers: List of Tracker instances.
        debug: Debug level.
        threads: Number of threads to use (default 0).
    """
    multiprocessing.set_start_method('spawn', force=True)

    print('Evaluating {:4d} trackers on {:5d} sequences'.format(len(trackers), len(dataset)))
    dataset_start_time = time.time()

    multiprocessing.set_start_method('spawn', force=True)

    if threads == 0:
        mode = 'sequential'
    else:
        mode = 'parallel'

    if mode == 'sequential':
        for seq in dataset:
            for tracker_info in trackers:
                run_sequence(seq, tracker_info, debug=debug)
    elif mode == 'parallel':
        param_list = [(seq, tracker_info, debug, num_gpus) for seq, tracker_info in product(dataset, trackers)]
        with multiprocessing.Pool(processes=threads) as pool:
            pool.starmap(run_sequence, param_list)
    print('Done, total time: {}'.format(str(timedelta(seconds=(time.time() - dataset_start_time)))))
