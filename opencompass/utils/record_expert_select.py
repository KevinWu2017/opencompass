import json
from pathlib import Path
import os
import hashlib
import uuid
import torch
import pickle

output_path = ''
output_file = ''

# 用于存储中间结果的临时变量
result_per_iter = []
result_per_batch = []
# 用于存储最终结果的变量，其结构为[num_batch * [num_iter * [num_layer * tensor]]]
# 其中num_iter为每个batch的迭代次数(= 处理prompt的次数 + 递推次数)
overall_result = []

# 用于存储具体的len的临时变量
temp_len = []
# 用于存储每个batch的初始prompt处理的seq_len，其结构为[num_batch * [处理prompt的次数 * 具体的len]]
seq_len = []

begin_recording = False

processing_prompt = True
expected_batch_size = 0


def set_output_file_path(filepath, filename):
    global begin_recording
    global output_path
    global output_file

    begin_recording = True

    output_path = filepath
    output_file = filename

    print(output_path, output_file)

    os.makedirs(output_path, exist_ok=True)


def start_one_batch(batch_size):
    global expected_batch_size
    global processing_prompt
    expected_batch_size = batch_size
    processing_prompt = True


def record_seq_len(slen, device_index):
    global processing_prompt
    global temp_len
    global expected_batch_size
    global seq_len
    if begin_recording:
        if begin_recording and processing_prompt and device_index == 0:
            temp_len.append(slen)
            expected_batch_size -= len(slen)
            if expected_batch_size == 0:
                seq_len.append(temp_len.copy())
                temp_len.clear()
                processing_prompt = False


def record_one_layer(tensor):
    if begin_recording:
        # TODO: avoid recording max seq len
        global result_per_iter
        result_per_iter.append(tensor.to("cpu"))


def store_one_iter():
    if begin_recording:
        global result_per_batch
        global result_per_iter
        result_per_batch.append(result_per_iter.copy())
        result_per_iter.clear()


def store_one_batch():
    if begin_recording:
        global result_per_batch
        global overall_result
        overall_result.append(result_per_batch.copy())
        result_per_batch.clear()


def dump_result():
    if begin_recording:
        global overall_result
        global output_path
        global output_file
        global seq_len

        f = open(Path(output_path) / output_file, 'wb')
        pickle.dump({"gate": overall_result, "prompt_len": seq_len}, f, -1)
        f.close()
        # dump_to_results_dict(overall_result, file_name)
        overall_result.clear()
