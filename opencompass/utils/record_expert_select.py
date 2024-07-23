import json
from pathlib import Path
import os
import hashlib
import uuid
import torch

output_path = ''
output_file = ''

batch_dict = {}
result_list = []

begin_recording = False

def dump_to_results_dict(results_dict, filename):
    with open(filename, 'w', encoding='utf-8') as json_file:
        json.dump(results_dict, json_file, indent=4, ensure_ascii=False)

def set_output_file_path(filepath, filename):
    global begin_recording
    global output_path
    global output_file
    begin_recording = True
    output_path = filepath
    output_file = filename
    print("Recording expert select results to ", output_path, output_file)
    os.makedirs(output_path, exist_ok=True)

def record_expert_select_result(layer_idx, tensor):
    if begin_recording:
        global batch_dict

        random_hash = hashlib.sha256(uuid.uuid4().bytes).hexdigest()
        filename = f"{random_hash}.pt"
        save_path = os.path.join(output_path, filename)

        # print("saving tensor as ", filename)
        torch.save(tensor, save_path)

        batch_dict[str(layer_idx)] = filename

def record_expert_select_result_dict():
    if begin_recording:
        global result_list
        global batch_dict
        result_list.append(batch_dict)
        batch_dict = {}

def dump_result():
    if begin_recording:
        global result_list
        global output_path
        global output_file
        file_name = Path(output_path) / output_file
        dump_to_results_dict(result_list, file_name)
        result_list = []
    