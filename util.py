import torch
import os
import shutil
import pickle as pkl
import torch.nn as nn
import copy
from torch.utils.data.distributed import DistributedSampler
from filelock import FileLock
import time

from data.lmdbReader import lmdbDataset, resizeNormalize
from config import config
from shutil import copyfile


# =======================
# Device
# =======================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =======================
# Alphabet (Character)
# =======================
with open(config['alpha_path'], 'r', encoding='utf-8') as f:
    alphabet_character = list(f.read().strip())

alphabet_character = ['START'] + alphabet_character + ['END']

alp2num_character = {char: idx for idx, char in enumerate(alphabet_character)}


# =======================
# DataLoader
# =======================
def get_dataloader(root, shuffle=False, distributed=False):
    if root.endswith('pkl'):
        with open(root, 'rb') as f:
            dataset = pkl.load(f)
    else:
        dataset = lmdbDataset(
            root,
            resizeNormalize((32, 32))
        )

    sampler = DistributedSampler(dataset, shuffle=shuffle) if distributed else None

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config['batch'],
        shuffle=(sampler is None and shuffle),
        num_workers=config['worker'],
        sampler=sampler,
        pin_memory=True
    )

    return dataloader, dataset


def get_data_package():
    train_dataset = []
    for root in config['train_dataset'].split(','):
        _, dataset = get_dataloader(root, shuffle=True, distributed=True)
        train_dataset.append(dataset)

    train_dataset_total = torch.utils.data.ConcatDataset(train_dataset)
    train_sampler = DistributedSampler(train_dataset_total)

    train_loader = torch.utils.data.DataLoader(
        train_dataset_total,
        batch_size=config['batch'],
        shuffle=False,
        num_workers=config['worker'],
        sampler=train_sampler,
        pin_memory=True
    )

    test_dataset = []
    for root in config['test_dataset'].split(','):
        _, dataset = get_dataloader(root, shuffle=False, distributed=True)
        test_dataset.append(dataset)

    test_dataset_total = torch.utils.data.ConcatDataset(test_dataset)
    test_sampler = DistributedSampler(test_dataset_total, shuffle=False)

    test_loader = torch.utils.data.DataLoader(
        test_dataset_total,
        batch_size=config['batch'],
        shuffle=False,
        num_workers=config['worker'],
        sampler=test_sampler,
        pin_memory=True
    )

    return train_loader, test_loader, train_sampler


# =======================
# Radical Alphabet
# =======================
r2num = {}
alphabet_radical = ['PAD']

with open(config['radical_path'], 'r', encoding='utf-8') as f:
    for line in f:
        alphabet_radical.append(line.strip('\n'))

alphabet_radical.append('$')

r2num = {char: idx for idx, char in enumerate(alphabet_radical)}


# =======================
# IDS Dictionary
# =======================
char_radical_dict = {}

with open(config['decompose_path'], 'r', encoding='utf-8') as f:
    for line in f:
        char, r_s = line.strip().split(':')
        char_radical_dict[char] = r_s.split(' ')


# =======================
# Label -> IDS tensor
# =======================
def convert_char(label):
    batch = len(label)

    text_tensor = torch.zeros(batch, 30, dtype=torch.long, device=DEVICE)

    for i, ch in enumerate(label):
        seq = char_radical_dict.get(ch, ['$'])
        seq = seq + ['$']

        for j, token in enumerate(seq[:30]):  # 防越界
            text_tensor[i][j] = r2num.get(token, 0)

    return text_tensor


# =======================
# Alphabet utils
# =======================
def get_radical_alphabet():
    return alphabet_radical


def get_alphabet():
    return alphabet_character

def get_radical_alphabet_ddcm():
    return alphabet_radical.remove('$')

def get_alphabet_ddcm():
    return alphabet_character

# =======================
# Main converter（
# =======================
def converter(label):
    string_label = label
    label = [list(i) for i in label]

    batch = len(label)

    length = torch.tensor([len(i) for i in label], dtype=torch.long, device=DEVICE)
    max_length = int(length.max())

    text_input = torch.zeros(batch, max_length, dtype=torch.long, device=DEVICE)

    for i in range(batch):
        for j in range(len(label[i]) - 1):
            text_input[i][j + 1] = alp2num_character.get(label[i][j], 0)

    # flatten
    sum_length = int(length.sum())
    text_all = torch.zeros(sum_length, dtype=torch.long, device=DEVICE)

    start = 0
    for i in range(batch):
        for j in range(len(label[i])):
            if j == len(label[i]) - 1:
                text_all[start + j] = alp2num_character['END']
            else:
                text_all[start + j] = alp2num_character.get(label[i][j], 0)
        start += len(label[i])

    # IDS sequences
    ids_sequences = []
    for i in range(batch):
        ids_sequence = []
        for char in label[i]:
            ids_sequence.append(char_radical_dict.get(char, ['$']))
        ids_sequences.append(ids_sequence)

    return length, text_input, text_all, string_label, ids_sequences



def tensor2str(tensor):
    alphabet = get_alphabet()
    string = ""
    for i in tensor:
        if i < len(alphabet) - 1:
            string += alphabet[i]
    return string


# =======================
# Screen check
# =======================
def must_in_screen():
    if not os.getenv("STY"):
        print("Run inside screen!")
        exit(0)

def saver():
    history_path = f'./history/{config["exp_name"]}'
    lock_path = history_path + ".lock"

    with FileLock(lock_path):
        if os.path.exists(history_path):
            shutil.rmtree(history_path)

        os.makedirs(history_path, exist_ok=True)

        print(f'**** Experiment Name: {config["exp_name"]} ****')

        time_str = time.strftime("%Y-%m-%d_%H-%M-%S")
        with open(os.path.join(history_path, "time.txt"), 'w') as f:
            f.write(f"Start time: {time_str}\n")

        # copy current files
        for file in os.listdir('./'):
            if os.path.isfile(file):
                copyfile(file, os.path.join(history_path, file))

        # copy model
        model_file = './model/ocr_encoder.py'
        if os.path.isfile(model_file):
            copyfile(model_file, os.path.join(history_path, 'ocr_encoder.py'))