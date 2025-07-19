from transformers import GPT2LMHeadModel, get_linear_schedule_with_warmup
from dataset import CaptionsDataset
from torch.optim import AdamW
from tqdm import tqdm
import argparse
import pickle
import torch
import json
import sys
import os

dataset = CaptionsDataset(data_path="./data/conceptual_clipcap.pkl",prefix_length=10)
