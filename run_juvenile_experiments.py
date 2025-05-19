import fire
from data.real_data import get_segmented_audio
from data.data_utils import get_loaders
from train.model_cv import model_cv_lambdas
from train.train import load_model
from train.eval import full_eval_model
import os
import numpy as np
import matplotlib.pyplot as plt
from visualization.model_vis import format_axes
import pickle
from analysis.analysis import get_budgie_fncs,get_marmo_fncs
import gc 
import glob

def run_experiments(juvenile_path='',model_path='',seed=92):
    


    pass 


if __name__ == '__main__':

    fire.Fire(run_experiments)