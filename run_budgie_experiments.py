import fire
from data.real_data import get_segmented_audio
from data.data_utils import get_loaders
from train.model_cv import model_cv_lambdas
from train.eval import full_eval_model
import os
import numpy as np
import pickle
from analysis.analysis import get_budgie_fncs


def run_experiments(budgie_data_path="", model_path="", seed=92):

    if not os.path.isdir(model_path):
        os.mkdir(model_path)
    budgie_model_path = model_path

    print("budgie training and analysis")
    budgie_audio_path, budgie_seg_path = budgie_data_path, budgie_data_path

    audios, sr = get_segmented_audio(
        budgie_audio_path,
        budgie_seg_path,
        audio_subdir="",
        seg_subdir="",
        envelope=False,
        context_len=0.15,
        audio_type="_cleaned.wav",
        seg_type="izationTimestamp.txt",
        max_pairs=3000,
        seed=seed,
    )
    dls = get_loaders(np.vstack(audios), cv=True, train_size=0.6, seed=seed)

    budgie_model = model_cv_lambdas(
        dls, 1 / sr, nEpochs=400, model_path=budgie_model_path
    )

    if os.path.isfile(os.path.join(budgie_model_path, "eval_data.pkl")):
        with open(os.path.join(budgie_model_path, "eval_data.pkl"), "rb") as f:
            budgie_eval_dict = pickle.load(f)
    else:
        (
            budgie_r2s,
            budgie_best,
            budgie_resids,
            budgie_spec_ratio,
            budgie_specs,
            budgie_ext,
        ) = full_eval_model(
            budgie_model,
            dls,
            audios,
            1 / sr,
            use_results=False,
            n_int=50,
            plot_dir=budgie_model_path,
            plot_steps=False,
        )

        budgie_eval_dict = {
            "r2s": budgie_r2s,
            "best_data": budgie_best,
            "resids": budgie_resids,
            "spec_ratio": budgie_spec_ratio,
            "specs": budgie_specs,
            "ext": budgie_ext,
        }

        with open(os.path.join(budgie_model_path, "eval_data.pkl"), "wb") as f:
            pickle.dump(budgie_eval_dict, f)

    if os.path.isfile(os.path.join(budgie_model_path, "func_data.pkl")):
        with open(os.path.join(budgie_model_path, "func_data.pkl"), "rb") as f:
            budgie_func_dict = pickle.load(f)
    else:
        b_o, b_g, b_k, b_w, b_a = get_budgie_fncs(
            budgie_model,
            budgie_audio_path,
            budgie_seg_path,
            seed=seed,
            cut_percentile=75,
        )

        budgie_func_dict = {
            "omegas": b_o,
            "gammas": b_g,
            "kernels": b_k,
            "weights": b_w,
            "audio": b_a,
        }

        with open(os.path.join(budgie_model_path, "func_data,pkl"), "wb") as f:
            pickle.dump(budgie_func_dict, f)


if __name__ == "__main__":
    fire.Fire(run_experiments)
