import fire
from data.real_data import get_segmented_audio
from data.data_utils import get_loaders
from train.model_cv import model_cv_lambdas
from train.eval import full_eval_model
import os
import numpy as np
import pickle
import glob


def run_experiments(data_path="", model_path="", seed=92, nEpochs=200):

    if not os.path.isdir(model_path):
        os.mkdir(model_path)

    pupil_audio_subdir = "[b,g,p]*[0-9]/motif_audio/synchro_cleaned_v1"
    pupil_seg_subdir = "[b,g,p]*[0-9]/motif_txt"
    tutor_audio_subdir = "[b,g,p]*[0-9]_tutor/motif_audio_tutor/synchro_cleaned_v1"
    tutor_seg_subdir = "[b,g,p]*[0-9]_tutor/motif_txt_tutor"

    pupil_audios, sr = get_segmented_audio(
        data_path,
        data_path,
        audio_subdir=pupil_audio_subdir,
        seg_subdir=pupil_seg_subdir,
        envelope=False,
        context_len=0.15,
        audio_type="_cleaned.wav",
        seg_type=".txt",
        max_pairs=2000,
        seed=seed,
    )
    tutor_audios, _ = get_segmented_audio(
        data_path,
        data_path,
        audio_subdir=tutor_audio_subdir,
        seg_subdir=tutor_seg_subdir,
        envelope=False,
        context_len=0.15,
        audio_type="_cleaned.wav",
        seg_type=".txt",
        max_pairs=2000,
        seed=seed,
    )
    audios = pupil_audios + tutor_audios

    dls = get_loaders(np.vstack(audios), cv=True, train_size=0.6, seed=seed)  # 1/sr,\
    # nEpochs=400,model_path=model_path)
    pupil_tutor_model = model_cv_lambdas(
        dls, 1 / sr, nEpochs=nEpochs, model_path=model_path
    )

    if os.path.isfile(os.path.join(model_path, "eval_data.pkl")):
        with open(os.path.join(model_path, "eval_data.pkl"), "rb") as f:
            pupil_tutor_eval_dict = pickle.load(f)
    else:
        (
            pupil_tutor_r2s,
            pupil_tutor_best,
            pupil_tutor_resids,
            pupil_tutor_spec_ratio,
            pupil_tutor_specs,
            pupil_tutor_ext,
        ) = full_eval_model(
            pupil_tutor_model,
            dls,
            audios,
            1 / sr,
            use_results=False,
            n_int=50,
            plot_dir=model_path,
            plot_steps=False,
        )

        pupil_tutor_eval_dict = {
            "r2s": pupil_tutor_r2s,
            "best_data": pupil_tutor_best,
            "resids": pupil_tutor_resids,
            "spec_ratio": pupil_tutor_spec_ratio,
            "specs": pupil_tutor_specs,
            "ext": pupil_tutor_ext,
        }

        with open(os.path.join(model_path, "eval_data.pkl"), "wb") as f:
            pickle.dump(pupil_tutor_eval_dict, f)

    ######## functions #############

    pupils = glob.glob(os.path.join(data_path, "[b,g,p]*[0-9]"))
    tutors = [p + "_tutor" for p in pupils]
    for p, t in zip(pupils, tutors):
        pupil, tutor = p.split("/")[-1], t.split("/")[-1]
        print(f"here's where we'll get model functions for {pupil},{tutor}")


if __name__ == "__main__":
    fire.Fire(run_experiments)
