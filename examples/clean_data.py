from data.preprocess import tune_preprocessing, preprocess, HP_DICT
import time
import glob
import os
from fire import Fire
import yaml
from joblib import repeat,Parallel,delayed

### TO DO! change back to multiple dirs......
### change structure to match segment funcs
def preprocess_data(
	audio_dirs:list[str],
	seg_dirs:list[str],
	cleaned_dirs:list[str],
	hp_loc:str,
	max_jobs=1,
	reprocess:bool=False,
	clean_type:str="ssq",
	reduce_noise:bool=True,
	):

	n_jobs = min(max_jobs,os.cpu_count())

	hp_name = os.path.join(hp_loc,'clean_params.yml')
	assert len(seg_dirs) == len(audio_dirs), print('must have same number of seg and audio dirs')
	assert len(cleaned_dirs) == len(audio_dirs), print('must have same number of cleaned and audio dirs')

	try:
		with open(hp_name,'r') as infile:
			hps = yaml.load(infile,loader=yaml.FullLoader)
		print("you have existing segmenting parameters! do you want to re-tune?")
		resp = input("(y)es/(n)o: ")
		if resp == 'y':
			hps = tune_preprocessing(
				audio_dirs,
				seg_dirs,
				hps,
				preprocess_type=clean_type,
				reduce_noise=reduce_noise,
			)
	except:
		# tunes parameters for preprocessing
		hps = tune_preprocessing(
			audio_dirs,
			seg_dirs,
			HP_DICT,
			preprocess_type=clean_type,
			reduce_noise=reduce_noise,
		)

	print("now cleaning data!")
	start = time.time()
	gen = zip(audio_dirs,cleaned_dirs,
		   repeat(hps),
		   repeat(reprocess),
		   repeat(clean_type),
		   repeat(reduce_noise))
	if n_jobs > 1:
		Parallel(n_jobs=n_jobs)(delayed(preprocess)(*args) for args in gen)
	else:
		for ad,cd,hp,rp,ct,rn in gen:
			preprocess(ad,cd,hp,rp,ct,rn)
	end = time.time()
	print(
		f"preprocessed your data in {end - start:.2f}s! If you have other files to preprocess, it'll probably take that long"
	)

if __name__ == "__main__":
	Fire(preprocess_data)