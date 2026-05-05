from third_party.segment import segment,tune_segmenting_params,P
from fire import Fire
import os
import yaml
from joblib import repeat,Parallel,delayed

def segment_data(audio_dirs:list[str],
				 seg_dirs:list[str],
				 hp_loc:str,
				 max_jobs = 1
				 ):

	"""
	example script for segmenting data. takes audio from
	audio_dirs, places corresponding segmentation files
	into seg_dirs. saves the hyperparameters used
	for segmentation in hp_loc. does this in parallel

	inputs
	--------
		audio_dirs: list of folders with audio
		seg_dirs: list of folders to place segmentation decisions
		hp_loc: location to save/load segmentation hyperparams
		max_jobs: max number of parallel segmentation jobs to run

	returns
	--------
		nada
	"""

	n_jobs = min(max_jobs,os.cpu_count())

	hp_name = os.path.join(hp_loc,'segment_params.yml')
	assert len(seg_dirs) == len(audio_dirs), print('must have same number of seg and audio dirs')
	try:
		with open(hp_name,'r') as infile:
			p = yaml.load(infile,loader=yaml.FullLoader)
		print("you have existing segmenting parameters! do you want to re-tune?")
		resp = input("(y)es/(n)o: ")
		if resp == 'y':
			p = tune_segmenting_params(audio_dirs, P, img_fn='temp.pdf')
	except:
		p = tune_segmenting_params(audio_dirs, P, img_fn='temp.pdf')

		with open(hp_name,'w') as outfile:
			yaml.dump(p,outfile)
	
	gen = zip(audio_dirs,seg_dirs,repeat(p))
	if n_jobs > 1:
		Parallel(n_jobs=n_jobs)(delayed(segment)(*args) for args in gen)
	else:
		for ad,sd,p in gen:
			segment(ad,sd,p)



if __name__ == "__main__":
	pass