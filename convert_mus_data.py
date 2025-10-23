import numpy as np
import fire
import os
import glob
import pandas as pd


def convert_mus(data_path,new_subdir='segs'):


    mats = glob.glob(os.path.join(data_path,'*.mat.csv'))
    data_days = glob.glob(os.path.join(data_path,'[0-9][0-9]'))

    sylls = []
    for mat in mats:

        m = pd.read_csv(mat)
        syll = np.unique(m.type)[0]
        sylls.append(syll)

    #seg_paths = []
    new_path = os.path.join(data_path,new_subdir)
    if not os.path.isdir(new_path):
        os.mkdir(new_path)
    #for syll in sylls:
        
    #    seg_paths.append(new_path)

    for mat, syll in zip(mats,sylls):

        m = pd.read_csv(mat)
        #unique_files = np.unique(m.file)

        for d in data_days:
            day = d.split('/')[-1]
            print(day)
            target_subdir = os.path.join(new_path,day)
            if not os.path.isdir(target_subdir):
                os.mkdir(target_subdir)
            target_subdir = os.path.join(target_subdir,'syllable_' + syll)
            if not os.path.isdir(target_subdir):
                os.mkdir(target_subdir)
                
            unique_files = glob.glob(os.path.join(d,'*.wav'))
            for f in unique_files:

                f = f.split('/')[-1]
                file_subset = m.loc[m.file == f,:].copy()
                ons = file_subset.onsetInFile
                offs = file_subset.offsetInFile
                file_subset['ons'] = file_subset.onsetInFile.transform(lambda x: np.round(float(x.split(' ')[0]),4))
                file_subset['offs'] = file_subset.offsetInFile.transform(lambda x: np.round(float(x.split(' ')[0]),4))
                ons = file_subset.ons.to_numpy()
                offs = file_subset.offs.to_numpy()
                new_fn = f[:-4] + '.txt'
                if len(ons)>0 and len(offs) > 0:
                    onoffs = np.stack([ons,offs],axis=-1)

                else:
                    #print(f'no onoffs in {f}')
                    onoffs = []
                np.savetxt(fname=os.path.join(target_subdir,new_fn),X=onoffs)


    return

if __name__ == '__main__':

    fire.Fire(convert_mus)

