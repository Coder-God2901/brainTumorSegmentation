import glob
import numpy as np
import nibabel as nib
from sklearn.model_selection import train_test_split


def main():
    data_root = 'data/Task01_BrainTumour/Task01_BrainTumour'
    imgs = sorted(glob.glob(f"{data_root}/imagesTr/*.nii.gz"))
    lbls = sorted(glob.glob(f"{data_root}/labelsTr/*.nii.gz"))
    _, val_imgs, _, val_lbls = train_test_split(imgs, lbls, test_size=0.2, random_state=42)

    global_uniques = set()
    per_file = []
    for p in val_lbls:
        img = nib.load(p).get_fdata().astype(np.int32)
        uniques = np.unique(img)
        per_file.append((p, uniques.tolist()))
        global_uniques.update(uniques.tolist())

    print('Global unique labels in validation set:', sorted(global_uniques))
    print('\nPer-file unique labels (first 10 files):')
    for p, u in per_file[:10]:
        print(p, u)

if __name__ == '__main__':
    main()
