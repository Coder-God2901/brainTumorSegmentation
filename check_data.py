import nibabel as nib, glob
files = sorted(glob.glob("data/Task01_BrainTumour/Task01_BrainTumour/imagesTr/*.nii.gz"))
print("Found", len(files), "files")
if len(files) > 0:
    img = nib.load(files[0]).get_fdata()
    print("Shape:", img.shape)
