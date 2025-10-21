from dataset_25d import TwoPointFiveDDataset
import glob
import matplotlib.pyplot as plt

data_root = "data/Task01_BrainTumour/Task01_BrainTumour"
img_files = sorted(glob.glob(f"{data_root}/imagesTr/*.nii.gz"))
lbl_files = sorted(glob.glob(f"{data_root}/labelsTr/*.nii.gz"))

ds = TwoPointFiveDDataset(img_files, lbl_files)
x, y = ds[0]

print(f"✅ Image tensor shape: {x.shape}")  # Should be [4, 240, 240]
print(f"✅ Label tensor shape: {y.shape}")  # Should be [240, 240]

# visualize one slice
plt.figure(figsize=(10,5))
plt.subplot(1,2,1); plt.imshow(x[0].numpy(), cmap='gray'); plt.title("MRI Channel 1")
plt.subplot(1,2,2); plt.imshow(y.numpy(), cmap='jet'); plt.title("Mask")
plt.show()
