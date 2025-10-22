# import nibabel as nib, torch, cv2, matplotlib.pyplot as plt
# from model_train import make_model
# from dataset_25d import load_nifti

# model = make_model("efficientnet-b5", encoder_weights=None, device="cuda")
# model.load_state_dict(torch.load("checkpoints/best_model.pth"))
# model.eval()

# sample_img = "data/Task01_BrainTumour/imagesTr/BRATS_001.nii.gz"
# sample_lbl = "data/Task01_BrainTumour/labelsTr/BRATS_001.nii.gz"

# img_vol = load_nifti(sample_img)
# lbl_vol = load_nifti(sample_lbl)
# z = img_vol.shape[2]//2

# # build triplet
# s1, s2, s3 = img_vol[:,:,z-1], img_vol[:,:,z], img_vol[:,:,z+1]
# x = (np.stack([s1,s2,s3], axis=-1) - img_vol.mean()) / (img_vol.std()+1e-8)
# x = torch.tensor(np.transpose(cv2.resize(x, (256,256)), (2,0,1))).unsqueeze(0).float().cuda()

# with torch.no_grad():
#     pred = torch.softmax(model(x), 1).argmax(1).cpu().numpy()[0]
# pred_rs = cv2.resize(pred, (img_vol.shape[1], img_vol.shape[0]), interpolation=cv2.INTER_NEAREST)

# plt.figure(figsize=(12,4))
# plt.subplot(1,3,1); plt.imshow(img_vol[:,:,z], cmap='gray'); plt.title('Input')
# plt.subplot(1,3,2); plt.imshow(lbl_vol[:,:,z], cmap='nipy_spectral'); plt.title('Ground Truth')
# plt.subplot(1,3,3); plt.imshow(pred_rs, cmap='nipy_spectral'); plt.title('Prediction')
# plt.show()

# # --------------------------------------------------------------------------

# # BraTS Dataset Implementation

