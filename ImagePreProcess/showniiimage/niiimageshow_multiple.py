import os
import nibabel as nb
import matplotlib.pyplot as plt
import numpy as np

# 设置包含nii文件的文件夹路径
folder_path = "C:/Users/23759/Desktop/MB319345/T1_FLASH_GRE_AX"  # 修改为你的实际路径

# 获取所有nii文件并排序
nii_files = sorted([f for f in os.listdir(folder_path) 
                  if f.endswith(('.nii', '.nii.gz'))])

if not nii_files:
    print("错误：文件夹中未找到.nii或.nii.gz文件！")
    exit()

# 创建画布和子图布局
fig, axes = plt.subplots(1, len(nii_files), 
            figsize=(15, 5))  # 根据文件数量自动调整宽度
if len(nii_files) == 1:  # 处理单个文件的情况
    axes = [axes]

# 遍历并显示每个文件
for idx, filename in enumerate(nii_files):
    try:
        img_path = os.path.join(folder_path, filename)
        nii_img = nb.load(img_path)
        data = np.asanyarray(nii_img.dataobj)        
        # 获取中间切片
        slice_idx = data.shape[2] // 2
        slice_data = data[:, :, slice_idx].T       
        # 显示图像
        axes[idx].imshow(slice_data, cmap='Greys_r')
        axes[idx].axis('off')
        axes[idx].set_title(f"{filename}\nShape: {data.shape}")
        
    except Exception as e:
        print(f"处理文件 {filename} 时出错: {str(e)}")

# 调整布局并显示
plt.tight_layout()
plt.subplots_adjust(wspace=0.05)  # 调整子图间距
plt.show()