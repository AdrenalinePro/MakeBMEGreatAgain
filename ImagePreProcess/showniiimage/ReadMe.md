这部分的文件用于展示nii文件

其中，niiimageshow_single.py 用于展示单张nii图像，niiimageshow_multiple.py可以用于在一横向队列中展示一个文件夹下的所有nii图像

不过需要注意，这里的所有代码没有指定nii图像的Z切片层。如果需要连续对比Z轴上的图像，还是建议使用mricron

运行前检查必要的安装库：

```python
import os
import nibabel as nb
import matplotlib.pyplot as plt
import numpy as np
```

使用python3运行。

后面或许会写一个程序用来轮流展示一张图片的多层切换效果。类似幻灯片这种。

补充，可以修改某些参数来展示在不同层上的图像效果，我们涉及的nii图像基本是三维层切片，而评价图像预处理的结果不能仅仅看中间层，而应该要把两端包括进去。