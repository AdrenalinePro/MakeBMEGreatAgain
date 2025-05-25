N4BiasFieldCorrection，即使用N4算法进行场不均匀校正。这一功能在ANTs软件与SimpleITK软件中均有提供，不过这里主要介绍ANTs里面提供的N4算法进行场不均匀校正。

这里的寻找图像逻辑是在文件夹的任意一个子文件夹中寻找T1开头的文件夹，然后在这一文件夹中寻找nii/nii.gz文件进行处理。

在终端中输入可以看到：

```bash
$ N4BiasFieldCorrection

COMMAND:
     N4BiasFieldCorrection
          N4 is a variant of the popular N3 (nonparameteric nonuniform normalization)
          retrospective bias correction algorithm. Based on the assumption that the
          corruption of the low frequency bias field can be modeled as a convolution of
          the intensity histogram by a Gaussian, the basic algorithmic protocol is to
          iterate between deconvolving the intensity histogram by a Gaussian, remapping
          the intensities, and then spatially smoothing this result by a B-spline modeling
          of the bias field itself. The modifications from and improvements obtained over
          the original N3 algorithm are described in the following paper: N. Tustison et
          al., N4ITK: Improved N3 Bias Correction, IEEE Transactions on Medical Imaging,
          29(6):1310-1320, June 2010.

OPTIONS:
     -d, --image-dimensionality 2/3/4
          This option forces the image to be treated as a specified-dimensional image. If
          not specified, N4 tries to infer the dimensionality from the input image.

     -i, --input-image inputImageFilename
          A scalar image is expected as input for bias correction. Since N4 log transforms
          the intensities, negative values or values close to zero should be processed
          prior to correction.

     -x, --mask-image maskImageFilename
          If a mask image is specified, the final bias correction is only performed in the
          mask region. If a weight image is not specified, only intensity values inside
          the masked region are used during the execution of the algorithm. If a weight
          image is specified, only the non-zero weights are used in the execution of the
          algorithm although the mask region defines where bias correction is performed in
          the final output. Otherwise bias correction occurs over the entire image domain.
          See also the option description for the weight image. If a mask image is *not*
          specified then the entire image region will be used as the mask region. Note
          that this is different than the N3 implementation which uses the results of Otsu
          thresholding to define a mask. However, this leads to unknown anatomical regions
          being included and excluded during the bias correction.

     -r, --rescale-intensities 0/(1)
          At each iteration, a new intensity mapping is calculated and applied but there
          is nothing which constrains the new intensity range to be within certain values.
          The result is that the range can "drift" from the original at each iteration.
          This option rescales to the [min,max] range of the original image intensities
          within the user-specified mask.

     -w, --weight-image weightImageFilename
          The weight image allows the user to perform a relative weighting of specific
          voxels during the B-spline fitting. For example, some studies have shown that N3
          performed on white matter segmentations improves performance. If one has a
          spatial probability map of the white matter, one can use this map to weight the
          b-spline fitting towards those voxels which are more probabilistically
          classified as white matter. See also the option description for the mask image.

     -s, --shrink-factor 1/2/3/(4)/...
          Running N4 on large images can be time consuming. To lessen computation time,
          the input image can be resampled. The shrink factor, specified as a single
          integer, describes this resampling. Shrink factors <= 4 are commonly used.Note
          that the shrink factor is only applied to the first two or three dimensions
          which we assume are spatial.

     -c, --convergence [<numberOfIterations=50x50x50x50>,<convergenceThreshold=0.0>]
          Convergence is determined by calculating the coefficient of variation between
          subsequent iterations. When this value is less than the specified threshold from
          the previous iteration or the maximum number of iterations is exceeded the
          program terminates. Multiple resolutions can be specified by using 'x' between
          the number of iterations at each resolution, e.g. 100x50x50.

     -b, --bspline-fitting [splineDistance,<splineOrder=3>]
                           [initialMeshResolution,<splineOrder=3>]
          These options describe the b-spline fitting parameters. The initial b-spline
          mesh at the coarsest resolution is specified either as the number of elements in
          each dimension, e.g. 2x2x3 for 3-D images, or it can be specified as a single
          scalar parameter which describes the isotropic sizing of the mesh elements. The
          latter option is typically preferred. For each subsequent level, the spline
          distance decreases in half, or equivalently, the number of mesh elements doubles
          Cubic splines (order = 3) are typically used. The default setting is to employ a
          single mesh element over the entire domain, i.e., -b [1x1x1,3].

     -t, --histogram-sharpening [<FWHM=0.15>,<wienerNoise=0.01>,<numberOfHistogramBins=200>]
          These options describe the histogram sharpening parameters, i.e. the
          deconvolution step parameters described in the original N3 algorithm. The
          default values have been shown to work fairly well.

     -o, --output correctedImage
                  [correctedImage,<biasField>]
          The output consists of the bias corrected version of the input image.
          Optionally, one can also output the estimated bias field.

     --version
          Get Version Information.

     -v, --verbose (0)/1
          Verbose output.

     -h
          Print the help menu (short version).

     --help
          Print the help menu.

```

核心参数

-d 指的是图像维度，一般来讲nii图像为二维单张图像或者是二维切片图像的集合，所以选择2/3即可

-i 输入文件的目录 ，这个没啥好说的

-x 输入蒙版的目录，来指定场不均匀校正的区域。不指定该参数的话，会利用图像本身生成一层模板

-r 重绘幅度，如果你懂一点StableDiffusion的话这玩意儿很好理解，就是算法对原始图像的修改程度。*一般不需要更改这个参数*

-s Shrink因子，缩小图像加速计算，s越大图像越小，一般去s为小于等于4的整数

s从4到3的时候计算时间会变为原来的大约3倍，在没有较为剧烈的不均匀场变化的时候修改这个值没太大意义

-w 权重图像， 我的理解是在去样条距离计算的时候赋予其一个权重而非默认数值。*一般不需要更改这个参数*

-c 收敛条件（格式：`[迭代次数,收敛阈值]`) 迭代次数一般会写成'100x100x100、80x80x80x80'这种形式，收敛阈值视图像具体情况决定。

-b 样条距离，即B-Spline的距离，减小样条距离则增加控制点密度，从而增加计算所需时间。

控制点数量应当大于样条阶数，否则可能引起报错。

-o 输出控制，[correctedImage, biasField]的形式，表示输出校正之后的图像的名称与不均匀场的图像名称。

-t 直方图锐化参数，这个参数的详细调节要涉及一些直方图的相关知识，不展开细讲

默认值：通常为 [0.15, 0.01, 200]（格式：--histogram-sharpening [winsor,radius,numPoints]）

winsor：截断直方图尾部的阈值（范围 0~0.5，例如 0.15 表示截断15%的高/低强度值）。

radius：直方图计算的局部邻域半径（默认 0.01）。

numPoints：直方图的点数（默认 200，或许可以调节成图像的较短边像素数量？）

常见调整场景：

噪声较多的图像：降低 winsor（如 0.1）以减少锐化对噪声的敏感度。

低对比度图像：提高 winsor（如 0.25）以增强结构分离。

高分辨率图像：增大 numPoints（如 500）以细化直方图。

要我说的话，numpoints选取越高图像对不均匀场的分化越细致。N4算法会对于图像的强度（场强度）生成一个直方图数据来估计偏置场，随后再根据这一偏置场进行图像校正。如果控制点数量太多，那么图像细节中的某些有明显对比度的结构也会被识别为偏置场，这时候反而会降低图像细节的对比度。

example：

若大部分使用默认参数：

```bash
N4BiasFieldCorrection -d 3 -i input.nii.gz -o output_corrected.nii.gz
```

自定义参数：

```bash
N4BiasFieldCorrection -d 3 \
  -i input.nii.gz \
  -o output_corrected.nii.gz \
  -s 2 \  # 缩小图像加速处理
  -c "[200x200x200,0.0001]" \  # 迭代200次，收敛阈值0.0001
  -b 200 \  
  --histogram-sharpening "[0.15,0.01,200]"  # 直方图锐化
```

校正之后的图像，为1.5T T1权MRI像

![Figure_2](.\Figure_2.png)



![Figure_1](.\Figure_1.png)

从左侧偏置场中清晰可见算法对偏置场的识别情况。鉴于0.35T的图像在MRI结构上有先天优势（这要涉及0.35T的MRI长啥样的问题，若果你实际去见过你就会理解了为啥永磁体拍脑子不会有明显场不均匀的情况了）

我自己找出的参数：

1.5T T1 MRI

```python
"N4BiasFieldCorrection",
"-d", "3",
"-v", "1",
"-s", "4",
"-b", "[120]",
"-c", "[100x100x100x100 ,0.001]",
"-i", input_path,
"--histogram-sharpening ","[0.25,0.01,352]",
"-o", "[%s,%s]" % (corrected_path, bias_path)
```

0.35T T1 MRI

```python
"N4BiasFieldCorrection",
"-d", "3",
"-v", "1",
"-s", "4",
"-b", "[100]",
"-c", "[80x80x80x80 ,0.001]",
"-i", input_path,
"--histogram-sharpening","[0.12,0.01,200]",
"-o", "[%s,%s]" % (corrected_path, bias_path)
```

