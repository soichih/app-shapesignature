<h1 align="center">Shape Signature</h1>

<p align="center">
  Computes a 128-value shape signature for each tract mask using the flatten layer of a TensorFlow 3D CNN trained to classify TractSeg bundles, and writes them to signature.json.
</p>

<p align="center">
  <a href="https://github.com/brainlife/abcd-spec"><img src="https://img.shields.io/badge/ABCD_Spec-v1.1-green.svg" alt="ABCD Spec v1.1"></a>
  <a href="https://doi.org/10.25663/brainlife.app.141"><img src="https://img.shields.io/badge/Brainlife-brainlife.app.141-blue.svg" alt="Run on brainlife.io"></a>
</p>

## At a glance

<!-- brainlife-app:begin (generated from the App manifest; edit the manifest, not this block) -->

> **Computes a 128-value shape signature for each tract mask using the flatten layer of a TensorFlow 3D CNN trained to classify TractSeg bundles, and writes them to signature.json.**

Crops each tract mask to its bounding box, resizes it to 64x64x64 and passes it through a Keras 3D convolutional network trained on TractSeg outputs of about 100 subjects, keeping the 128-element flatten layer output as a numerical descriptor of the tract shape. Use it to obtain feature vectors for comparing or clustering tract shapes. Resizing stretches each mask to the box independently along each axis, so absolute size is not preserved.

| | |
|---|---|
| **Inputs** | `neuro/tractmasks` |
| **Outputs** | `raw` |
| **Container** | `brainlife/tensorflow:1.12.0` |
| **Resources** | 16 CPUs · 0.17 h walltime |
| **Run it** | [doi.org/10.25663/brainlife.app.141](https://doi.org/10.25663/brainlife.app.141) |

#### Inputs

| Input | Datatype | Required | Description |
|---|---|---|---|
| `masks` | `neuro/tractmasks` | yes | Collection of binary tract masks (NIfTI), for example TractSeg bundle masks. |

#### Outputs

| Output | Datatype | Description |
|---|---|---|
| `raw` | `raw` | signature.json mapping each mask file name to its 128 flatten-layer values. |

<!-- brainlife-app:end -->

## Running the App

### On brainlife.io

Open the App at [doi.org/10.25663/brainlife.app.141](https://doi.org/10.25663/brainlife.app.141), choose `neuro/tractmasks` from your project and press **Execute**. The results are saved to your project as `raw`.

### Locally

You need [Singularity/Apptainer](https://apptainer.org/) (the App runs in `brainlife/tensorflow:1.12.0`), [`jq`](https://jqlang.github.io/jq/).

1. Clone this repository.
2. Create `config.json` with the paths to your inputs and the parameters:

   ```json
   {
       "masks": "/path/to/masks/masks"
   }
   ```

3. Run `./main`.

## Method

We ran [TractSeg](https://github.com/MIC-DKFZ/TractSeg) on HCP3T subjects and generated 3D mask images of 78 individual tracts for each subject. We then trained a 3D convolutional model to predict the name(label) of each tracts. We've designed the model so that it has a small bottleneck (128nodes) between Conv3D and dense network. The model was able to predict at near 100% accuracy. This means that those 128 nodes at the bottleneck contains enough information to accurately distinguish various human tract shapes. The dense layer part of the model is then discarded so that the model will output those 128 values from a given tract masks. These values could then be used to numerically describe the shape of input tract.

## Classification Model

The classification model was trained using TractSeg outputs from roughtly 100 subjects (20% of them used to validate the model). The model can accurately classify each tracts at near 100% accuracy, but we discard the layer below the flatten layer as we are not interested in classifiying tracts - we use the trained model to extract activation outputs from the conv3d layers.

```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv3d (Conv3D)              (None, 60, 60, 60, 8)     1008      
_________________________________________________________________
conv3d_1 (Conv3D)            (None, 56, 56, 56, 16)    16016     
_________________________________________________________________
max_pooling3d (MaxPooling3D) (None, 14, 14, 14, 16)    0         
_________________________________________________________________
dropout (Dropout)            (None, 14, 14, 14, 16)    0         
_________________________________________________________________
conv3d_2 (Conv3D)            (None, 11, 11, 11, 16)    16400     
_________________________________________________________________
conv3d_3 (Conv3D)            (None, 8, 8, 8, 16)       16400     
_________________________________________________________________
max_pooling3d_1 (MaxPooling3 (None, 2, 2, 2, 16)       0         
_________________________________________________________________
dropout_1 (Dropout)          (None, 2, 2, 2, 16)       0         
_________________________________________________________________
flatten (Flatten)            (None, 128)               0         
_________________________________________________________________
dense (Dense)                (None, 72)                9288      
=================================================================
Total params: 59,112
Trainable params: 59,112
Non-trainable params: 0
_________________________________________________________________

```

Training / Test stats

```
5235/5235 [==============================] - 71s 14ms/step - loss: 0.0098 - acc: 0.3727 - val_loss: 0.0045 - val_acc: 0.7815
Epoch 2/8
5235/5235 [==============================] - 64s 12ms/step - loss: 0.0022 - acc: 0.8877 - val_loss: 6.6277e-04 - val_acc: 0.9672
Epoch 3/8
5235/5235 [==============================] - 64s 12ms/step - loss: 5.4972e-04 - acc: 0.9759 - val_loss: 1.4308e-04 - val_acc: 0.9931
Epoch 4/8
5235/5235 [==============================] - 64s 12ms/step - loss: 2.3532e-04 - acc: 0.9901 - val_loss: 1.1028e-04 - val_acc: 0.9947
Epoch 5/8
5235/5235 [==============================] - 64s 12ms/step - loss: 1.9655e-04 - acc: 0.9914 - val_loss: 9.6726e-05 - val_acc: 0.9954
Epoch 6/8
5235/5235 [==============================] - 64s 12ms/step - loss: 1.4950e-04 - acc: 0.9943 - val_loss: 1.1865e-04 - val_acc: 0.9954
Epoch 7/8
5235/5235 [==============================] - 65s 12ms/step - loss: 9.4822e-05 - acc: 0.9966 - val_loss: 5.9258e-05 - val_acc: 0.9969
Epoch 8/8
5235/5235 [==============================] - 65s 12ms/step - loss: 1.0702e-04 - acc: 0.9952 - val_loss: 8.0987e-05 - val_acc: 0.9962
evaluating
728/728 [==============================] - 5s 7ms/step
Test loss: 7.794678107513892e-05
Test accuracy: 0.9958791208791209

```

## Citing

If you use this App, please cite it, the software it builds on and brainlife.io:

- Shape Signature. brainlife.io App. [doi:10.25663/brainlife.app.141](https://doi.org/10.25663/brainlife.app.141)
- Wasserthal et al. TractSeg - Fast and accurate white matter tract segmentation. *NeuroImage* 183:239-253 (2018). [doi:10.1016/j.neuroimage.2018.07.070](https://doi.org/10.1016/j.neuroimage.2018.07.070)
- Hayashi S, Caron BA, Heinsfeld AS, et al. brainlife.io: a decentralized and open-source cloud platform to support neuroscience research. *Nature Methods* 21:809–813 (2024). [doi:10.1038/s41592-024-02237-2](https://doi.org/10.1038/s41592-024-02237-2)
- Avesani P, McPherson B, Hayashi S, et al. The open diffusion data derivatives, brain data upcycling via integrated publishing of derivatives and reproducible open cloud services. *Scientific Data* 6:69 (2019). [doi:10.1038/s41597-019-0073-y](https://doi.org/10.1038/s41597-019-0073-y)

## Authors

Soichi Hayashi
