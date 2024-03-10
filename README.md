# PlantTraits2024---FGVC11
The provided code defines and builds a deep learning model using the Keras library. The model consists of two input branches: one for image data and another for tabular/feature data. The image branch utilizes the EfficientNetV2 backbone for feature extraction, while the tabular branch uses dense layers with SELU activation. The outputs from both branches are concatenated and fed into two output layers: the main output layer (head) and an auxiliary output layer (aux_head). The model is compiled with the Adam optimizer and uses a custom R2Loss function for the loss calculation. The R2Metric is used as the evaluation metric for the main task.

# Deep Learning Model with Image and Tabular Inputs

This repository contains the code for building a deep learning model that takes both image and tabular data as inputs. The model is built using the Keras library and utilizes the EfficientNetV2 backbone for image feature extraction.

## Model Architecture

The model consists of two input branches:
1. Image Branch:
   - Takes an input shape of `(*CFG.image_size, 3)`, where `CFG.image_size` represents the dimensions of the input images.
   - Uses the EfficientNetV2 backbone from the `keras_cv.models` module to extract features from the input images.
   - Applies global average pooling and dropout to the backbone output.

2. Tabular/Feature Branch:
   - Takes an input shape of `(len(FEATURE_COLS),)`, where `FEATURE_COLS` represents the number of tabular features.
   - Consists of two dense layers with SELU activation and dropout.

The outputs from both branches are concatenated and fed into two output layers:
- Main Output Layer (head):
  - Dense layer with `CFG.num_classes` units and no activation function.
- Auxiliary Output Layer (aux_head):
  - Dense layer with `CFG.aux_num_classes` units and ReLU activation function.

## Model Compilation

The model is compiled with the following configuration:
- Optimizer: Adam optimizer with a learning rate of 1e-4.
- Loss Functions:
  - Main Output Layer (head): R2Loss without masking.
  - Auxiliary Output Layer (aux_head): R2Loss with masking to ignore `NaN` auxiliary labels.
- Loss Weights:
  - Main Output Layer (head): 1.0
  - Auxiliary Output Layer (aux_head): 0.3
- Evaluation Metric: R2Metric, applied only to the main task.

## Learning Rate Scheduler

The code includes a learning rate scheduler callback `get_lr_callback` that can be used during model training. The scheduler adjusts the learning rate based on the specified parameters:
- `batch_size`: The batch size used during training (default: 8).
- `mode`: The learning rate decay mode. Can be 'cos' (cosine decay), 'exp' (exponential decay), or 'step' (step decay) (default: 'cos').
- `epochs`: The total number of epochs for training (default: 10).
- `plot`: If set to True, plots the learning rate curve (default: False).

The learning rate schedule follows these steps:
1. Ramp-up phase: The learning rate increases linearly from `lr_start` to `lr_max` over `lr_ramp_ep` epochs.
2. Sustain phase: The learning rate remains constant at `lr_max` for `lr_sus_ep` epochs.
3. Decay phase: The learning rate decays according to the specified `mode` over the remaining epochs.

The learning rate at each epoch is determined by the `lrfn` function, which takes the current epoch as input and returns the corresponding learning rate.

To use the learning rate scheduler, create an instance of the callback using `get_lr_callback` and pass it to the `model.fit` function during training.

## Usage

To use this model:
1. Ensure you have the necessary dependencies installed, including Keras and keras_cv.
2. Prepare your image and tabular data according to the required input shapes.
3. Instantiate the model using the provided code.
4. Compile the model with the desired optimizer, loss functions, loss weights, and evaluation metric.
5. Create an instance of the learning rate scheduler callback using `get_lr_callback` with the desired parameters.
6. Train the model using your prepared data and pass the learning rate scheduler callback to `model.fit`.
7. Evaluate the model's performance using appropriate metrics.

## Additional Notes

- The model summary can be viewed by calling `model.summary()`.
- A visual representation of the model architecture can be plotted using `keras.utils.plot_model(model, show_shapes=True)`.

Feel free to explore and modify the model architecture, hyperparameters, and learning rate scheduler to suit your specific requirements.
