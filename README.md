Apple Leaf Disease Semantic Segmentation
This repository contains a Jupyter notebook for semantic segmentation of apple leaf diseases using deep learning models. The notebook implements several segmentation architectures to identify and segment diseased regions on apple leaves from the ATLDSD dataset.
Overview
The project focuses on detecting and segmenting apple leaf diseases such as Alternaria leaf spot, Brown spot, Gray spot, Rust, and healthy leaves. It uses models like UNet (with MobileNetV2 backbone), DeepLabV3+, FCN, and SegNet. Key features include:

Data augmentation and oversampling for imbalanced classes.
Custom loss functions (weighted cross-entropy + focal Tversky loss).
Evaluation metrics focused on IoU for non-background and disease-only classes.
Visualization of predictions with disease severity percentages.

The notebook is designed to run on platforms like Kaggle, with GPU acceleration.
Dataset
The dataset used is the Apple Tree Leaf Disease Segmentation Dataset (ATLDSD) from Kaggle. It includes:

Images and corresponding semantic masks for 5 classes (background + leaf + 3 disease types).
Total paired samples: ~1641.
Classes: Alternaria leaf spot, Brown spot, Gray spot, Healthy leaf, Rust.

Data is loaded from /kaggle/input/apple-dataset/ATLDSD/, with splits: 70% train, 15% validation, 15% test.
Dependencies
To run the notebook, you'll need:

Python 3.11+
Libraries:
TensorFlow/Keras
NumPy
Pandas
OpenCV (cv2)
Matplotlib
Scikit-learn
tqdm



Install via pip:
pip install tensorflow numpy pandas opencv-python matplotlib scikit-learn tqdm

On Kaggle, these are pre-installed. For local runs, ensure GPU support if available.
Usage

Clone the repository:
git clone https://github.com/your-username/apple-leaf-segmentation.git
cd apple-leaf-segmentation


Download the dataset (if not using Kaggle):

Place it in a directory like data/ATLDSD/ and update BASE_DIR in the notebook.


Run the notebook:

Open semetic-segmentation.ipynb in Jupyter Notebook or JupyterLab.
Execute cells sequentially.
Outputs (models, visualizations, CSV results) are saved to an outputs/ directory.



Key configurations in the notebook:

IMG_SIZE = 256 (input resolution).
NUM_CLASSES = 6 (background, leaf, and 4 disease subclasses).
EPOCHS = 50 (training epochs).
Models are trained and evaluated; best checkpoints are saved.

Models

UNet_MobileNetV2: Pretrained encoder with fine-tuning.
UNet: From-scratch UNet.
DeepLabV3Plus: Lightweight version with ASPP.
FCN: Fully Convolutional Network.
SegNet: Encoder-decoder with upsampling.
ConvAutoencoder: Convolutional autoencoder for segmentation.
FewShotLearning: Few-shot learning approach for segmentation.

Results are compared in outputs/model_comparison.csv and fsl_cae_comparison.csv based on IoU (disease-only) and other metrics.
Model Performance
Below is a performance comparison of the segmentation models evaluated on the dataset:



Model
Val Loss
Val IoU(no-bg)
Val Acc
Test Loss
Test IoU(no-bg)
Test Acc



DeepLabV3Plus
0.2261
0.5441
0.9531
0.2426
0.5105
0.9487


BiSeNetV2
0.2627
0.5082
0.9567
0.2826
0.4656
0.9556


UNet
0.2708
0.4677
0.9360
0.2807
0.4486
0.9366


FCN
0.2865
0.4564
0.9138
0.3033
0.4242
0.9136


SegNet
0.3887
0.3328
0.8765
0.4050
0.3033
0.8767


ConvAutoencoder
0.5791
0.9230
0.1577
0.5984
0.9219
0.1574


FewShotLearning
0.2055
0.2900
0.9425
0.1767
0.2858
0.9440


Analysis

DeepLabV3Plus achieves the highest IoU for non-background classes, indicating strong segmentation performance.
BiSeNetV2 shows the highest validation accuracy, suggesting robust overall classification.
FewShotLearning has the lowest loss but lower IoU, indicating a trade-off between classification accuracy and precise segmentation.
ConvAutoencoder excels in IoU but has poor accuracy, possibly due to overfitting to specific patterns.
SegNet performs the worst across metrics, likely due to its simpler architecture.

Outputs

Trained model checkpoints (*.keras).
Training curves (*_curves.png).
Visualization panels with severity estimates (panel_severity_*.png).
Model comparison tables (outputs/model_comparison.csv, fsl_cae_comparison.csv).

Notes

The notebook includes error handling for GPU memory and safe deserialization.
For custom runs, adjust hyperparameters like batch size (BATCH_SIZE=8) or augmentation probabilities.
If running on non-Kaggle environments, modify paths and ensure no internet-dependent code (e.g., no pip installs inside the notebook).

License
MIT License. Feel free to use and modify.
Acknowledgments

Dataset: ATLDSD from Kaggle.
Inspired by standard semantic segmentation implementations in Keras.
