# MRI Recovery with Self-calibrated Denoisers without Fully-sampled Data (ReSiDe)
Required packages can be installed using:
```python
pip install -r requirements.txt
```
# Data
"For static data, you can find the brain MRI data in the 'Brain' folder, which contains T1 and T2 weighted images to run the demo for ReSiDe-S and ReSiDe-M.
For dynamic data, you can download the MRXCAT digital perfusion phantom data from [https://figshare.com/s/57f6689ec5d89e608a11] and place it into the 'Perfusion/MRXCAT/data' directory to run the demo for ReSiDe-S and ReSiDe-M."
# ReSiDe-S
To run ReSiDe-S, execute the following command:
```python
python ReSiDe-S.py
```
# ReSiDe-M
For training Brain data (2D), run:
```python
python ReSiDe_M_brain_training .py
```
For inference with Brain data (2D), run:
```python
python ReSiDe_M_brain_inference .py
```
For training MRXCAT data (3D), run:
```python
python ReSiDe_M_mrxcat_training.py
```
For inference with MRXCAT data (3D), run:
```python
python ReSiDe_M_mrxcat_inference.py
```
