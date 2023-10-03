# MRI Recovery with Self-calibrated Denoisers without Fully-sampled Data (ReSiDe)
Required packages can be installed using:
```python
pip install -r requirements.txt
```
# ReSiDe-S
To run ReSiDe-S, execute the following command:
```python
python ReSiDe-S.py
```
# ReSiDe-M
For training Brain data (2D), run:
```python
ReSiDe_M_brain_training .py
```
For inference with Brain data (2D), run:
```python
ReSiDe_M_brain_inference .py
```
For training MRXCAT data (3D), run:
```python
python ReSiDe_M_mrxcat_training.py
```
For inference with MRXCAT data (3D), run:
```python
python ReSiDe_M_mrxcat_inference.py
```
