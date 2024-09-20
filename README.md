Dependencies
```
python -m pip install --no-deps opencv-python scikit-image lazy_loader simplejson numba llvmlite czifile tifffile roifile
python -m pip install pandas
```

To plot graphs
```
sudo apt install libgraphviz-dev -y
python -m pip install --no-deps pygraphviz
```

Data
```
wget -q -O data.zip https://figshare.com/ndownloader/files/42287322
unzip -q data.zip
```

```
cp -r SourceData/1-cell_image_library/dataset/ 1-cil
cd 1-cil
PYTHONPATH=.. python train_model.py
```

```
cp -r SourceData/2-melanoma_nuclei/dataset 2-melanoma_nuclei/
cd 2-melanoma_nuclei
PYTHONPATH=.. python train.py . dataset 0 MCW
```
