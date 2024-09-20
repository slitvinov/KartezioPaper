```
wget -q -O data.zip https://figshare.com/ndownloader/files/42287322
unzip data.zip
cp -r SourceData/1-cell_image_library/dataset/ 1-cil
```

```
cd 1-cil
PYTHONPATH=.. python train_model.py
```