All the data is preprocessed and stored in https://drive.google.com/drive/folders/1ymwB5XxtvUMENn8eetn-1mYk9ORitOFL?usp=sharing. Please download and store this folder in the root directory. 

Getting trajectory via motion model - 

```
python3 main.py --dataset 20 --motion_model
```


Getting refined trajectory via ICP - 

```
python3 main.py --dataset 20 --icp_refinement
```

Getting refined trajectory via GTSAM - 

```
python3 main.py --dataset 20 --gtsam_refinement
```

Getting occupancy map - 

```
python3 main.py --dataset 20 --occupancy_grid
```

Getting texture map - 

```
python3 main.py --dataset 20 --create_texture_map
```

please set the correct trajectory path in the main.py file. All the output files are stores in data/<dataset> folder.