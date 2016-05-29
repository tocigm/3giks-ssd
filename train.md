## Data preparation
Data is formatted with PASCAL_VOC format
Data should be put in $HOME/data
Structure:

```
$HOME/data/VOCdevkit/VOC_custom
	├── Annotations  #annotation PASC_VOC format, please see IV to convert form MIT_CSAIL format
	├── ImageSets
	└── Main
		├── test.txt
		└── trainval.txt
	└──  JPEGImages
```

Test.txt, and train.txt is files to list images without extension.
```
can_coffer_bing0491
can_coffer_bing1056
can_coffer_bing0007
can_coffer_bing0630
...
```
- Create datasets file: Edit some file in $PROJECT/data/VOC_custom

```
.
├── create_data.sh
├── create_list.sh
└── labelmap_voc.prototxt
```

- We define class - label mapping in labelmap_voc.prototxt, data path in create_list.sh and create_data.sh
- Example labelmap_voc.prototxt:

```
item {
  name: "none_of_the_above"
  label: 0
  display_name: "background"
}
item {
  name: "can"
  label: 1
  display_name: "cancoffe"
}
```

- create_list.sh: (edit 13th line to your dataset name)

```	
for name in VOC_custom

```

- Create_data.sh: (edit 8th line to you dataset name)

```
dataset_name="VOC_custom"

```

Run create_list.sh and create_data.sh in order to create dataset file and lmdb database.


## Training

- Edit examples/ssd/ssd_pascal.py
- Some importance line we need to change

```
# The database file for training data. Created by data/VOC10/create_data.sh
train_data = "examples/VOC10/VOC10_trainval_lmdb"
# The database file for testing data. Created by data/VOC10/create_data.sh
test_data = "examples/VOC10/VOC10_test_lmdb"
# Specify the batch sampler.
resize_width = 300
resize_height = 300

model_name = "VGG_VOC_custom_{}".format(job_name)

# Directory which stores the model .prototxt file.
save_dir = "models/VGGNet/VOC_custom/{}".format(job_name)
# Directory which stores the snapshot of models.
snapshot_dir = "models/VGGNet/VOC_custom/{}".format(job_name)
# Directory which stores the job script and log file.
job_dir = "jobs/VGGNet/VOC_custom/{}".format(job_name)
# Directory which stores the detection results.
output_result_dir = "{}/data/VOCdevkit/results/VOC_custom/{}/Main".format(os.environ['HOME'], job_name)

# model definition files.
train_net_file = "{}/train.prototxt".format(save_dir)
test_net_file = "{}/test.prototxt".format(save_dir)
deploy_net_file = "{}/deploy.prototxt".format(save_dir)
solver_file = "{}/solver.prototxt".format(save_dir)
# snapshot prefix.
snapshot_prefix = "{}/{}".format(snapshot_dir, model_name)
# job script path.
job_file = "{}/{}.sh".format(job_dir, model_name)

# Stores the test image names and sizes. Created by data/VOC10/create_list.sh
name_size_file = "data/VOC_custom_cancoffe/test_name_size.txt"
# The pretrained model. We use the Fully convolutional reduced (atrous) VGGNet.
pretrain_model = "models/VGGNet/VGG_ILSVRC_16_layers_fc_reduced.caffemodel"
# Stores LabelMapItem.
label_map_file = "data/VOC_custom_cancoffe/labelmap_voc.prototxt"
num_classes = 2
num_test_image = 100

```

- When we run examples/ssd/ssd_pascal.py, it will generate train.prototxt, test.prototxt, deploy.prototxt, solver.prototxt and and do training automatically.


## Demo

- We should edit something in demo.py like below :

```
voc_labelmap_file = "data/VOC10/labelmap_voc.prototxt"
model_def = 'models/VGGNet/VOC10/SSD_300x300/deploy.prototxt'
model_weights = 'models/VGGNet/VOC10/SSD_300x300/model.caffemodel'
```

- Run:

```
python demo.py image_input_path
```

## Converting annotation

If we have MIT_CSAIL annotation files, we should to convert to PASC_VOC format.
Script to convert is scripts/convert_mit2pasc.py

```
python scripts/convert_mit2pasc.py list_annotation out_dir number_threads
```
- Example: 
```
python scripts/convert_mit2pasc.py VOC_custom/list_ann.txt VOC_custom/Annotations/ 5
```
	
