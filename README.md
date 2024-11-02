This is the landing page for Sketch Stuctured Linear code for  [Accelerating Inference with Fast and Expressive Sketch Structured Transform](https://openreview.net/forum?id=nrgyOGU7ZP&referrer=%5BAuthor%20Console%5D(%2Fgroup%3Fid%3DNeurIPS.cc%2F2024%2FConference%2FAuthors%23your-submissions))

# Sketch Structured Transform parameter sharing scheme
![alt text](https://github.com/apd10/Sketch-Structured-Linear/blob/main/SSL1.png)


# How to use for your own models.

### Installing the kernel

```
git clone https://github.com/kimiasa/SSLinear/tree/clean  # note use the clean branch
# follow instructions on the github to install the kernel and sample usage
```

### Convert a standard model linear layers to use SS1


SS1 layers can be simply created by the following and used in standar model building.

```
from sketch_structured_linear.SSL import SSL
layer = SSL(in_dim, out_dim, redn_factor=red_fac, seed=seed, bias=True)
```

We provide utility for creating a SS1 model out of standard transformer models for easier usage. The recipe to generate SS1 model is to first create a standard model and then use the convert_to_ss_linear method to obtain the corresponding SS1 model.
```
def convert_to_ss_linear(
    model,
    reduction_factor: int,
    layer_indices: Optional[List[int]] = None,  # if you only want to keep certain layers
    skip_attention: Optional[bool] = False, # if you want to skip the attention matrices (K,Q,V)
    init_seed: Optional[int] = 42,
    skip_pattern: Optional[List[str]] = None # pattern based skipping.
)
```


An example would be:
```
from sketch_structured_linear.SSLProjection import convert_to_ss_linear
#model =  some pretrained / scratch model
model = convert_to_ss_linear(
    model,
    reduction_factor=8,
    layer_indices = [1,2,3,4,5],
    skip_attention=False,
    init_seed=42,
    skip_pattern=['pooler', 'embeddings'],
)
```
During conversion, block sizes for each layer are adjusted to optimize efficiency with respect to GPU capabilities and model dimensions. So expect model conversion to be slow. If you know what block sizes to use , you can skip autotuning and set the block sizes after initializing layer.

### Projecting a pre-trained model onto SS1


# Reproducing results in paper.

### GPT2 end-to-end training & inference latency benchmarking
Ensure that [SS1 Kernel](https://github.com/kimiasa/SSLinear/tree/clean) is properly installed. <br /><br />
We build on the top of Hazy Research's [fly](https://github.com/HazyResearch/fly) repository. Please follow the instructions at the following repostory,branch:

To run the GPT2 end-to-end training, check out the train branch at [Experiments](https://github.com/kimiasa/Experiments/tree/train) repo.
```
git clone https://github.com/kimiasa/Experiments/tree/train  # use train branch
```
For Inference latency benchmarking:
```
git clone https://github.com/kimiasa/Experiments/tree/inference  # use inference branch
```

### Bert fine-tuning
Ensure that [SS1 Kernel](https://github.com/kimiasa/SSLinear/tree/clean) is properly installed. <br /><br />
To project BERT models onto SS1 layers, use the following code:
```python
from transformers import AutoModel
from sketch_structured_linear.SSLProjection import convert_to_ss_linear

# For BERT-large
model = AutoModel.from_pretrained("bert-large-uncased")
# These layers were selected using GLUE's RTE task - adjust for your dataset
layer_indices = [1, 6, 8, 9, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
model = convert_to_ss_linear(
    model,
    reduction_factor=8,
    layer_indices=layer_indices,
    skip_attention=False,
    init_seed=42,
    skip_pattern=['pooler', 'embeddings']
)

# For BERT-base
# Example using GLUE-optimized layer selection
model = AutoModel.from_pretrained("bert-base-uncased")
# These layers were selected using GLUE's RTE task - adjust for your dataset
layer_indices = [1, 7, 8, 9, 10, 11, 12]
model = convert_to_ss_linear(
    model,
    reduction_factor=8,
    layer_indices=layer_indices,
    skip_attention=False,
    init_seed=42,
    skip_pattern=['pooler', 'embeddings']
)
```

#### Layer Selection Guide
To select optimal layers for your specific dataset:

- Use a small validation set from your target task as calibration data
- Measure performance impact by compressing one layer at a time
- Select layers that show minimal performance degradation when compressed
- Validate the selected configuration on your full dataset

Adjust layer_indices and reduction_factor based on your desired compression/performance trade-off. The model can then be fine-tuned using standard HuggingFace training methods.

You can also reproduce our experiments using the following command: 
```
export TASK_NAME=mrpc
python experiments/run_glue_no_trainer.py \
    --model_name_or_path google-bert/bert-base-cased \
    --task_name $TASK_NAME \
    --max_length 128 \
    --per_device_train_batch_size 32 \
    --learning_rate 2e-5 \
    --num_train_epochs 3 \
    --output_dir /tmp/$TASK_NAME/ \
    --reduction_factor 8 \
    --layers_to_skip  1,7,8,9,10,11,12 \
    --skip_attention False
```

# Correspondence
If you need help with your own work using the repository, it would be best to email apdesai@berkeley.edu AND adirid.7090@gmail.com 


# Previous work . If you find our work useful. Please cite the following:
```
@inproceedings{ss1,
 author = {Desai, Aditya and Saedi, Kimia and Walia Apoorv, and Lee, Jihyeong and Zhou, Keren and Shrivastava Anshumali},
 booktitle = {Advances in Neural Information Processing Systems},
 title = {{Accelerating Inference with Fast and Expressive Sketch Structured Transform}},
 year = {2024}
}

@inproceedings{MLSYS2022_1eb34d66,
 author = {Desai, Aditya and Chou, Li and Shrivastava, Anshumali},
 booktitle = {Proceedings of Machine Learning and Systems},
 editor = {D. Marculescu and Y. Chi and C. Wu},
 pages = {762--778},
 title = {{R}andom {O}ffset {B}lock {E}mbedding ({ROBE}) for compressed embedding tables in deep learning recommendation systems},
 volume = {4},
 year = {2022},
 award = {Outstanding Paper Award},
 note = {}
}


@inproceedings{NEURIPS2022_dbae9151,
 author = {Desai, Aditya and Shrivastava, Anshumali},
 booktitle = {Advances in Neural Information Processing Systems},
 editor = {S. Koyejo and S. Mohamed and A. Agarwal and D. Belgrave and K. Cho and A. Oh},
 pages = {33961--33972},
 publisher = {Curran Associates, Inc.},
 title = {{T}he trade-offs of model size in large recommendation models: {100GB} to {10MB} {Criteo-tb DLRM} model},
 url = {https://proceedings.neurips.cc/paper_files/paper/2022/file/dbae915128892556134f1c5375855590-Paper-Conference.pdf},
 volume = {35},
 year = {2022}
}


@inproceedings{desai2023hardware,
  title={{H}ardware-{A}ware {C}ompression with {R}andom {O}peration {A}ccess {S}pecific {T}ile ({ROAST}) {H}ashing},
  author={Desai, Aditya and Zhou, Keren and Shrivastava, Anshumali},
  booktitle={International Conference on Machine Learning},
  pages={7732--7749},
  year={2023},
  organization={PMLR}
}


@inproceedings{
stablerps,
title={In defense of parameter sharing for model compression},
author={Desai, Aditya and Shrivastava, Anshumali},
booktitle={The Twelfth International Conference on Learning Representations },
year={2024},
}

```
