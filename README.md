# Final Solution for KDDCUP20201-MAG240M

## Results on MAG240M
Here, we demonstrate the following performance on the OGB-MAG240M dataset from KDDCUP 2021.

| Model          |Test Acc    |Validation Acc  | Parameters    | Hardware |
| -------------- |--------------- | ----------------- | -------------- |----------|
|  Our Model     | 0.7447 | 0.7669 ± 0.0003 (ensemble 0.7696) | 743,449 | Tesla V100 (21GB) |

## Reproducing results

### 0. Requirements

Here just list python3 packages we used in this competition:

```
numpy==1.19.2
torch==1.5.1+cu101
dgl-cu101==0.6.0.post1
ogb==1.3.1
sklearn==0.23.2
tqdm==4.46.1
```

###  1. Prepare Graph and Features

The preprocess code modifed from [dgl baseline](https://github.com/dmlc/dgl/tree/master/examples/pytorch/ogb_lsc/MAG240M).  We created graph with 6 different edge types instead of 5.

```
# Time cost: 3hours,30mins

python3 $MAG_CODE_PATH/preprocess.py
        --rootdir $MAG_INPUT_PATH \
        --author-output-path $MAG_PREP_PATH/author.npy \
        --inst-output-path $MAG_PREP_PATH/inst.npy \
        --graph-output-path $MAG_PREP_PATH \
        --graph-as-homogeneous \
        --full-output-path $MAG_PREP_PATH/full_feat.npy
```
The graphs and features will be saved in `MAG_PREP_PATH` , where the  `MAG_PREP_PATH` is specified in `run.sh`.

#### Calculate features

The meta-path based features are generated by this script. Details can be found in our technical report.

```
# Time cost: 2hours,20mins (only generate label related features)

python3 $MAG_CODE_PATH/feature.py
        $MAG_INPUT_PATH \
        $MAG_PREP_PATH/dgl_graph_full_heterogeneous_csr.bin \
        $MAG_FEAT_PATH \
        --seed=42
```

#### Train RGAT model and prepare RGAT features

The RGAT model is modifed from [dgl baseline](https://github.com/dmlc/dgl/tree/master/examples/pytorch/ogb_lsc/MAG240M).  The validation accuracy is  0.701 , as same as described in the dgl baseline github.

```
# Time cost: 33hours,40mins (20mins for each epoch)

python3 $MAG_CODE_PATH/rgat.py
        --rootdir $MAG_INPUT_PATH \
        --graph-path $MAG_PREP_PATH/dgl_graph_full_homogeneous_csc.bin \
        --full-feature-path $MAG_PREP_PATH/full_feat.npy \
        --output-path $MAG_RGAT_PATH/ \
        --epochs=100 \
        --model-path $MAG_RGAT_PATH/model.pt \
        --submission-path $MAG_RGAT_PATH/
```
You will get embeddings as input features of the following MPLP models.

### 2. Train MPLP models
We add weight for each class to  allievate the imblance problem by using the function below:

$$
weight = 153 \times normalise(log_{10}(\frac{cnt_{2018}}{cnt_{<=2018}} + 5.))
$$

The train process splits to two step:

1. train the model with full train samples at a large learning rate (here we use *StepLR(lr=0.01, step_size=100, gamma=0.25)*)
2. then fine tune the model with latest train samples (eg, paper with year >= 2018) with a small learning rate (0.000625)

You can train the MPLP model by running the following commands:

```
# Time cost: 2hours,40mins for each seed

for seed in $(seq 0 7);
do
    python3 $MAG_CODE_PATH/mplp.py \
            $MAG_INPUT_PATH \
            $MAG_MPLP_PATH/data/ \
            $MAG_MPLP_PATH/output/seed${seed} \
            --gpu \
            --seed=${seed} \
            --batch_size=10240 \
            --epochs=200 \
            --num_layers=2 \
            --learning_rate=0.01 \
            --dropout=0.5 \
            --num_splits=5
done
```

### 3. Ensemble MPLP results

While having all the results with k-fold cross validation training under 8 different seeds, you can average the results by running code below:

```
python3 $MAG_CODE_PATH/ensemble.py $MAG_MPLP_PATH/output/ $MAG_SUBM_PATH
```
