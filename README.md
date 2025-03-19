**UPDATE: We released gLM2, the latest iteration of gLM, please check it out [here](https://github.com/TattaBio/gLM2)**

**The [manuscript](https://www.biorxiv.org/content/10.1101/2024.08.14.607850v2) describing gLM2 is now accepted at ICLR2025**
# gLM 
## Genomic Language Model
This repository contains the training and inference code for gLM described in preprint: "[Genomic language model predicts protein co-regulation and function](https://www.biorxiv.org/content/10.1101/2023.04.07.536042v3)"

## License
Our model and accompanying scripts are distributed for academic and non-commercial use only. Please refer to the LICENSE attached to this repo and reach out if you have any questions.  

© President and Fellows of Harvard College 2023.


## Set up python environment 
#### using conda
```
conda env create -f environment.yml python==3.10.8
conda activate glm-env
pip install torch==1.12.1+cu116  torchvision==0.13.1+cu116 -f https://download.pytorch.org/whl/torch_stable.html
```
This set up was tested using python 3.10.8

## Download model 
The latest checkpoint of our model used for the preprint is available for download from https://zenodo.org/record/7855545
```
mkdir model 
cd model 
wget https://zenodo.org/record/7855545/files/glm.bin
```

## Compute gLM embeddings 
gLM embeddings can be computed using the following steps:
#### 1. Prepare two input files.

a) FASTA file of your proteins (amino acid sequences) in your contig
```
>prot_1
MNYSHDNWSAILAHIGKPEELDTSARNAGALTRRREIRDAATLLRLGLAYGPGGMSLREVTAWAQLHDVA
TLSDVALLKRLRNAADWFGILAAQTLAVRAAVTGCTSGKRLRLVDGTAISAPGGGSAEWRLHMGYDPHTC
QFTDFELTDSRDAERLDRFAQTADEIRIADRGFGSRPECIRSLAFGEADYIVRVHWRGLRWLTAEGMRFD
MMGFLRGLDCGKNGETTVMIGNSGNKKAGAPFPARLIAVSLPPEKALISKTRLLSENRRKGRVVQAETLE
AAGHVLLLTSLPEDEYSAEQVADCYRLRWQIELAFKRLKSLLHLDALRAKEPELAKAWIFANLLAAFLID
DIIQPSLDFPPRSAGSEKKN
>prot_2
MAKQDYYEILGVSKTAEEREIRKAYKRLAMKYHPDRNQGDKEAEAKFKEIKEAYEVLTDSQKRAAYDQYG
HAAFEQGGMGGGGFGGGADFSDIFGDVFGDIFGGGRGRQRAARGADLRYNMELTLEEAVRGVTKEIRIPT
LEECDVCHGSGAKPGTQPQTCPTCHGSGQVQMRQGFFAVQQTCPHCQGRGTLIKDPCNKCHGHGRVERSK
TLSVKIPAGVDTGDRIRLAGEGEAGEHGAPAGDLYVQVQVKQHPIFEREGNNLYCEVPINFAMAALGGEI
EVPTLDGRVKLKVPGETQTGKLFRMRGKGVKSVRGGAQGDLLCRVVVETPVGLNERQKQLLQELQESFGG
PTGEHNSPRSKSFFDGVKKFFDDLTR
````

b) subcontig to protein mapping with orientation
in the following format. 

Where '-' refers to reverse direction and '+' refers to forward direction relative to the rest of the contig. 

Make sure the number of proteins in subcontigs does not exceed max_seq_length = 30. 
```
contig_0  +prot_1;-prot_2;-prot_3;-prot_4;-prot_5;+prot_6;-prot_7;+prot_8;+prot_9;+prot_10;-prot_11;-prot_12;-prot_13;-prot_14;-prot_15;-prot_16;
contig_1  +prot_17;-prot_18;-prot_19;-prot_20;-prot_21;+prot_22;-prot_23;+prot_24;+prot_25;+prot_26;-prot_27;
```
see contig_to_prots.tsv and test.fa in example_data as an example.

#### 2. compute pLM embeddings. 
We use [esm2](https://github.com/facebookresearch/esm) to embed proteins. 
```
cd data
python plm_embed.py example_data/inference_example/test.fa example_data/inference_example/test.esm.embs.pkl
```
we provide the expected output example_data/inference_example/test.esm.embs.pkl for your reference and on a A100 GPU this test example took less than 2 minutes to complete. 
#### 3. batch your data for gLM inference. 
```
cd data
# make output directory
mkdir batched_data  
python batch_data.py example_data/inference_example/test.esm.embs.pkl example_data/inference_example/contig_to_prots.tsv example_data/batched_data
```
The output data directory (batched_data) now contains two files. The output directory (batched_data) which contains batch.pkl and prot_index_dict.pkl files. The former is the input containing your data input embeddings, and the latter contains the dictionary mapping from protein index to protein ID.

we provide the expected output data/example_data/batched_data/ for your reference and this particular test example took us less than 1 minutes to run. 


#### 4. compute gLM embeddings.
```
cd data
python ../gLM/glm_embed.py -d example_data/batched_data -m ../model/glm.bin -b 100 -o test_results
```
If you come across GPU memory errors, try reducing batch size (-b).

gLM embeddings will be saved as *.glm.embs.pkl file in the output directory. 

You can output all inference results (plm_embs/glm_embs/prot_ids/outputs/output_probabilitess) by adding --all_results/-a flag. This will be saved as a *.results.pkl file in the output directory. 

You can also output attention matrices by adding --attention flag. Attentions will be saved for post processing in your ourput directory *.attention.pkl

We provide the expected output in data/test_results/results/batch.pkl.glm.embs.pkl and the expected runtime for this on A100 is ~2 minutes. 

We are working on making inference code available as a colab notebook. so **stay tuned**. 
## Train your own gLM on your custom dataset
We provide the training script for gLM for your custom dataset. 
```
cd data
python ../gLM/train.py -d example_data/training_example -o test_train_dir
```
The data directory (data/example_data/training_example) contains batched training data which can be generated using batch_data.py (see sections 1-3 in "Compute gLM embeddings" above). Make sure pkl files containing training data starts with "train" and pkl files containing eval data starts with "eval". 
For example: 
```
ls example_data/training_example
eval.0.PC_100.pkl  train.0.PC_100.pkl  train.1.PC_100.pkl
```
python train.py -h will show many hyperparameter flags that can be tweaked to suit your training. 

Training log file, checkpoints and pretrained models are stored in the output directory. 

Note: When there are checkpoints already saved in the specified output directory, the script will automatically load the latest checkpoint and continue training from there. 

## Visualization
We included scripts used for downstream analyses and visualizations (e.g. EC number analysis and operon prediction) in gLM directory. 

## Citations
If you find gLM useful in your work, please cite our paper:

Hwang, Y. Cornman, A. Kellogg, E. Ovchinnikov, S. and Girguis, P. (2023) "[Genomic language model predicts protein co-regulation and function](https://www.biorxiv.org/content/10.1101/2023.04.07.536042v3)", BioRxiv

```bibtex
@article {Hwang2023.04.07.536042,
	author = {Yunha Hwang and Andre L. Cornman and Elizabeth H. Kellogg and Sergey Ovchinnikov and Peter R. Girguis},
	title = {Genomic language model predicts protein co-regulation and function},
	elocation-id = {2023.04.07.536042},
	year = {2023},
	doi = {10.1101/2023.04.07.536042},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2023/10/15/2023.04.07.536042},
	eprint = {https://www.biorxiv.org/content/early/2023/10/15/2023.04.07.536042.full.pdf},
	journal = {bioRxiv}
}
```

