# ProToken: Differentiable Protein Engineering through Diffusion Transformer over Neural Amino Acids
This is the github repo for the paper *Differentiable Protein Engineering through using Diffusion Transformer over ProTokens*. An early version is preprinted at [biorxiv](https://doi.org/10.1101/2023.11.27.568722).

<p align="center"><img src="https://github.com/issacAzazel/ProToken/blob/main/figs/model_arch.jpg" width="100%"></p>

**ProTokens**, developed by the Gao Group at Peking University, are machine-learned "amino acids" derived from protein structure databases via self-supervised learning, providing a compact yet informative representation that bridges "1D" (sequence) and "3D" (structure) modalities of proteins. Building on **ProTokens**, we develop **PT-DiT**, a [diffusion transformer](https://www.wpeebles.com/DiT.html) that jointly models protein sequences and structures through a generative objective. **ProTokens** and **PT-DiT** enable efficient encoding of 3D folds, contextual protein design, sampling of metastable states, and directed evolution for diverse objectives. 

<p align="center"><img src="https://github.com/issacAzazel/ProToken/blob/main/figs/app.jpeg" width="100%"></p>

## Installation 
Running example scripts in [example_scripts](./example_scripts) requires:
### Main dependencies
* python==3.10
* jax==0.4.28, jaxlib==0.4.28
* flax==0.8.3
* ml-collections==0.1.1
* dm-tree==0.1.8
* dm-haiku==0.0.12
### Other dependencies
* diffrax==0.6.0
* scikit-learn==1.5.1
* biopandas==0.5.1
* biopython==1.84

In theory, any environment compatible with the packages mentioned above should run successfully. We also provide [requirements.txt](requirements.txt) which can directly create a compatible environment via `pip install -r requirements.txt` (with possible redundant dependencies). Our configuration includes Ubuntu 22.04 (GNU/Linux x86_64), NVIDIA A100-SXM4-80GB, CUDA 12.2 and Anaconda 23.7.2. The complete notebook execution takes approximately 0.5 hours.

### Remark
We implement [FlashAttention](https://github.com/Dao-AILab/flash-attention) to accelerate PT-DiT inference. Note that protein sequences must be padded such that their length (`NRES`) becomes a multiple of 128 for compatibility. Users may disable this feature by setting `flash_attention_flag = False` in the configuration file ([dit_config.py](configs/dit_config.py)).

## Exploring ProTokens's capabilities!
Before running scripts, download pre-trained checkpoints of **ProToken** encoder/decoder and **PT-DiT** in [here](https://drive.google.com/drive/folders/1sO_FQh1AF4iEQnV2W4_Px29QVA0WN7QM), and move them into [ckpts](ckpts).

### Highly Compact and Informative Structure Representation with ProTokens
* Compactness: **ProTokens** represent protein structures as machine-learned "amino acids", a structure-aware "amino acid" (``vocabulary size=512``) is assigned to each residue of proteins with ProToken encoder. 
* Informativeness:  Compared with natural amino acid (``vocabulary size=20``), **ProTokens** can be faithfully "(re)fold" into corresponding structures. Furthermore, **ProTokens** can distinguish between different conformations, which are considered as degenerate in sequence-based representations.

<p align="center"><img src="https://github.com/issacAzazel/ProToken/blob/main/figs/recon.png" width="100%"></p>

[``example_scripts/encode_decode_structure.sh``](example_scripts/encode_decode_structure.sh) and [``example_scripts/encode_decode_structure_batch.sh``](example_scripts/encode_decode_structure_batch.sh) illustrate how to encode structure (in .pdb format) or structures (a directory of pdb files) into **ProTokens**, and decode them back into reconstructed structure(s). The resulting ProTokens are saved in `vq_code_indexes.pkl` or `vq_code_indexes_dict.pkl` file. 

For applications requiring continuous *embeddings* rather than discrete *tokens*, all ProToken embeddings (32-dimensional vectors) are archived in [``protoken_emb.pkl``](embeddings/protoken_emb.pkl) as a 512×32 array. Structure embeddings can be retrieved by array indexing with the corresponding ProToken identifiers.

We also provide a [Colab notebook](https://colab.research.google.com/drive/15bBbfa7WigruoME089cSfE242K1MvRGz) for ProToken encoding and decoding process. 

### Protein Engineering via Diffusion Transformer over Unified Sequence and Structure Representations
**ProTokens** provide a unified perspective on protein sequences and structures. **PT-DiT**, a [diffusion transformer](https://www.wpeebles.com/DiT.html) that models the joint probability of protein sequences and structures, can co-generate diverse proteins in the form of compatible sequence/structure pairs. 

<p align="center"><img src="https://github.com/issacAzazel/ProToken/blob/main/figs/co_generation.jpg" width="75%"></p>

Furthermore, through [RePaint](https://github.com/andreas128/RePaint) algorithm, **PT-DiT** can "infill" sequences/structures with given context. Therefore, pre-trained **PT-DiT** becomes a zero-shot toolkit for various tasks in protein engineering, such as (contextual) inverse folding, functional site scaffolding, etc.

<p align="center"><img src="https://github.com/issacAzazel/ProToken/blob/main/figs/inpaint.jpg" width="100%"></p>

[``example_scripts/de_novo_design.ipynb``](example_scripts/de_novo_design.ipynb) and [``example_scripts/repaint.ipynb``](example_scripts/repaint.ipynb) illustrate how to generate and edit proteins with **PT-DiT**. 

### Manipulating Proteins in the Latent Space of PT-DiT

Using probability flow ordinary differential equation formulated in diffusion models. Pre-training of **PT-DiT** yields a compact and organized latent Gaussian space (which is *sequence-aware* and *structure-aware*) of proteins. By virtue of its compactness and informativeness, this latent space is suitable for manipulating and evolving proteins: 

* Interpolation between two functional conformations reveals intermediate states in protein dynamics. Conjugately, Interpolation between two remote structural homologs (re)discovers evolutionary and novel proteins.

<p align="center"><img src="https://github.com/issacAzazel/ProToken/blob/main/figs/interpolation.png" width="100%"></p>

* Few-shot activate learning of activity profile improves the efficiency of directed evolution. 

<p align="center"><img src="https://github.com/issacAzazel/ProToken/blob/main/figs/evo.png" width="75%"></p>

[``example_scripts/latent_space.ipynb``](example_scripts/latent_space.ipynb) illustrates how proteins can be interpolated and evolved in the latent space of **PT-DiT**.

...

## Citation
```python
@article{lin2023tokenizing,
    title={Tokenizing Foldable Protein Structures with Machine-Learned Artificial Amino-Acid Vocabulary},
    author={Lin, Xiaohan and Chen, Zhenyu and Li, Yanheng and Ma, Zicheng and Fan, Chuanliu and Cao, Ziqiang and Feng, Shihao and Gao, Yi Qin and Zhang, Jun},
    journal={bioRxiv},
    pages={2023--11},
    year={2023},
    publisher={Cold Spring Harbor Laboratory}
}
```

## Contact
For questions or further information, please contact [fengsh@cpl.ac.cn](fengsh@cpl.ac.cn) or [jzhang@cpl.ac.cn](jzhang@cpl.ac.cn).