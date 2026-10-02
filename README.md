# AlphaFold Latent Flooding: Zero-shot Generation of Protein Conformational Ensembles

Implementation of the Latent Flooding for [AlphaFold2 v2.3.2](https://github.com/google-deepmind/alphafold).

**[Zilin Song](https://github.com/ZL-Song)** (song.zilin@outlook.com)  
**[Runtong Qian](https://github.com/RitaQian-westlake)** (qianruntong@westlake.edu.cn)  

## Installation

For a non-Docker installation from scratch:
```bash
conda create -n aftools python=3.9
conda activate aftools
conda install -c conda-forge openmm=7.5.1 pdbfixer
pip install --no-cache-dir -r requirements.txt "jax[cuda12_pip]==0.4.18" optax==0.1.7 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

Note that `requirements.txt` must: 
1. pin `dm-haiku==0.0.10` (0.0.9 is incompatible with jax >= 0.4.14 and fails with `AttributeError: module 'jax' has no attribute 'xla'`);
2. the `numpy<2` and `nvidia-cudnn-cu12<9` bounds are required because the 2023 dependency metadata otherwise resolves to binary-incompatible NumPy 2.x and cuDNN 9.x.

<!-- **Option b. JAX 0.3.25 with CUDA 11**

On machines with an old glibc/GCC toolchain, where the JAX 0.4.18 wheels may report GCC runtime errors.

First replace the requirements:
```
# requirement.txt
absl-py==1.0.0
biopython==1.79
dm-haiku==0.0.9
dm-tree==0.1.6
docker==5.0.0
immutabledict==2.0.0
ml-collections==0.1.0
numpy==1.21.6
pandas==1.3.4
scipy==1.7.0
tensorflow-cpu==2.11.0
```

and do:
```bash
conda create -n aftools python=3.9
conda activate aftools
conda install -c conda-forge openmm=7.5.1 pdbfixer
pip install --no-cache-dir -r requirements.txt jax==0.3.25 jaxlib==0.3.25+cuda11.cudnn805 optax==0.1.7 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html.
```
You may need `module load cuda/11.x` to locate the CUDA dependencies. -->

For unit tests, install the following:  
```bash
conda install parameterized
```

## Examples
**UBQ**
```bash
cd ./results_layer0/test0_ubq
python run_chkp0_ubq.py   # runs checkpoint modeling.
python run_test0_ubq.py   # runs inference modeling.
```

**AdK**
```bash
cd ./results_layer0/test1_adk
python run_chkp1_adk.py   # runs checkpoint modeling.
python run_test1_adk.py   # runs inference modeling.
```

**TEM-1**
```bash
cd ./results_layer0/test2_tem1
python run_chkp2_tem1.py   # runs checkpoint modeling.
python run_test2_tem1.py   # runs inference modeling.
```

## File descriptions
- `./afmisc/`: miscellaneous files from the AlphaFold repository.
- `./aftools/`: the AFLF algorithm and helper functions.
- `./alphafold/`: the AlphaFold source code.
- `./results_ablation/`: working directory for the ablation results reported in this work.
- `./results_benchmark/`: working directory for the benchmark results reported in this work.
- `./results_bfactors/`: working directory for the B-factor results reported in this work.
- `./results_layer0/`: working directory for the representative results reported in this work.
- `./featurizer.*`: scripts that execute the native AlphaFold featurization pipeline and generate `features.pkl`.
- `./LICENSE`: the AlphaFold license.

## Metadata

The simulation trajectories generated in this study will be deposited in a public data repository upon finalization.

## Reference

If you find this code useful in your research, we would appreciate it if you cite:

> R. Qian, R. Zhan, Z. Song, and J. Huang, "Zero-Shot Generation of Protein Conformational Ensembles Through AlphaFold Latent Flooding," *bioRxiv* (2026). [doi:10.64898/2026.04.16.718914](https://doi.org/10.64898/2026.04.16.718914)

```latex
@article{qian2026aflf,
  title={Zero-Shot Generation of Protein Conformational Ensembles Through AlphaFold Latent Flooding},
  author={Qian, Runtong and Zhan, Rui and Song, Zilin and Huang, Jing},
  journal={bioRxiv},
  pages={2026--04},
  year={2026},
  publisher={Cold Spring Harbor Laboratory},
  doi={https://doi.org/10.64898/2026.04.16.718914},
}
```