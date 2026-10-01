# AlphaFold Latent Flooding: Zero-shot Generation of Protein Conformational Ensemble

Implementation of the Latent Flooding for [AlphaFold2 v2.3.2](https://github.com/google-deepmind/alphafold).

**[Zilin Song](https://github.com/ZL-Song)** (song.zilin@outlook.com)  
**[Runtong Qian](https://github.com/RitaQian-westlake)** (qianruntong@westlake.edu.cn)  

## Installation
If installing from scratch, the following provides a non-Docker approach:
```bash
conda create -n aftools python=3.9
conda activate aftools
conda install -c conda-forge openmm=7.5.1 pdbfixer
pip install --no-cache-dir -r requirements.txt "jax[cuda12_pip]==0.4.18" optax==0.1.7 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

<!-- 
### Option 2. JAX 0.3.25 with CUDA 11
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
You may need `module load cuda/11.x` to run with jax. -->


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

`run_prep*.py` additionally requires `MDAnalysis` and generates the segmentation used by AFLF.

## File descriptions
`./afmisc/` Miscellaneous files from AlphaFold repository;  
`./aftools/` Implements the AFLF algorithm and helper functions;  
`./alphafold/` The AlphaFold source code;  
`./results_ablation/` The working directory for the ablation results reported in this work;  
`./results_benchmark/` The working directory for the benchmark results reported in this work;  
`./results_bfactors/` The working directory for the B-factor results reported in this work;  
`./results_layer0/` The working directory for the representative results reported in this work;  
`./featurizer.*` Customized scripts that execute the native AlphaFold featurization pipeline and generate `features.pkl`;  
`./LICENSE` The AlphaFold license.

<!-- ## Reference -->