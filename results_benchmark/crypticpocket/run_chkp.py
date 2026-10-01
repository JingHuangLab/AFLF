r"""AFRE checkpoint inference."""
# Authors: Runtong Qian.


import aftools.utils        as _u
import aftools.chkp.runners as _r

import sys, os
system = sys.argv[1]
dest = f'./{system}'
os.makedirs(dest, exist_ok=True)


config = _u.af.AFConfig(model_name='model_1_ptm')
params = _u.af.AFParams(model_name='model_1_ptm', params_dir="/u/songzl/3.alphafoldtools/params")

runner = _r.InferForCheckpointRunner(project_dir=f'{dest}')
runner.execute(config=config,
               params=params,
               checkpoint_at=40,
               features_seed=123456,
               features_pkl=_u.io.pkl.load(file=f'{dest}/features.pkl'))


checkpoint_pkl = _u.io.pkl.load(file=f'{dest}/aftools_runtime/checkpoint/infer_for_checkpoint/checkpoint.pkl')
print(checkpoint_pkl.keys())
for _, (k, v) in enumerate(checkpoint_pkl.items()):
  print(k, v.keys())

print(checkpoint_pkl['latent_reprs']['repr_pair'].shape)

runner = _r.InferFromCheckpointRunner(project_dir=f'{dest}')
runner.execute(config=config, params=params, checkpoint_pkl=checkpoint_pkl, use_bf16=False)
