r"""AFRE checkpoint inference."""
# Authors: Zilin Song.

import sys
sys.path.insert(0, "../../")
import aftools.utils        as _u
import aftools.chkp.runners as _r

dest = '.'

config = _u.af.AFConfig(model_name='model_1_ptm')
params = _u.af.AFParams(model_name='model_1_ptm', params_dir="/u/songzl/3.alphafoldtools/params")

runner = _r.InferForCheckpointRunner(project_dir=f'./{dest}')
runner.execute(config=config,
               params=params,
               checkpoint_at=0,
               features_seed=None,
               features_pkl=_u.io.pkl.load(file=f'./{dest}/features.pkl'))

checkpoint_pkl = _u.io.pkl.load(file=f'./{dest}/aftools_runtime/checkpoint/infer_for_checkpoint/checkpoint.pkl')
print(checkpoint_pkl.keys())
for _, (k, v) in enumerate(checkpoint_pkl.items()):
  print(k, v.keys() if isinstance(v, dict) else v)

print(checkpoint_pkl['latent_reprs']['repr_pair'].shape, flush=True)

runner = _r.InferFromCheckpointRunner(project_dir=f'./{dest}')
runner.execute(config=config, params=params, use_bf16=True, checkpoint_pkl=checkpoint_pkl)
