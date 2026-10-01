"""Extract the Latent Dynamics and compute the OTD and the Jaccard index."""
# Authors: Zilin Song.

import time
import functools
import jax
import jax.numpy as jnp

import sys
sys.path.insert(0, "../../")
import aftools.utils as _u
import aftools.afre  as _re
from aftools.lora._lora import _lora


@jax.jit
def rebuild_latent(traj_i: _u.af.TAFFeatures, chkp: _u.af.TAFFeatures) -> jnp.ndarray:
  r"""Rebuilts the latent Evoformer MSA representation."""
  copied = dict(**chkp)
  rebuilt = _lora(lora_feats=traj_i,
                  lora_masks=copied['evoformer_lora_masks'],
                  reprs     =copied['evoformer_reprs'],
                  lora_to='msa', )
  return rebuilt['repr_msa']
def ji(a: jnp.ndarray, b: jnp.ndarray) -> float:
  r"""Computes the set overlap as the Jaccard index = |A and B| / |A or B|."""
  union = jnp.logical_or (a, b).sum() # FT, TF, TT
  inter = jnp.logical_and(a, b).sum() #         TT
  return inter / (union + (union==0.))
def ot(a: jnp.ndarray, b: jnp.ndarray) -> float:
  r"""Computes the 1D optimal transport distances."""
  return jnp.linalg.norm(jnp.sort(a) - jnp.sort(b), ord=1) / a.shape[0]
def ma_mask(z: jnp.ndarray) -> jnp.ndarray:
  r"""Return the ijk indexes and the mask of all MSA ma-ed entries."""
  p1  = jnp.percentile(z,  1., method='midpoint')
  p99 = jnp.percentile(z, 99., method='midpoint')
  ijk = jnp.concatenate([jnp.column_stack(jnp.where(z<=p1)), jnp.column_stack(jnp.where(z>=p99))])
  mask = jnp.zeros(z.shape)
  mask = mask.at[ijk[:, 0], ijk[:, 1], ijk[:, 2]].set(True)
  return mask

def compute_dists(traj_i: _u.af.TAFFeatures,
                  traj_j: _u.af.TAFFeatures,
                  chkp:   _u.af.TAFFeatures,
                  topk:   int,
                  ) -> tuple[jnp.ndarray, jnp.ndarray]:
  # state i.
  zi = rebuild_latent(traj_i=traj_i, chkp=chkp).flatten()
  zi_idx = jnp.concatenate([jax.lax.top_k(-zi, topk)[1], jax.lax.top_k( zi, topk)[1]])
  ma_mi = jnp.zeros(zi.shape).at[zi_idx].set(True)
  ma_zi = zi.at[zi_idx].get()
  # state j.
  zj = rebuild_latent(traj_i=traj_j, chkp=chkp).flatten()
  zj_idx = jnp.concatenate([jax.lax.top_k(-zj, topk)[1], jax.lax.top_k( zj, topk)[1]])
  ma_mj = jnp.zeros(zj.shape).at[zj_idx].set(True)
  ma_zj = zj.at[zj_idx].get()
  # dists.
  ji_ = ji(a=ma_mi, b=ma_mj)
  ot_ = ot(a=ma_zi, b=ma_zj)
  return ji_, ot_


dest = './aftools_runtime/afre/'


if __name__ == "__main__":
  traj = _re.models.AFREState.from_json(open(f"{dest}/afre_state.json").read()).hist_lora_feats
  chkp = _u.io.pkl.load(f'{dest}/lora_checkpoint.pkl')
  topk = int(chkp['evoformer_reprs']['repr_msa'].size * 0.01)

  compute_dists_f = jax.jit(functools.partial(compute_dists, chkp=chkp, topk=topk))

  with open(f"{dest}/../../latdyn_dists.log", 'w') as f:
    for   i in jnp.arange(0, len(traj))[::20]:
      for j in jnp.arange(i, len(traj))[::20]:
        _t = time.perf_counter()
        ji_, ot_ = compute_dists_f(traj_i=traj[i], traj_j=traj[j])
        print  (f"{i:>6}|{j:>6}|{ji_:>12.8f}|{ot_:>12.8f}|{time.perf_counter() - _t:6.2f}s", flush=True)
        f.write(f"{i:>6}|{j:>6}|{ji_:>12.8f}|{ot_:>12.8f}|{time.perf_counter() - _t:6.2f}s\n")
        if j%1000==0: f.flush()
