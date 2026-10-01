r"""AFTools checkpoint extraction and inference."""
# Authors: Zilin Song.


from aftools.chkp._confidence import (af_confidence, AFConfidenceModule,
                                      af_confidence_loss, AFConfidenceLossSpecPreset,
                                      af_multimer_confidence, AFMultimerConfidenceModule, )
from aftools.chkp import modules
from aftools.chkp import modules_multimer
from aftools.chkp import models
from aftools.chkp import models_multimer
from aftools.chkp import runners
from aftools.chkp import runners_multimer


# NOTE (ZS):
#   checkpoint.pkl hierarchy of keys:
#     `batch`
#     |   `seq_mask`
#     |   `aatype`
#     |   `residue_index`
#     |   `atom14_atom_exists`      - only in AF
#     |   `atom37_atom_exists`      - only in AF
#     |   `residx_atom37_to_atom14` - only in AF
#     |   `asym_id`                 - only in AF-Multimer
#     |   `aftools_layer_index`     - the index of the latent layer input state
#     `latent_reprs`  - if evoformer checkpoint.
#     |   `repr_msa`
#     |   `repr_pair`
#     |   `mask_msa`
#     |   `mask_pair`
#     `latent_reprs`  - if structmod checkpoint.
#     |   `repr_single`
#     |   `repr_pair`