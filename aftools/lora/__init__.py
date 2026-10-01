r"""AFTools Low-Rank Adaptation (LoRA)."""
# Authors: Zilin Song.


from aftools.lora._lora import LoraCheckpointPreset, lora_func
from aftools.lora import models
from aftools.lora import models_multimer


# NOTE (ZS):
#   lora_checkpoint.pkl hierarchy of keys:
#     `latent_lora_feats` - if evoformer checkpoint.
#     |   `lora_feat_msa`  - `A`, `B`, `C`
#     |   `lora_feat_pair` - `A`, `B`, `C`
#     `latent_lora_masks` - if evoformer checkpoint.
#     |   `lora_mask_msa`
#     |   `lora_mask_pair`
#     `latent_reprs`      - if evoformer checkpoint.
#     |   `repr_msa`
#     |   `repr_pair`
#     |   `mask_msa`
#     |   `mask_pair`
#     `latent_lora_feats` - if structmod checkpoint.
#     |   `lora_feat_single` - `A`, `B`
#     |   `lora_feat_pair`   - `A`, `B`, `C`
#     `latent_lora_masks` - if structmod checkpoint.
#     |   `lora_mask_single`
#     |   `lora_mask_pair`
#     `latent_reprs`      - if structmod checkpoint.
#     |   `repr_single`
#     |   `repr_pair`
#     `batch`           - From checkpoint.pkl
#     |   `seq_mask`
#     |   `aatype`
#     |   `residue_index`
#     |   `atom14_atom_exists`      - only in AF
#     |   `atom37_atom_exists`      - only in AF
#     |   `residx_atom37_to_atom14` - only in AF
#     |   `asym_id`                 - only in AF-Multimer
#     |   `aftools_layer_index`     - the index of the latent layer input state
