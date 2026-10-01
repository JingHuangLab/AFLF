# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License")
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""AlphaFoldTools featurization script, revised from AlphaFold run script."""

import os
import pickle
import shutil

from absl import app
from absl import flags
from absl import logging
from alphafold.data import pipeline
from alphafold.data import pipeline_multimer
from alphafold.data import templates
from alphafold.data.tools import hhsearch
from alphafold.data.tools import hmmsearch

# Internal import (7716).

logging.set_verbosity(logging.INFO)

flags.DEFINE_string('fasta_path', None, 'Path to the FASTA file. If a FASTA file contains multiple sequences, then it will be folded as a multimer. This script permits only one FASTA to be processed per run.')
flags.DEFINE_string('output_dir', None, 'Path to a directory that will store the results.')

flags.DEFINE_string('jackhmmer_binary_path', shutil.which('jackhmmer'), 'Path to the JackHMMER executable.')
flags.DEFINE_string(  'hhblits_binary_path', shutil.which(  'hhblits'), 'Path to the   HHblits executable.')
flags.DEFINE_string( 'hhsearch_binary_path', shutil.which( 'hhsearch'), 'Path to the  HHsearch executable.')
flags.DEFINE_string('hmmsearch_binary_path', shutil.which('hmmsearch'), 'Path to the hmmsearch executable.')
flags.DEFINE_string( 'hmmbuild_binary_path', shutil.which( 'hmmbuild'), 'Path to the  hmmbuild executable.')
flags.DEFINE_string(   'kalign_binary_path', shutil.which(   'kalign'), 'Path to the    Kalign executable.')

flags.DEFINE_string(  'uniref90_database_path', None, 'Path to the      Uniref90 database for use by JackHMMER.')
flags.DEFINE_string(    'mgnify_database_path', None, 'Path to the        MGnify database for use by JackHMMER.')
flags.DEFINE_string(       'bfd_database_path', None, 'Path to the           BFD database for use by HHblits.')
flags.DEFINE_string( 'small_bfd_database_path', None, 'Path to the small ver BFD database for use by HHblits with the "reduced_dbs" preset.')
flags.DEFINE_string(  'uniref30_database_path', None, 'Path to the      UniRef30 database for use by HHblits.')
flags.DEFINE_string(   'uniprot_database_path', None, 'Path to the       Uniprot database for use by JackHMMer.')
flags.DEFINE_string(     'pdb70_database_path', None, 'Path to the         PDB70 database for use by HHsearch.')
flags.DEFINE_string('pdb_seqres_database_path', None, 'Path to the    PDB seqres database for use by hmmsearch.')

flags.DEFINE_string('template_mmcif_dir', None, 'Path to a directory with template mmCIF structures, each named <pdb_id>.cif')
flags.DEFINE_string('obsolete_pdbs_path', None, 'Path to file containing a mapping from obsolete PDB IDs to the PDB IDs of their replacements.')
flags.DEFINE_string( 'max_template_date', None, 'Maximum template release date to consider. Important if folding historical test sets.')

flags.DEFINE_enum(   'db_preset', 'full_dbs', ['full_dbs', 'reduced_dbs'],                              'Choose preset MSA database configuration - smaller genetic database config (reduced_dbs) or full genetic database config  (full_dbs)')
flags.DEFINE_enum('model_preset',  'monomer', ['monomer', 'monomer_casp14', 'monomer_ptm', 'multimer'], 'Choose preset model configuration - the monomer model, the monomer model with extra ensembling, monomer model with pTM head, or multimer model')

flags.DEFINE_boolean('use_precomputed_msas', False, 'Whether to read MSAs that have been written to disk instead of running the MSA tools. The MSA files are looked up in the output directory, so it must stay the same between multiple runs that are to reuse the MSAs. WARNING: This will not check if the sequence, database or configuration have changed.')

FLAGS = flags.FLAGS

MAX_TEMPLATE_HITS = 4

def _check_flag(flag_name: str,
                other_flag_name: str,
                should_be_set: bool):
  if should_be_set != bool(FLAGS[flag_name].value):
    verb = 'be' if should_be_set else 'not be'
    raise ValueError(f'{flag_name} must {verb} set when running with "--{other_flag_name}={FLAGS[other_flag_name].value}".')

def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  for tool_name in ('jackhmmer', 'hhblits', 'hhsearch', 'hmmsearch', 'hmmbuild', 'kalign'):
    if not FLAGS[f'{tool_name}_binary_path'].value:
      raise ValueError(f'Could not find path to the "{tool_name}" binary. Make sure it is installed on your system.')

  use_small_bfd = FLAGS.db_preset == 'reduced_dbs'
  _check_flag('small_bfd_database_path', 'db_preset', should_be_set=    use_small_bfd)
  _check_flag(      'bfd_database_path', 'db_preset', should_be_set=not use_small_bfd)
  _check_flag( 'uniref30_database_path', 'db_preset', should_be_set=not use_small_bfd)

  run_multimer_system = 'multimer' in FLAGS.model_preset
  _check_flag(     'pdb70_database_path', 'model_preset', should_be_set=not run_multimer_system)
  _check_flag('pdb_seqres_database_path', 'model_preset', should_be_set=    run_multimer_system)
  _check_flag(   'uniprot_database_path', 'model_preset', should_be_set=    run_multimer_system)

  if run_multimer_system:
    template_searcher = hmmsearch.Hmmsearch(binary_path         =FLAGS.hmmsearch_binary_path,
                                            hmmbuild_binary_path=FLAGS.hmmbuild_binary_path,
                                            database_path       =FLAGS.pdb_seqres_database_path, )
    template_featurizer = templates.HmmsearchHitFeaturizer(mmcif_dir         =FLAGS.template_mmcif_dir,
                                                           max_template_date =FLAGS.max_template_date,
                                                           max_hits          =MAX_TEMPLATE_HITS,
                                                           kalign_binary_path=FLAGS.kalign_binary_path,
                                                           release_dates_path=None,
                                                           obsolete_pdbs_path=FLAGS.obsolete_pdbs_path)
  else:
    template_searcher = hhsearch.HHSearch(binary_path=FLAGS.hhsearch_binary_path,
                                          databases  =[FLAGS.pdb70_database_path])
    template_featurizer = templates.HhsearchHitFeaturizer(mmcif_dir         =FLAGS.template_mmcif_dir,
                                                          max_template_date =FLAGS.max_template_date,
                                                          max_hits          =MAX_TEMPLATE_HITS,
                                                          kalign_binary_path=FLAGS.kalign_binary_path,
                                                          release_dates_path=None,
                                                          obsolete_pdbs_path=FLAGS.obsolete_pdbs_path)

  monomer_data_pipeline = pipeline.DataPipeline(jackhmmer_binary_path  =FLAGS.jackhmmer_binary_path,
                                                hhblits_binary_path    =FLAGS.hhblits_binary_path,
                                                uniref90_database_path =FLAGS.uniref90_database_path,
                                                mgnify_database_path   =FLAGS.mgnify_database_path,
                                                bfd_database_path      =FLAGS.bfd_database_path,
                                                uniref30_database_path =FLAGS.uniref30_database_path,
                                                small_bfd_database_path=FLAGS.small_bfd_database_path,
                                                template_searcher      =template_searcher,
                                                template_featurizer    =template_featurizer,
                                                use_small_bfd          =use_small_bfd,
                                                use_precomputed_msas   =FLAGS.use_precomputed_msas)

  if run_multimer_system:
    data_pipeline = pipeline_multimer.DataPipeline(monomer_data_pipeline=monomer_data_pipeline,
                                                   jackhmmer_binary_path=FLAGS.jackhmmer_binary_path,
                                                   uniprot_database_path=FLAGS.uniprot_database_path,
                                                   use_precomputed_msas =FLAGS.use_precomputed_msas)
  else:
    data_pipeline = monomer_data_pipeline

  logging.info(f'Processing: {FLAGS.fasta_path}')

  base_output_dir   = FLAGS.output_dir
  msas_output_dir   = os.path.join(base_output_dir, 'msas')
  features_pkl_path = os.path.join(base_output_dir, 'features.pkl')

  if not os.path.exists(base_output_dir): os.makedirs(base_output_dir)
  if not os.path.exists(msas_output_dir): os.makedirs(msas_output_dir)

  feature_dict = data_pipeline.process(input_fasta_path=FLAGS.fasta_path, msa_output_dir=msas_output_dir)

  with open(features_pkl_path, 'wb') as f:
    pickle.dump(feature_dict, f, protocol=4)

if __name__ == '__main__':
  flags.mark_flags_as_required(['fasta_path',
                                'output_dir',
                                'uniref90_database_path',
                                'mgnify_database_path',
                                'template_mmcif_dir',
                                'max_template_date',
                                'obsolete_pdbs_path', ])

  app.run(main)