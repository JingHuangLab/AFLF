#!/bin/bash
#SBATCH -N 1
#SBATCH -c 8
#SBATCH -p cpu
#SBATCH --mem=20GB

fasta_path=/home/songzl/projects/3.aftools/test2_tem1/tem1.fasta
output_dir=/home/songzl/projects/3.aftools/test2_tem1/

echo "Processing: ${output_dir}"

base_data_dir=/u/xiety/database/alphafold-2.2.0
base_binary_dir=/apps/anaconda/anaconda3/envs/alphafold/bin

# For monomer.
python ./featurizer.py \
       --fasta_path ${fasta_path} \
       --output_dir ${output_dir} \
       --jackhmmer_binary_path ${base_binary_dir}/jackhmmer \
       --hhblits_binary_path   ${base_binary_dir}/hhblits   \
       --hhsearch_binary_path  ${base_binary_dir}/hhsearch  \
       --hmmsearch_binary_path ${base_binary_dir}/hmmsearch \
       --hmmbuild_binary_path  ${base_binary_dir}/hmmbuild  \
       --kalign_binary_path    ${base_binary_dir}/kalign    \
       --uniref90_database_path ${base_data_dir}/uniref90/uniref90.fasta                                      \
       --mgnify_database_path   ${base_data_dir}/mgnify/mgy_clusters.fa                                       \
       --bfd_database_path      ${base_data_dir}/bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt \
       --uniref30_database_path ${base_data_dir}/uniclust30/uniclust30_2018_08/uniclust30_2018_08             \
       --pdb70_database_path    ${base_data_dir}/pdb70/pdb70                                                  \
       --template_mmcif_dir ${base_data_dir}/pdb_mmcif/mmcif_files  \
       --obsolete_pdbs_path ${base_data_dir}/pdb_mmcif/obsolete.dat \
       --max_template_date 1700-01-01 \
       --model_preset monomer

# # For multimer.
# python ./run_featurizer.py \
#        --fasta_path ${fasta_path} \
#        --output_dir ${output_dir} \
#        --jackhmmer_binary_path ${base_binary_dir}/jackhmmer \
#        --hhblits_binary_path   ${base_binary_dir}/hhblits   \
#        --hhsearch_binary_path  ${base_binary_dir}/hhsearch  \
#        --hmmsearch_binary_path ${base_binary_dir}/hmmsearch \
#        --hmmbuild_binary_path  ${base_binary_dir}/hmmbuild  \
#        --kalign_binary_path    ${base_binary_dir}/kalign    \
#        --uniref90_database_path   ${base_data_dir}/uniref90/uniref90.fasta                                      \
#        --mgnify_database_path     ${base_data_dir}/mgnify/mgy_clusters.fa                                       \
#        --bfd_database_path        ${base_data_dir}/bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt \
#        --uniref30_database_path   ${base_data_dir}/uniclust30/uniclust30_2018_08/uniclust30_2018_08             \
#        --uniprot_database_path    ${base_data_dir}/uniprot/uniprot.fasta                                        \
#        --pdb_seqres_database_path ${base_data_dir}/pdb_seqres/pdb_seqres.txt                                    \
#        --template_mmcif_dir ${base_data_dir}/pdb_mmcif/mmcif_files  \
#        --obsolete_pdbs_path ${base_data_dir}/pdb_mmcif/obsolete.dat \
#        --max_template_date 1700-01-01 \
#        --model_preset multimer