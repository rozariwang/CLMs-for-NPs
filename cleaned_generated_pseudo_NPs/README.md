# Cleaned Generated Pseudo-NPs

This directory contains the cleaned molecule CSVs generated from the original
`generated_pseudo_NPs/*_100000.csv` files. 

## Cleaning Summary

Each source CSV was processed independently with RDKit:

1. Removed duplicate raw SMILES within the file.
2. Removed molecules with RDKit structural problems, including explicit valence
   errors, kekulization failures, sanitization errors, ambiguous stereochemistry,
   and isolated hydrogen atoms without neighbors.
3. Standardized accepted molecules with `rdMolStandardize`: fragment parent
   selection, functional group normalization, and reionization.
4. Neutralized charges and canonicalized tautomers.
5. Wrote canonical isomeric SMILES and removed duplicates that collapsed to the
   same canonical form.
6. Removed molecules also present in the already processed pre-training files:
   `train_rd.txt`, `val_rd.txt`, and `test_rd.txt`.

Detailed rejection counts are stored in `cleaning_summary.csv`.

The 48 cleaned files contain **1,107,486** rows after pre-training overlap
removal. The combined cross-file unique set is
`combined_unique_cleaned_molecules.csv`, which contains **1,048,872** unique
molecules after removing **58,614** duplicates across cleaned files.

## Molecules Left Per File

| File | Molecules left | Pre-training overlaps removed |
|---|---:|---:|
| GPT_Char_rds_generated_mol.csv | 28,304 | 2,457 |
| GPT_Char_sfs_generated_mol.csv | 27,173 | 2,484 |
| GPT_ais_rds_generated_mol.csv | 27,818 | 2,896 |
| GPT_ais_sfs_generated_mol.csv | 28,284 | 3,596 |
| GPT_bpe_rds_generated_mol.csv | 25,008 | 3,753 |
| GPT_bpe_sfs_generated_mol.csv | 25,955 | 3,660 |
| GPT_npbpe1000_rds_generated_mol.csv | 24,982 | 6,922 |
| GPT_npbpe1000_sfs_generated_mol.csv | 24,935 | 7,837 |
| GPT_npbpe100_rds_generated_mol.csv | 24,721 | 4,548 |
| GPT_npbpe100_sfs_generated_mol.csv | 24,776 | 5,244 |
| GPT_npbpe30k_rds_generated_mol.csv | 16,409 | 4,422 |
| GPT_npbpe30k_sfs_generated_mol.csv | 12,817 | 3,788 |
| GPT_npbpe60_rds_generated_mol.csv | 25,207 | 3,425 |
| GPT_npbpe60_sfs_generated_mol.csv | 25,937 | 3,819 |
| GPT_npbpe7924_rds_generated_mol.csv | 20,194 | 3,969 |
| GPT_npbpe7924_sfs_generated_mol.csv | 19,192 | 4,596 |
| M1_Char_rds_generated_mol.csv | 24,733 | 4,923 |
| M1_Char_sfs_generated_mol.csv | 24,634 | 5,202 |
| M1_ais_rds_generated_mol.csv | 25,696 | 4,214 |
| M1_ais_sfs_generated_mol.csv | 26,655 | 4,924 |
| M1_bpe_rds_generated_mol.csv | 23,800 | 4,841 |
| M1_bpe_sfs_generated_mol.csv | 24,628 | 5,684 |
| M1_npbpe1000_rds_generated_mol.csv | 23,274 | 3,603 |
| M1_npbpe1000_sfs_generated_mol.csv | 23,826 | 4,528 |
| M1_npbpe100_rds_generated_mol.csv | 24,452 | 6,560 |
| M1_npbpe100_sfs_generated_mol.csv | 24,435 | 5,824 |
| M1_npbpe30k_rds_generated_mol.csv | 16,090 | 4,094 |
| M1_npbpe30k_sfs_generated_mol.csv | 15,652 | 5,175 |
| M1_npbpe60_rds_generated_mol.csv | 23,309 | 4,840 |
| M1_npbpe60_sfs_generated_mol.csv | 25,456 | 5,568 |
| M1_npbpe7924_rds_generated_mol.csv | 19,212 | 9,322 |
| M1_npbpe7924_sfs_generated_mol.csv | 19,953 | 4,899 |
| M2_Char_rds_generated_mol.csv | 25,307 | 5,100 |
| M2_Char_sfs_generated_mol.csv | 24,351 | 5,993 |
| M2_ais_rds_generated_mol.csv | 26,208 | 4,295 |
| M2_ais_sfs_generated_mol.csv | 25,596 | 4,699 |
| M2_bpe_rds_generated_mol.csv | 22,715 | 4,745 |
| M2_bpe_sfs_generated_mol.csv | 25,038 | 6,121 |
| M2_npbpe1000_rds_generated_mol.csv | 22,645 | 8,016 |
| M2_npbpe1000_sfs_generated_mol.csv | 22,034 | 8,701 |
| M2_npbpe100_rds_generated_mol.csv | 24,764 | 3,218 |
| M2_npbpe100_sfs_generated_mol.csv | 26,267 | 4,097 |
| M2_npbpe30k_rds_generated_mol.csv | 15,563 | 4,320 |
| M2_npbpe30k_sfs_generated_mol.csv | 13,389 | 3,436 |
| M2_npbpe60_rds_generated_mol.csv | 25,597 | 5,653 |
| M2_npbpe60_sfs_generated_mol.csv | 23,592 | 2,911 |
| M2_npbpe7924_rds_generated_mol.csv | 19,134 | 3,864 |
| M2_npbpe7924_sfs_generated_mol.csv | 17,769 | 10,143 |
