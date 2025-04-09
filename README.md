# PVGA

![GitHub License](https://img.shields.io/github/license/yourusername/yourrepository)
![Version](https://img.shields.io/badge/version-1.0-blue)

### Overview
**PVGA** is a powerful virus-focused assembler that does both assembly and polishing. For virus genomes, small changes will lead to significant differences in terms of viral function and pathogenicity.  Thus, for virus-focused assemblers, high-accuracy results are crucial. Our approach heavily depends on the input reads as evidence to produce the reported genome. It first adopts a reference genome to start with.  We then align all the reads against the reference genome to get an alignment graph. After that, we use a dynamic programming algorithm to compute a path with the maximum weight of edges supported by reads. Most importantly, the obtained path is used as the new reference genome and the process is repeated until no further improvement is possible. 


### Installation
To install and use **PVGA**, please follow these steps:

#### pip install version
```bash
   conda create -n pvga python=3.10
   conda install bioconda::blasr
   pip install pvga
   pip install -r requirements.txt
   ``` 
#### conda install version

```bash
   conda create -n pvga python=3.10
   conda install bioconda:: blasr
   conda install pvga
  ```
### Usage

To display the help message and see the available command-line options for the pvga script, run the following command in your terminal:
```bash
pvga -h
```

To perform assembly using the pvga.py script, use the following command structure:

```bash
pvga -r [reads location] -b [backbone locatino] -o [output dir]
```
#### Arguments

- **`-r [reads location]`, `--reads [reads location]`**:  
  Path to the input reads file or directory containing the sequencing reads (e.g., FASTQ or FASTA files).

- **`-b [backbone location]`, `--backbone [backbone location]`**:  
  Path to the backbone sequence file (e.g., a reference genome or plasmid in FASTA format).

- **`-n [ITERATION NUM]`, `--iterations [ITERATION NUM]`**:  
  (Optional) Number of iterations to run the assembly process. This controls the depth or refinement of the assembly.

- **`-o [output dir]`, `--output_dir [output dir]`**:  
  Path to the directory where the output files (e.g., assembled sequences, logs, and reports) will be saved.


#### Example Command
```bash
pvga -r hiv_30x_4k_id90_98_2.5.fastq -b HXB2.fa -n 10 -o test_pvga
```

### For paired-end reads
Users can also assembly pair-end reads using PVGA. Please use tool bbmap to merge paired-end reads while preserving paired-end information. The merging process command shows as below:

```bash
conda install bioconda::bbmap
bbmerge.sh in1=reads1.fastq in2=reads2.fastq out=merged.fastq outu1=unmerged1.fastq outu2=unmerged2.fastq
cat merged.fastq unmerged1.fastq unmerged2.fastq > all.fastq
```

Then the all.fastq can be assemblied using PVGA.


### License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

### Contact
For questions or support, please contact [songzhics@gmail.com] or open an issue on GitHub.
```



