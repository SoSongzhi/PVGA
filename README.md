# PVGA

![GitHub License](https://img.shields.io/github/license/yourusername/yourrepository)
![Version](https://img.shields.io/badge/version-1.0-blue)

### Overview
**PVGA** is a powerful virus-focused assembler that does both assembly and polishing. For virus genomes, small changes will lead to significant differences in terms of viral function and pathogenicity.  Thus, for virus-focused assemblers, high-accuracy results are crucial. Our approach heavily depends on the input reads as evidence to produce the reported genome. It first adopts a reference genome to start with.  We then align all the reads against the reference genome to get an alignment graph. After that, we use a dynamic programming algorithm to compute a path with the maximum weight of edges supported by reads. Most importantly, the obtained path is used as the new reference genome and the process is repeated until no further improvement is possible. 


### Installation
To install and use **PVGA**, please follow these steps:

```bash
   sudo apt install blasr
   conda create -n pvga python=3.10
   conda activate pvga
   git clone https://github.com/SoSongzhi/PVGA.git
   cd PVGA
   pip install -r requirements.txt

   ``` 

### Usage

To display the help message and see the available command-line options for the pvga.py script, run the following command in your terminal:
```bash
python pvga.py -h
```

To perform assembly using the pvga.py script, use the following command structure:

```bash
python pvga.py -r [reads location] -b [backbone locatino] -n [ITERATION NUM] -od [output dir]
```
#### Arguments

- **`-r [reads location]`, `--reads [reads location]`**:  
  Path to the input reads file or directory containing the sequencing reads (e.g., FASTQ or FASTA files).

- **`-b [backbone location]`, `--backbone [backbone location]`**:  
  Path to the backbone sequence file (e.g., a reference genome or plasmid in FASTA format).

- **`-n [ITERATION NUM]`, `--iterations [ITERATION NUM]`**:  
  Number of iterations to run the assembly process. This controls the depth or refinement of the assembly.

- **`-od [output dir]`, `--output_dir [output dir]`**:  
  Path to the directory where the output files (e.g., assembled sequences, logs, and reports) will be saved.


### Example Command
```bash
python pvga.py -r hiv_30x_4k_id90_98_2.5.fastq -b HXB2.fa -n 10 -od test_pvga
```


### License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

### Contact
For questions or support, please contact [songzhics@gmail.com] or open an issue on GitHub.
```



