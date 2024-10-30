# PVGA

![GitHub License](https://img.shields.io/github/license/yourusername/yourrepository)
![Version](https://img.shields.io/badge/version-1.0-blue)

### Overview
**PVGA** is a powerful virus-focused assembler that does both assembly and polishing. For virus genomes, small changes will lead to significant differences in terms of viral function and pathogenicity.  Thus, for virus-focused assemblers, high-accuracy results are crucial. Our approach heavily depends on the input reads as evidence to produce the reported genome. It first adopts a reference genome to start with.  We then align all the reads against the reference genome to get an alignment graph. After that, we use a dynamic programming algorithm to compute a path with the maximum weight of edges supported by reads. Most importantly, the obtained path is used as the new reference genome and the process is repeated until no further improvement is possible. 


### Installation
To install and use **PVGA**, please follow these steps:

```bash
   git clone https://github.com/SoSongzhi/PVGA.git
   cd PVGA
   pip install -r requirements.txt

   ``` 

### Usage

```bash
python pvga.py -r [reads location] -b [backbone locatino] -n [ITERATION NUM] -od [output dir]
```

### License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

### Contact
For questions or support, please contact [songzhics@gmail.com] or open an issue on GitHub.
```
