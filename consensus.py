import pysam
import numpy as np
import os
from Bio import SeqIO

def get_consensus_sequence(reads, reference, folder):
    os.system(f"rm -rf {folder}")
    os.system(f"mkdir {folder}")

    samfile = f"{folder}/align.sam"
    unique_samfile = f"{folder}/align_unique.sam"
    bamfile = f"{folder}/align_unique.bam"
    sorted_bamfile = f"{folder}/align_unique_sorted.bam"

    


    os.system(f"minimap2 -a {reference} {reads} > {samfile}")
    os.system(f"samtools view -h -F 0x900 {samfile} > {unique_samfile}")
    os.system(f"samtools view -b {unique_samfile} > {bamfile} ")
    os.system(f"samtools sort {bamfile} -o {sorted_bamfile} ")
    os.system(f"samtools index {sorted_bamfile}")

    file_bam = sorted_bamfile
    bamfile = pysam.AlignmentFile(file_bam)
    ref_name = bamfile.references[0]
    length = bamfile.lengths[0]
    a, c, g, t = bamfile.count_coverage(ref_name, quality_threshold=0)

    reads_all = []
    pos_all = []
    for column in bamfile.pileup(ref_name, 0, length, min_base_quality=0, max_depth=5000000):
        reads_all.append(column.n)
        pos_all.append(column.pos)

    reads_all = np.array(reads_all)
    pos_all = np.array(pos_all)

    consensus_seq = []
    chara = {0: "A", 1: "C", 2: "G", 3: "T", 4: "-"}
    seq_list = []

    for i in range(len(pos_all)):
        index = pos_all[i]
        depth = reads_all[i]
        pos = index + 1
        if depth > 0:
            deletion = depth - a[index] - c[index] - g[index] - t[index]
            temp = np.array([a[index], c[index], g[index], t[index], deletion])
            flag = np.argsort(-temp)[0]
            if flag == 4:
                continue
            else:
                seq_list.append(chara[flag])

    seq = "".join(seq_list)
    reads_name = os.path.splitext(os.path.basename(reads))[0]
    out_file = f"{folder}/consensus_{reads_name}_{os.path.splitext(os.path.basename(reference))[0]}.fa"

    with open(out_file, "w") as f:
        f.write(f">consensus\n{seq}\n")

    return out_file