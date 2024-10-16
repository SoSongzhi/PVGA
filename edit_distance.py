import sys
import pysam
import numpy as np
import os
from Bio import SeqIO

def calculate_alignment_stats(query, truth):

    os.system("rm -rf alignment_file")
    os.system("mkdir alignment_file")
    truth_name = os.path.basename(os.path.dirname(truth))
    query_name = os.path.basename(os.path.dirname(query))

    samfile = f"alignment_file/{query_name}_on_{truth_name}.sam"
    unique_samfile = f"alignment_file/{query_name}_on_{truth_name}_unique.sam"
    bamfile = f"alignment_file/{query_name}_on_{truth_name}_unique.bam"
    sorted_bamfile = f"alignment_file/{query_name}_on_{truth_name}_unique_sorted.bam"

    os.system(f'minimap2 -a {truth} {query} > {samfile}')
    os.system(f"samtools view -h -F 0x900 {samfile} > {unique_samfile}")
    os.system(f'samtools view -b {unique_samfile} > {bamfile}')
    os.system(f'samtools sort {bamfile} -o {sorted_bamfile}')
    os.system(f'samtools index {sorted_bamfile}')

    file_bam = sorted_bamfile
    bamfile = pysam.AlignmentFile(file_bam)
    
    for s in SeqIO.parse(query, 'fasta'):
        query_length = len(s.seq)
        print({query_name} ,":", {query_length})

    for s in SeqIO.parse(truth, 'fasta'):
        truth_length = len(s.seq)
        print({truth_name} ,":", {truth_length})

    contig_length = []
    ref_length = []
    cigar_length = []
    mismatch = []
    indel = []
    edit_distance = []
    contig_name = []
    contig_ref = []

    for contig in bamfile.fetch():
        align_length = contig.reference_length
        cigar = contig.get_cigar_stats()[0]
        indel = cigar[1] + cigar[2]
        unaligned = cigar[4]
        diff = cigar[-1]
        mismatch = diff - indel
        edit_dist = unaligned + diff + truth_length - align_length
        print(f"unaligned_length: {unaligned}")
        print(f"edit_distance: {edit_dist}")
        print(f"total_mismatch: {mismatch}")
        print(f"total_indel: {indel}")
