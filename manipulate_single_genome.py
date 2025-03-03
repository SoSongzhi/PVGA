import random
from Bio import SeqIO

# 读取FASTA文件并获取基因组序列
def read_genome(fasta_file):
    genome = SeqIO.read(fasta_file, "fasta")
    return genome.seq

# 随机改变指定比例的碱基
def random_change(genome, change_ratio):
    genome_list = list(genome)  # 转换为列表，便于修改
    total_bases = len(genome_list)
    num_changes = int(total_bases * change_ratio)  # 计算需要改变的碱基数量

    for _ in range(num_changes):
        position = random.randint(0, total_bases - 1)  # 随机选择一个位置
        original_base = genome_list[position]
        bases = ['A', 'T', 'C', 'G']
        bases.remove(original_base)  # 确保选择不同的碱基
        new_base = random.choice(bases)  # 随机选择新的碱基
        genome_list[position] = new_base  # 替换为新的碱基

    return "".join(genome_list)

# 模拟基因组的随机改变
def simulate_random_changes(input_fasta, output_fasta, change_ratio):
    # 读取基因组
    genome = read_genome(input_fasta)

    # 执行随机改变
    mutated_genome = random_change(genome, change_ratio)

    # 保存变异后的基因组
    with open(output_fasta, "w") as out_f:
        out_f.write(f">mutated_genome\n{mutated_genome}")

    print(f"Original Genome Length: {len(genome)}")
    print(f"Mutated Genome Length: {len(mutated_genome)}")
    print(f"Change Ratio: {change_ratio * 100}%")
    print(f"Number of changes: {int(len(genome) * change_ratio)}")

# 主程序，指定输入和输出路径
input_fasta = "JRCSF.fa"  # 输入的基因组FASTA文件路径
output_fasta = "30_jrcsf.fasta"  # 输出的突变基因组FASTA文件路径
change_ratio = 0.30  # 改变的比例，例如15%

simulate_random_changes(input_fasta, output_fasta, change_ratio)