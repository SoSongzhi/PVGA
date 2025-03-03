import sys
import numpy as np
from Bio import SeqIO
file_in = sys.argv[1]
f = open(file_in,'r')
seq = []
name = []

for s in SeqIO.parse(file_in,'fasta'):
    name.append(s.description)
    seq.append([c for c in s.seq])

# while True:
#     line = f.readline().strip()
#     if not line:
#         break
#     seq.append([c for c in f.readline().strip()])
#     name.append(line)

print(name)

l = len(name)
seq = np.array(seq)
flag = []
for s in seq:
	flag.append(s!='-')

# seq_type = {'896':0,'HXB2':1,'JRCSF':2,'NL43':3,'YU2':4}


length_all = np.zeros([l,l])
length2 = np.zeros([l,l])
div_mutation = np.zeros([l,l])
div_all = np.zeros([l,l])
for i in range(l-1):
	for j in range(i+1,l):
		diff = seq[i]!=seq[j]
		flag_ij = flag[i]*flag[j]
		all = np.count_nonzero(diff)
		mut = np.count_nonzero(diff*flag_ij)
		div_mutation[i,j] = mut
		div_all[i,j] = all
		length_all[i,j] = np.count_nonzero(flag[i]+flag[j])
		length2[i,j] = np.count_nonzero(flag[i]*flag[j])

div_mutation = div_mutation + div_mutation.T
div_all = div_all + div_all.T
length_all = length_all + length_all.T
length2 = length2 + length2.T
# print(div_mutation)
# print(div_all)
# print(len(seq[0]))
# print(length_all)
# print(np.round(div_all/(length_all+1e-16),4))
print((1-np.round(div_all/(length_all+1e-16),4)))



diversity_all = 100-np.round(div_all/(length_all+1e-16),4)*100


from matplotlib import pyplot as plt
# alpha = [i for i in seq_type.keys()]
alpha = [i for i in name]

fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(diversity_all, interpolation='nearest')
for (x, y), value in np.ndenumerate(diversity_all):
    plt.text(x, y, f"{value:.2f}", va="center", ha="center")

fig.colorbar(cax)

xaxis = np.arange(len(alpha))
ax.set_xticks(xaxis)
ax.set_yticks(xaxis)
ax.set_xticklabels(alpha)
ax.set_yticklabels(alpha)
plt.show()
