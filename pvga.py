import networkx as nx
import readgraph as rg
import choosepaths as cp
import writesequence as ws
import os
import argparse
import utils
import polisher as pl

import edit_distance as ed
from networkx.readwrite.graph6 import write_graph6
import shutil
import os
import time
import datetime

# 构图
if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('-r', type=str, required=True, help="Reads file for graph construction (in fasta format).")
	parser.add_argument('-b', type=str, required=True, help="Backbone sequence file for graph construction (in fasta format).")
	parser.add_argument('-n', type=int, required=True, help="Number of Iterations for graph construction.")
	parser.add_argument('-gt',type=str, required=True, help="Ground truth sequence" )
	parser.add_argument('-od', type=str, required=True, help="Outdir")

	args = parser.parse_args()
	ec_reads = args.r
	backbone = args.b
	groundtruth = args.gt
	num_iterations = args.n

	graph_out_pref = os.path.join(ec_reads + "_ON_" + os.path.splitext(os.path.basename(backbone))[0])
	graph_out_pref = os.path.basename(graph_out_pref)


	for i in range(num_iterations):

		graph_filename = f"iter{i+1}.graph"
		graph_folder = os.path.join(graph_out_pref, f"iter{i+1}")
		graph_location = os.path.join(graph_folder, graph_filename)

		os.makedirs(graph_folder, exist_ok=True)
	


		start_time = time.perf_counter()
		oldbackbone = backbone

		# print(f"Function execution time: {end_time - start_time:.6f} seconds")
		if i == 0 : 
			aln_graph = utils.construct_aln_graph_from_fasta(ec_reads, backbone)
			aln_graph.merge_nodes()
		else :
			aln_graph = utils.construct_aln_graph_from_fasta(ec_reads, pb)
			aln_graph.merge_nodes()
		# aln_graph = utils.construct_aln_graph_from_fasta(ec_reads, consensus_location)

		# aln_graph.merge_nodes()
		# pbdconsensus = aln_graph.generate_consensus()[0]
	
		# pbdconsensus_location = os.path.join( os.path.join(graph_folder , f"pbdconsensus"), 'output_{}_{}.fa'.format(f"iter{i+1}", 0))
	
		nx_pack = utils.aln2nx(aln_graph)

		utils.nx2gfa(nx_pack[0], graph_filename, graph_folder)

		# 读取图文件
		graph = rg.read_graph_from_file(graph_location)
		print(graph)

		topological_order = list(nx.topological_sort(graph))
		start_node = topological_order[0]
		start_nodes = [node for node in graph.nodes() if graph.in_degree(node) == 0]
		end_nodes = [node for node in graph.nodes() if graph.out_degree(node) == 0]
		end_node = topological_order[-1]
	
		# 计算最大权重路径
		# max_weight, max_weight_path, max_weight_sequences, node_graph = cp.max_weight_path_cuda(graph)
		max_weight, max_weight_path, max_weight_sequences, node_graph = cp.max_weight_path_old(graph, start_node, end_node)
		# max_weight, max_weight_path, max_weight_sequences, node_graph = cp.max_weight_path_gpu(graph, start_node, end_node)

		end_time = time.perf_counter()

		polished_graph, polished_sequences = pl.polish(node_graph)
		pb=ws.store_labels_as_fa_hanshuming(polished_sequences, graph_folder,'dp_polish')
		
		backbone = graph_location = os.path.join(graph_folder, 'output_{}_{}.fa'.format(graph_out_pref, 0))
		print("graph_folder", graph_folder)
		sequence_location = ws.store_labels_as_fa(max_weight_sequences, graph_folder)
		# pbdconsensus =  ws.store_labels_as_fa([pbdconsensus[::-1]], os.path.join(graph_folder , f"pbdconsensus"))
		print(f"Iteration {i+1} completed.")
		print("-----------------------------------------------------------------------------------------------")
		if utils.are_sequences_identical(oldbackbone, sequence_location):
			break
		
		sequence_location = os.path.join(graph_folder, 'output_{}_{}.fa'.format(graph_out_pref, 0))
		if i == 1:
			 dp_noiter_result = pb


		
	
	dp_time = end_time - start_time
	#os.system(f"quast.py {backbone} -r {groundtruth}")
	polish_folder = os.path.join(graph_out_pref, f"polish")
	os.makedirs(polish_folder, exist_ok=True)
	polished_location = os.path.join(polish_folder, 'output_{}_{}.fa'.format(graph_out_pref, 0))



	polished_graph, polished_sequences = pl.polish(node_graph)
	# nx.write_gml(node_graph, os.path.join(polish_folder, 'graph.gml'))
	# nx.write_gml(polished_graph, os.path.join(polish_folder, 'polished_graph.gml'))

	ws.store_labels_as_fa(polished_sequences, polish_folder)

	if not os.path.exists(args.od):
		os.makedirs(args.od)
	# move together
	dst_file=os.path.join(args.od, "dp_polished.fa")
	shutil.copy(polished_location, dst_file)

	dst_file=os.path.join(args.od, "dp_result.fa")
	shutil.copy(sequence_location, dst_file)

	dst_file=os.path.join(args.od, "dp_noiter_result.fa")
	shutil.copy(dp_noiter_result, dst_file)

	# dst_file=os.path.join(args.od, "vote_consensus.fa")
	# shutil.copy(consensus_location, dst_file)

	# import writegraphconsensus as wss
	# wss.store_labels_as_fa([graph_consensus], polish_folder)