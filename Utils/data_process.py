from scipy.io import loadmat
import pickle
from collections import defaultdict
import scipy.sparse as sp
from tqdm import tqdm

"""
	Reads data and save the adjacency matrices to adjacency lists
"""

def sparse_to_adjlist(sp_matrix, filename):
	"""
	Transfer sparse matrix to adjacency list
	:param sp_matrix: the sparse matrix
	:param filename: the filename of adjlist
	"""
	# add self loop
	homo_adj = sp_matrix + sp.eye(sp_matrix.shape[0])
	# create adj_list
	adj_lists = defaultdict(set)
	edges = homo_adj.nonzero()
	for index, node in tqdm(enumerate(edges[0])):
		adj_lists[node].add(edges[1][index])
		adj_lists[edges[1][index]].add(node)
	with open(filename, 'wb') as file:
		pickle.dump(adj_lists, file)
	file.close()


if __name__ == "__main__":
	yelp = loadmat('./Data/YelpChi.mat')
	net_rur = yelp['net_rur']
	net_rtr = yelp['net_rtr']
	net_rsr = yelp['net_rsr']
	yelp_homo = yelp['homo']

	sparse_to_adjlist(net_rur, './Data/yelp_rur_adjlists.pickle')
	sparse_to_adjlist(net_rtr, './Data/yelp_rtr_adjlists.pickle')
	sparse_to_adjlist(net_rsr,  './Data/yelp_rsr_adjlists.pickle')
	sparse_to_adjlist(yelp_homo, './Data/yelp_homo_adjlists.pickle')

	amz = loadmat('./Data/Amazon.mat')
	net_upu = amz['net_upu']
	net_usu = amz['net_usu']
	net_uvu = amz['net_uvu']
	amz_homo = amz['homo']

	sparse_to_adjlist(net_upu, './Data/amz_upu_adjlists.pickle')
	sparse_to_adjlist(net_usu, './Data/amz_usu_adjlists.pickle')
	sparse_to_adjlist(net_uvu, './Data/amz_uvu_adjlists.pickle')
	sparse_to_adjlist(amz_homo, './Data/amz_homo_adjlists.pickle')
