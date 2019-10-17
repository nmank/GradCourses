
import numpy as np
import networkx as nx
import impute as imp
from sklearn import datasets, linear_model
from scipy.cluster.hierarchy import dendrogram, linkage
import scipy.spatial.distance as ssd
import sklearn
import calcom
from numpy import genfromtxt
import scipy
import sklearn.metrics as sk 
import matplotlib
# include this for katrina
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import pylab
from scipy.sparse import linalg 
import scipy as sp



def adjacency_matrix(X,msr = 'parcor', epsilon = 0, h_k_param = 2, negative= False, weighted =True):
	'''
	A function that builds an adjacecny matrix out of data using two methods

	inputs: 1) data matrix 
					n rows m columns (m data points living in R^n)
	        2) method for calculating distance between data points 
					corrolation or heatkernel or partial correlation
			 3) epsilon
					a user-parameter that determines will disconnect 
					all points that are further away or less corrolated than epsilon
			 3) weighted
					create a weighted matrix if true
			 4) negative 
					include negative correlations? (default is False)
	outputs: 1) adjacency matrix
					represents a directed weighted graph of the data
			 2) node_labels
			 		not completely functional yet
	'''
	n,m = X.shape;
	AdjacencyMatrix = np.zeros((m,m))
	node_labels = np.zeros(m)
	if msr == 'correlation':
		for i in range(m):
			for j in range(m):
				if i > j:
					#compute correlation
					nXi = np.linalg.norm(X[:,i])
					nXj = np.linalg.norm(X[:,j])
					if nXi == 0 or nXj == 0:	
							tmp = 0
					else:
						Cor = np.dot(X[:,i],X[:,j])/(nXi*nXj)
					#if want absolute value
					if negative == False:
						Cor = abs(Cor);
					#if using epsilon
					if epsilon != 0 and Cor < epsilon:
						Cor = 0
					AdjacencyMatrix[j,i] =Cor;
					AdjacencyMatrix[i,j] =Cor;
	elif msr == 'heatkernel':
		for i in range(m):
			for j in range(m):
				if i > j:
					#calculate the euclidean norm distance
					Dist = np.exp(-(np.linalg.norm(X[:,i] - X[:,j]) ** 2 /(h_k_param ** 2)));
					#if we are using epsilon
					if epsilon != 0 and Dist < epsilon:
						Dist = 0;
					#if we're not using epsilon
					AdjacencyMatrix[i,j] = Dist;
					AdjacencyMatrix[j,i] = Dist;

	elif msr == 'mutual_information':
		for i in range(m):
			for j in range(m):
				if i > j:

					AdjacencyMatrix[i,j] = sk.mutual_info_score(X[:,i],X[:,j])
					AdjacencyMatrix[j,i] = AdjacencyMatrix[i,j]
					if epsilon != 0 and AdjacencyMatrix[i,j] < epsilon:
						#get rid of all edges that have weights less than epsilon
						AdjacencyMatrix[i,j] = 0
						AdjacencyMatrix[j,i] = 0

	elif msr == 'parcor':
		# create linear regression object 
		reg = linear_model.LinearRegression();

		vis = list(range(m));

		for i in range(m):
			for j in range(m):
				if i > j:

					#compute projections (aka linear regressions)
					vis.remove(i);
					vis.remove(j);					
					reg.fit(X[:,vis], X[:,i]); 
					x_hat_i = reg.predict(X[:,vis]);
					reg.fit(X[:,vis], X[:,j]);
					x_hat_j = reg.predict(X[:,vis]);

					#compute residuals
					Y_i = X[:,i] - x_hat_i;
					Y_j = X[:,j] - x_hat_j;

					Y_in = np.linalg.norm(Y_i)
					Y_jn = np.linalg.norm(Y_j)

					#store note labels
					node_labels[i] = Y_in

					if Y_in == 0 or Y_jn == 0:	
						tmp = 0

					else:
						tmp = np.dot(Y_i, Y_j)/(Y_in*Y_jn)
					if negative == True:
						PC = tmp
					else:
						PC = np.abs(tmp)
						if epsilon != 0 and PC < epsilon:
							#get rid of all edges that have weights less than epsilon
							PC = 0
					#why are we getting partial correlations of 1?
					if PC > 1 and PC < 1.0000001:
						PC = 1


					AdjacencyMatrix[i,j] =PC;
					AdjacencyMatrix[j,i] =PC;
					vis = list(range(m));
	return(AdjacencyMatrix, node_labels)


def wgcna(x, beta = 1, den_gen = 'average', threshold = 0, den_fname = 'wgcna_den.png', den_title = 'WGCNA Shedder Data at 60 Hours After Infection', tom = True):
	'''
	Basic WGCNA implementation	
	'''
	m,n = x.shape

	a = np.zeros((n,n))
	for i in range(n): 
		for j in range(n): 
			if i > j: 
				nxi = np.linalg.norm(x[:,i])
				nxj = np.linalg.norm(x[:,j])
				a[i,j] = np.abs((x[:,i] @ x[:,j])/(nxi*nxj))**beta
				a[j,i] = a[i,j] 

	if tom == True:
		#topological overlap measure
		w = np.zeros((n,n)) 
		for i in range(n): 
			for j in range(n): 
				if i > j: 
					l = a[:,i] @ a[j,:]
					k = np.min((np.sum(a[:,i]),np.sum(a[:,j])))
					w[i,j] = (l+a[i,j])/(k+1-a[i,j])
					w[j,i] = w[i,j]
				elif i == j:
					w[i,i]=1
	else:
		w = a
		for i in range(n):
			w[i,i] = 1
	print(w)

	d = 1-w

	sd = sp.spatial.distance.squareform(d)

	Z = sp.cluster.hierarchy.linkage(sd, den_gen)

	# dn = sp.cluster.hierarchy.dendrogram(Z)
	fig = pylab.figure(figsize=(8,8))
	ax1 = fig.add_axes([0.07,0.03,0.26,0.88])
	Z = dendrogram(Z,orientation='left')
	ax1.set_xticks([])
	ax1.set_yticks([])

	axmatrix = fig.add_axes([0.34,0.03,0.6,0.88])
	fig.suptitle(den_title)
	idx1 = Z['leaves']
	x0 = np.e ** x[:,idx1].T
	im = axmatrix.matshow(x0, aspect='auto', origin='lower', cmap=pylab.cm.YlGnBu)
	cbar = fig.colorbar(im)
	axmatrix.set_xticks([])
	axmatrix.set_yticks([])

	#fig.show()

	pylab.savefig(den_fname)

	#cut
	Z = sp.cluster.hierarchy.linkage(sd, den_gen)

	return sd, Z

def cluster_den(Z,x,cut_height = .7):
	m,n = x.shape
	cluster_ind_ar = sp.cluster.hierarchy.cut_tree(Z, height = cut_height)
	cluster_ind_ls = list(map(int, cluster_ind_ar))

	#seperate into clusters
	nodes = np.arange(n)
	clusters_d = {}
	for i in cluster_ind_ls:
		if i not in clusters_d.keys():
			clusters_d[i] = []
	for i in nodes:
		clusters_d[cluster_ind_ls[i]].append(i)
	clusters = list(clusters_d.values())


	#calculate eigengenes
	eigengenes = []
	for c in clusters:
		x1 = np.zeros((m,len(c)))
		for i in range(len(c)):
			x1[:,i] = x[:,c[i]] 
		if x1.size == m:
			eigengenes.append(x1.T/np.linalg.norm(x1))
		else:
			u,s,v = np.linalg.svd(x1)
			eg1 = u[:,np.argmax(s)]
			eigengenes.append(eg1)

	return clusters, eigengenes

def random_graph(n):
	A = np.zeros((n,n))
	for i in range(n):
		for j in range(n):
			if i > j:
				A[i,j] = np.abs(np.random.random())
				A[j,i] = A[i,j]
	return A

def dist_matrix(X, metric = 'euclidean', epsilon = 0):
	m= X.shape[1]
	Dist = np.zeros((m,m));
	for i in range(m):
		for j in range(m):
			Dist[i,j] = np.linalg.norm(X[:,i] - X[:,j]);
	return Dist


def check_scale_free(A, tolerance = .5, plot_it = False, exp = 2.5):
	#BROKEN
	n,m = A.shape
	A1 = np.zeros((n,n))
	#threshold
	for i in range(n):
		for j in range(n):
			if A[i,j] > 0:
				A1[i,j] = 1
	#degree vector
	D = np.sum(A1, axis = 0).astype('int')
	tmp = np.unique(D)
	kk = tmp.size
	deg_prob = np.zeros(kk)
	for i in range(len(tmp)):
		deg_prob[i] = np.count_nonzero(D == tmp[i])/n
	error = []
	if len(tmp)-np.count_nonzero(tmp) != 0:
		print('The graph is disconnected')
		return False
	pwr_dist = tmp**(-2.5)
	if kk == 1:
		print('Uniform degree')
		return False
	#fix this
	npwr_dist = pwr_dist/np.linalg.norm(pwr_dist)
	ndeg_prob = deg_prob/np.linalg.norm(deg_prob)
	print(npwr_dist)
	print(ndeg_prob)
	print(np.cross(npwr_dist,ndeg_prob))
	error1 = np.abs(np.cross(npwr_dist,ndeg_prob))
	error = error1 -1
	print(error)
	if error < tolerance:
		return True
	else:
		return False

	



def sim2dist(S):
	'''
	A function that converts similarity to distance

	inputs: similarity matrix
	outputs: distance matrix
	'''
	n= S.shape[0];
	D = np.zeros((n,n));
	for i in range(n):
		for j in range(n):
			if i != j:
				#this formula comes from Mardia's Analysis book
				if 1.00001 >= S[i,j] and 1 < S[i,j]:
					D[i,j] = 0
				elif S[i,j] > 1.00001:
					print('similarity is not between zero and one')
					print('setting distance equal to zero')
					D[i,j] = 0
				else:
					D[i,j]= np.sqrt(2*(1-S[i,j]));
	return D;



#BROKEN
def laplace_partition(A, fiedler = True, k = 1):
	'''
	A function that partitions the data using the graph Laplacian

	inputs: 1) adjacency matrix
					represents a directed weighted graph of the data
			 2) fiedler
					use the laplacian or normalized laplacian
			 3) k
					the use the eigenvector associated with the kth smallest eigenvalue 
	outputs: Classes A and B and C
					a partition of the nodes of the graph of A into two sets 
					via removing zero characteristically valuated nodes
					classa C is 0 valuated nodes
	'''
	n,m = A.shape;
	D = np.zeros((m,m));
	for i in range(m):
		D[i,i] = sum(A[i,:])
	if fiedler == True:
		L = D-A;
	else:
		snD = np.zeros((m,m));
		for i in range(m):
			D[i,i] = sum(A[i,:])
			snD[i,i] = 1/np.sqrt(D[i,i])
		L = snD @ (D-A) @ snD
	

	#generate eigenvalues and eigenvectors
	# if L.size > 4:
	# 	Evals, Evecs= linalg.eigsh(L,2,which = 'SM')
	# else:
	Evals, Evecs= np.linalg.eigh(L)

	#find the minimum (non-zero) eigenvalues
	positive_evals = np.zeros(Evals.size)
	for i in range(Evals.size):
		#only look at positive evals
		if Evals[i] > .00000000001:
			positive_evals[i] = Evals[i];
		else:
			positive_evals[i] = np.max(Evals);
	point = np.argmin(positive_evals);
	# point = np.argmax(Evals)


	#segment data using (+) and (-) components of the evec associated with the chosed eigenvalue
	raw_genes_in_class = np.zeros(m);
	for i in range(m):
		if Evecs[i,[point]] >= 0:
			raw_genes_in_class[i] = 0;
		elif Evecs[i,[point]] < 0:
			raw_genes_in_class[i] = 1;
	classA = [];
	classB = [];
	classA = np.argwhere(raw_genes_in_class==0);
	classB = np.argwhere(raw_genes_in_class==1);
	#also return the maximum absolute value of the fiedler vector

	return classA,classB;



def cluster_laplace(A, clst_adj, nodes, min_clust_sz, clst_node, all_clusters_node, fiedler_switch =True):
	'''
	A recursive function that clusters the graph using laplace partitiions

	inputs: 1) adjacency matrix
					represents a directed weighted graph of the data
			 2) clst_adj
					the cluster adjacancy matrices, just pass in []
			 3) nodes
					pass in a numpy array with the numbers of all the nodes (eg 0,...,30)
			 4) min_clust_sz
					stop cutting when the cluster size drops below this value
					the cut before the cluster size drops below this value are the returned clusters
			 5) clst_node
					the cluster nodes, just pass in []		
			 6) all_clusters_node
			 		A list of arrays containing the nodes from each cluster 
	outputs: none
	'''
	#partition the data using the fiedler vector
	N1,N2 = laplace_partition(A,fiedler_switch,1);
	#sizes of the clusters
	s1 = N1.size;
	s2 = N2.size;
	#nodes in each cluser
	nodes1 = np.zeros((1,s1));
	nodes2 = np.zeros((1,s2));
	if s1 > 0:
		for i in range(s1):
			nodes1[0,i] = nodes[0,N1[i]];
	if s2 > 0:
		for i in range(s2):
			nodes2[0,i] = nodes[0,N2[i]];
	#adjacency matrix for each cluster
	A1 = np.zeros((s1,s1));
	A2 = np.zeros((s2,s2));
	for i in range(s1):
		for j in range(s1):
			A1[i,j] = A[N1[i],N1[j]];
	for i in range(s2):
		for j in range(s2):
			A2[i,j] = A[N2[i],N2[j]];
	#add this cluster of nodes to the list of nodes
	all_clusters_node.append(nodes);
	#store the final clusters and their adjacency matrices
	if s1 < min_clust_sz or s2 < min_clust_sz:
		clst_adj.append(A);
		clst_node.append(nodes)
	#if we are not done, recurse
	if s1 >= min_clust_sz and s2 >= min_clust_sz:
		cluster_laplace(A1, clst_adj, nodes1, min_clust_sz, clst_node, all_clusters_node);
		cluster_laplace(A2, clst_adj, nodes2, min_clust_sz, clst_node, all_clusters_node);





def displaygraph(A,node_sizes,labels = False,layout = 'shell', plt_name = 'new_graph.png'):
	'''
	A function that plots the graph

	inputs: 1) adjacency matrix
					represents a directed weighted graph of the data
			 2) labels
			 3) layout
					shell- plots the graph in a circle, only plots largest connected component
					circular- plots the graph in a circle, only plots largest connected component
					spectral- plots the graph using two eigenvectors of laplacian as coordinates
					spring- plots graph so we have the smallest number of crossing edges
	outputs: plots the graph (no return values)
	'''
	#will not display individual nodes (yet) :)
	graph = [];
	it = np.arange(A[:,1].size)
	for i in it:
		for j in it:
			if  i > j and A[i,j] !=0:
				graph.append((i,j))

	# extract nodes from graph
	nodes = set([n1 for n1, n2 in graph] + [n2 for n1, n2 in graph])

	# create networkx graph
	G=nx.Graph()

	# add nodes
	for node in nodes:
		G.add_node(node)

	# add edges
	for edge in graph:
		G.add_edge(edge[0], edge[1])

 	# draw graph
	if layout =='shell':
		pos = nx.shell_layout(G)
	elif layout =='circular':
		pos = nx.circular_layout(G)
	elif layout == 'spring':
		pos = nx.spring_layout(G)
	elif layout == 'spectral':
		pos = nx.spectral_layout(G)

	if labels == True:
		nx.draw(G, pos, node_size = node_sizes)
		nx.draw_networkx_labels(G, pos)
	else:
		nx.draw(G, pos, node_size = node_sizes)

	# show graph
	#plt.show()
	plt.savefig(plt_name)



def connected_components(A):
	'''
	A function that returns the number of connected components

	inputs: adjacency matrix
					represents a directed weighted graph of the data
	outputs: number of connected components
	'''
	components = 0;

	#calculate the normalized laplacian
	n,m = A.shape;
	D = np.zeros((m,m));
	for i in range(m):
		D[i,i] = sum(A[i,:]);
	L = (D-A);

	#generate eigenvalues and eigenvectors
	# if L.size > 20:
	# 	Evals, Evecs= linalg.eigsh(L,2,which = 'SM')
	# else:
	Evals, Evecs= np.linalg.eigh(L)
	print(Evals)

	#the number of connected components is the number of evals that are close to 0
	for ev in Evals:
		if -.000000000001 <= ev  and ev <= .000000000001:
			components = components +1;

	return components;


def plot_spectrum(A, lbl = 'line'):
	'''
	A function that plots spectrum of the laplacian of the graph

	inputs: adjacency matrix
					represents a directed weighted graph of the data
	outputs: none
	'''

	#calculate the laplacian
	n,m = A.shape
	D = np.zeros((m,m))
	for i in range(m):
		D[i,i] = sum(A[i,:])
	L = D-A

	#generate eigenvalues and eigenvectors
	Evals, Evecs= np.linalg.eig(L);

	sorted_evals = np.sort(Evals);

	plt.plot(sorted_evals,label=lbl)
	plt.legend();



def cluster_centers(A, clst_adj, clst_node):
	'''
	A function that finds the center of each cluster using degree centrality

	inputs: 1) adjacency matrix
					represents a directed weighted graph of the data
			 2) clst_adj
					a list of the adjacency matrices of each cluster
			 5) clst_node
					a list of the nodes in each cluster
	outputs:1) newA
					the new center adjacancy matrix
			 2) newN
					the new center nodes
	'''

	nAsz = len(clst_adj);
	newN= np.zeros(nAsz);
	newA = np.zeros((nAsz,nAsz));

	#count the weighted degree of each node in each cluster
	for ii in range(nAsz):
		Aclass = clst_adj[ii]
		Nclass = clst_node[ii]
		Asz = clst_node[ii].size
		score = np.zeros(Asz);
		for i in range(Asz):
			for j in range(Asz):
				if j != i:
					score[i] += Aclass[i,j];

		#store the winning node in the ii-th cluster
		newN[ii] = Nclass[0,np.argmax(score)];

	newN.sort()

	#store the winning node adjacency matrix
	for i in range(nAsz):
		for j in range(nAsz):
			newA[i,j]= A[int(newN[i]),int(newN[j])]

	return newA, newN;


#in construction
def plot_dendrogram(all_clusters_node, A, X, clst_dst = 'dumb', fname = 'generated_dendrogram.png', title='Dendrogram'):
	
	'''
	A function that generates a dendrogram

	inputs: node clusters
				generally the output of cluster laplace
			A
				adjacency matrix
			clst_dst
				the distance between clusters (default dumb) eventually implement others
	outputs: (none) plots the dendrogram
			Saves said plot in the current directory as generated_dendrogram.png
	'''

	n,m = A.shape
	sorted_clusters = [];
	l = m+1;
	for j in range(m):
		for i in range(len(all_clusters_node)):
			if all_clusters_node[i].size== j and j!=1:
				sorted_clusters.append(all_clusters_node[i][0,:]);
			if j ==1 and i <m:
				sorted_clusters.append(np.array([i]));
	sorted_clusters.append(np.arange(m));

	sz = len(sorted_clusters)-1;

	subsets = np.zeros(len(sorted_clusters));
	for i in range(len(sorted_clusters)):
		for j in range(len(sorted_clusters)):
			if j > i:
				if set(sorted_clusters[i].tolist()).issubset(sorted_clusters[j].tolist())== True:
					print(sorted_clusters[i])
					print(sorted_clusters[j])
					print(j)
					print('------------------')
					subsets[i] = j;
					break;

	print(sorted_clusters)
	subsets_lst = subsets.tolist()
	tmp = np.zeros((len(sorted_clusters),4));
	tmp.fill(-1);
	for j in range(len(sorted_clusters)):
		if j in subsets:
			if tmp[j,0] < 0:
				if sorted_clusters[np.where(subsets == j)[0][0]].size == 1:
					tmp[j,0] = sorted_clusters[np.where(subsets == j)[0][0]][0];
				else:
					tmp[j,0] = np.where(subsets == j)[0][0];
			else:
				if  j >=m:
					if np.where(subsets == j)[0][1] <= sz:
						if sorted_clusters[np.where(subsets == j)[0][1]].size == 1:
							tmp[j,1] = sorted_clusters[np.where(subsets == j)[0][1]][0];
						else:
							tmp[j,1] = np.where(subsets == j)[0][1];
			tmp[j,3] = len(sorted_clusters[j]);
	for j in range(len(sorted_clusters)):
		if j in subsets:
			if tmp[j,0] < 0:
				if sorted_clusters[np.where(subsets == j)[0][0]].size == 1:
					tmp[j,0] = sorted_clusters[np.where(subsets == j)[0][0]][0];
				else:
					tmp[j,0] = np.where(subsets == j)[0][0];
				
			else:
				if  j >=m:
					if np.where(subsets == j)[0][1] <= sz:
						if sorted_clusters[np.where(subsets == j)[0][1]].size == 1:
							tmp[j,1] = sorted_clusters[np.where(subsets == j)[0][1]][0];
						else:
							tmp[j,1] = np.where(subsets == j)[0][1];

	Z = np.zeros((tmp.shape[0]-m,4));

	Z = tmp[m:,:]

	if clst_dst == 'dumb':
		Z[:,2] = Z[:,3];
	elif clst_dst == 'group_average':
		Dist = sim2dist(A);
		for ii in range(m-1):
			jj = m + ii;
			cl1 = sorted_clusters[np.where(subsets == jj)[0][0]]
			cl2 = sorted_clusters[np.where(subsets == jj)[0][1]]
			sz1 = cl1.size
			sz2 = cl1.size
			temp = []
			for i in range(sz1):
				for j in range(sz2):
					temp.append(Dist[int(cl1[i]),int(cl2[j])])
			Z[ii,2] = sum(temp)/len(temp)
	elif clst_dst == 'cut_edges':
		for ii in range(m-1):
			jj = m + ii;
			# if not list(np.where(subsets == jj)[0]):
			# 	print('EMPPPPTY')
			# 	print('m')
			# else:
			cl1 = sorted_clusters[np.where(subsets == jj)[0][0]]
			cl2 = sorted_clusters[np.where(subsets == jj)[0][1]]
			sz1 = cl1.size
			sz2 = cl2.size
			Z[ii,2] = np.log(sz1*sz2+1)

		for i in range(m-1):
			if Z[i,0] >m-1:
				Z[i,2] = Z[i,2]+Z[int(Z[i,0])-m,2]
			if Z[i,1] >m-1:
				Z[i,2] = Z[i,2]+Z[int(Z[i,1])-m,2]
		# Z[:,2] = np.log(Z[:,2]+1)

	elif clst_dst == 'max':
		Dist = sim2dist(A);
		for ii in range(m-1):
			jj = m + ii;
			cl1 = sorted_clusters[np.where(subsets == jj)[0][0]]
			cl2 = sorted_clusters[np.where(subsets == jj)[0][1]]
			sz1 = cl1.size
			sz2 = cl2.size
			temp = []
			for i in range(sz1):
				for j in range(sz2):
					temp.append(Dist[int(cl1[i]),int(cl2[j])])
			Z[ii,2] = max(temp)

	elif clst_dst == 'min':
		Dist = sim2dist(A);
		for ii in range(m-1):
			jj = m + ii;
			cl1 = sorted_clusters[np.where(subsets == jj)[0][0]]
			cl2 = sorted_clusters[np.where(subsets == jj)[0][1]]
			sz1 = cl1.size
			sz2 = cl2.size
			temp = []
			for i in range(sz1):
				for j in range(sz2):
					temp.append(Dist[int(cl1[i]),int(cl2[j])])
			Z[ii,2] = 2-min(temp)

	print(Z)


	# elif cst_dst == 'UPMGA':
	# 	D = sim2dist(A)
	# 	tmp = 0;
	# 	for k in range(30- sorted_clusters):
	# 		for i in sorted_clusters[k][0]: 
	# 			tmp = tmp + A[int(i),int(j)] 
	# 		tmp = tmp/(all_clusters_node[k][0].size)

	# fig = plt.figure(figsize=(8, 8))
	# dn = dendrogram(Z)
	# plt.xlabel('genes')
	# plt.ylabel('cluster distance')
	# plt.show()

	fig = pylab.figure(figsize=(8,8))
	ax1 = fig.add_axes([0.07,0.03,0.26,0.88])
	Z = dendrogram(Z,orientation='left')
	ax1.set_xticks([])
	ax1.set_yticks([])

	axmatrix = fig.add_axes([0.34,0.03,0.6,0.88])
	fig.suptitle(title)
	idx1 = Z['leaves']
	print(idx1)
	X = X[:,idx1].T
	im = axmatrix.matshow(X, aspect='auto', origin='lower', cmap=pylab.cm.YlGnBu)
	cbar = fig.colorbar(im)
	axmatrix.set_xticks([])
	axmatrix.set_yticks([])

	#fig.show()

	pylab.savefig(fname)

	#make dendrogram the other way:

	#B = 1-A
	#for i in range(30):
	#	B[i,i] = 0
	#distArray = ssd.squareform(B)
	#Z = linkage(distArray, 'ward')
	#fig = plt.figure(figsize=(25, 10))
	#dn = dendrogram(Z)
	#plt.show()	



def load_data(dataset, tid, shed = False, lbl = True, ratio = .8):
	'''
	loads the data into a numpy array

	inputs: 1) dataset
					which dataset to use? only works for duke prospective cytokine and
					pathway 404 from gse 73072
			 2) tid
					the time stamp
	        3) shed
					bool, true if you want to seperate into shedders and nonshedders
			 4) lbl
					bool, true of you want shedders, false if you dont want shedders
			 5) ratio
					the ratio for the impute progam, tunable. impute throws out nan data and chooses how to replace it
	outputs: the data matrix
	'''
	#using duke cytokine
	if dataset == 'duke_prosp':
		ccd = calcom.io.CCDataSet('./data/ccd_duke_prospective_cytokine_plasma_v2.h5');
		data_raw = np.log(imp.impute_llod_nans(ccd.generate_data_matrix().T,ratio).T)
	#using pathway 404
	elif dataset == 'pw404':
		ccd = calcom.io.CCDataSet('./data/ccd_gse73072_geneid.h5')
		dt = genfromtxt('./data/pathway404.csv',delimiter = ',')
		data_raw = dt[1:,1:]

	if shed == True:
		labels = ccd.generate_labels('shedding', make_dict = False)
		dta = data_raw[np.intersect1d(np.where(labels ==lbl), ccd.find('time_id', tid))]
	else:
		dta = data_raw[ccd.find('time_id', tid)]
	X = np.log(dta)
	#fix nans and zeros
	#X = np.log(np.transpose(imp.impute_llod_nans(np.transpose(X_raw),ratio)));
	return X


def count_bin_size(bins, ccd):
	sizes = []
	for t in bins: 
		q1 = {'time_id': t, 'shedding':True} 
		idx1 = ccd.find(q1) 
		sizes.append(idx1.size)
	return sizes







