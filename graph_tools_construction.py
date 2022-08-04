import numpy as np
# from sklearn import datasets, linear_model
from sklearn import linear_model
import scipy.cluster.hierarchy as sch
import networkx as nx
# import scipy.spatial.distance as ssd
# import sklearn
# from numpy import genfromtxt
# import sklearn.metrics as sk 
# import matplotlib
# include this for katrina
# matplotlib.use('Agg')
from matplotlib import pyplot as plt
import pylab
# from scipy.sparse import linalg 

from scipy.spatial.distance import squareform
import pandas as pd


'''
To Do:
    -plotting with weights is weird in displaygraph
    -normalized cut for linkage matrix
    -speed up eval calculation
    -partial correlation???
    -mutual information score for adjacency matrix
    -dynamic cut
    -subspace distances
'''



def adjacency_matrix(X, msr = 'parcor', epsilon = 0, h_k_param = 2, negative = False, h_k_ord = 2):
    '''
    A function that builds an adjacecny matrix out of data using two methods.

    Nodes are columns of the data matrix X!

    inputs: data matrix 
                a numpy array with n rows m columns (m data points living in R^n)
            msr
                a string for method for calculating distance between data points 
                corrolation or heatkernel or partial correlation
            epsilon
                a number that is a user-parameter that determines will disconnect 
                all points that are further away or less corrolated than epsilon
            h_k_param
                a number for the heat parameter for the heat kernel similarity measure
            weighted
                a boolean that creates a weighted matrix if true
            negative 
                a boolean to include negative correlations? (default is False)
    outputs: adjacency matrix
                    represents a directed weighted graph of the data (dimensions m x m)
    '''
    n,m = X.shape

    if msr == 'correlation':
        norms = np.repeat(np.expand_dims(np.linalg.norm(X, axis=0),axis= 0), n, axis=0)
        norms[np.where(norms==0)] = 1 #so we don't divide by 0s
        normalized_X = X/norms
        AdjacencyMatrix = normalized_X.T @ normalized_X - np.eye(m)
        if not negative:
            AdjacencyMatrix  = np.abs(AdjacencyMatrix)
        AdjacencyMatrix[np.where(AdjacencyMatrix > 1)] = 1


    elif msr == 'heatkernel':
        # Diffs = np.repeat(np.expand_dims(X,axis = 1), m, axis=1)-np.repeat(np.expand_dims(X,axis = 2), m, axis=2)
        # DistanceMatrix = np.linalg.norm(Diffs, axis= 0)
        # AdjacencyMatrix = np.exp(-(DistanceMatrix ** 2 /(h_k_param ** 2)))

        AdjacencyMatrix = np.zeros((m,m))

        for i in range(m):
            for j in range(i+1,m):
                AdjacencyMatrix[i,j] = np.exp(-(np.linalg.norm( X[:,i]-X[:,j], ord = h_k_ord)**2 )/(2*h_k_param))
                AdjacencyMatrix[j,i] = AdjacencyMatrix[i,j].copy()

    #old partial correlation
    # elif msr == 'parcor':
    #     # create linear regression object 
    #     reg = linear_model.LinearRegression()

    #     vis = list(range(m))

    #     for i in range(m):
    #         for j in range(m):
    #             if i > j:

    #                 #compute projections (aka linear regressions)
    #                 vis.remove(i)
    #                 vis.remove(j);					
    #                 reg.fit(X[:,vis], X[:,i]); 
    #                 x_hat_i = reg.predict(X[:,vis])
    #                 reg.fit(X[:,vis], X[:,j])
    #                 x_hat_j = reg.predict(X[:,vis])

    #                 #compute residuals
    #                 Y_i = X[:,i] - x_hat_i
    #                 Y_j = X[:,j] - x_hat_j

    #                 Y_in = np.linalg.norm(Y_i)
    #                 Y_jn = np.linalg.norm(Y_j)


    #                 if Y_in == 0 or Y_jn == 0:	
    #                     tmp = 0

    #                 else:
    #                     tmp = np.dot(Y_i, Y_j)/(Y_in*Y_jn)
    #                 if negative == True:
    #                     PC = tmp
    #                 else:
    #                     PC = np.abs(tmp)
    #                     if epsilon != 0 and PC < epsilon:
    #                         #get rid of all edges that have weights less than epsilon
    #                         PC = 0
    #                 #why are we getting partial correlations of 1?
    #                 if PC > 1 and PC < 1.0000001:
    #                     PC = 1


    #                 AdjacencyMatrix[i,j] =PC
    #                 AdjacencyMatrix[j,i] =PC
    #                 vis = list(range(m))

    if epsilon > 0:
        AdjacencyMatrix[AdjacencyMatrix < epsilon] = 0

    #force diagonal 0
    np.fill_diagonal(AdjacencyMatrix, 0)

    return AdjacencyMatrix



def zobs(X, class_labels, negative = False):
    '''
    A function that builds a zobs matrix where we take z_class0-z_class1

    Nodes are columns of the data matrix X!

    inputs: data matrix 
                a numpy array with n rows m columns (m data points living in R^n)
            class_labels
                a numpy array of class labels that are ints 0s or 1s
            negative 
                a boolean to include negative correlations? (default is False)
    outputs: adjacency matrix
                    represents a directed weighted graph of the data (dimensions m x m)
    '''
    idx_class0 = np.where(class_labels == 0)[0]
    idx_class1 = np.where(class_labels == 1)[0]

    X_class0 = X[idx_class0,:]
    X_class1 = X[idx_class1,:]

    r_class0 =  adjacency_matrix(X_class0, msr = 'correlation', negative = True)
    r_class1 =  adjacency_matrix(X_class1, msr = 'correlation', negative = True)

    if np.sum(np.abs(r_class0) == 1) > 0 or np.sum(np.abs(r_class1) == 1) > 0:
        print('correlation of -1 or 1 for some genes. setting correlations to .99999999')

    r_class0[r_class0==1] = 0.99999999
    r_class1[r_class1==1] = 0.99999999

    r_class0[r_class0==-1] = -0.99999999
    r_class1[r_class1==-1] = -0.99999999

    z_class0 = np.arctanh(r_class0)
    z_class1 = np.arctanh(r_class1)

    if len(idx_class0) < 4 or len(idx_class1) < 4:
        print('potential divide by 0 in zobs!')
    zobs = (z_class0-z_class1) / np.sqrt( 1/(len(idx_class0)-3) + 1/(len(idx_class1)-3))

    if not negative:
        zobs = np.abs(zobs)

    return zobs


def wgcna(x, beta = 1, den_gen = 'average', threshold = 0, den_fname = 'wgcna_den.png', den_title = 'WGCNA Shedder Data at 60 Hours After Infection', tom = True):
    '''
    Basic WGCNA implementation	
    '''
    n,m = x.shape

    a = adjacency_matrix(x, msr = 'correlation')

    if tom == True:
        #topological overlap measure
        w = np.zeros((m,m)) 
        for i in range(m): 
            for j in range(m): 
                if i > j: 
                    l = a[:,i] @ a[j,:]
                    k = np.min((np.sum(a[:,i]),np.sum(a[:,j])))
                    w[i,j] = (l+a[i,j])/(k+1-a[i,j])
                    w[j,i] = w[i,j]
                elif i == j:
                    w[i,i]=1
    else:
        w = a
        for i in range(m):
            w[i,i] = 1
    print(w)

    d = 1-w

    sd = sp.spatial.distance.squareform(d)

    Z = sch.linkage(sd, den_gen)

    # dn = sp.cluster.hierarchy.dendrogram(Z)
    fig = pylab.figure(figsize=(8,8))
    ax1 = fig.add_axes([0.07,0.03,0.26,0.88])
    Z = sch.dendrogram(Z,orientation='left')
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
    Z = sch.linkage(sd, den_gen)

    return sd, Z

def cluster_den(Z,x,cut_height = .7):
    '''
    Also for WGCNA
    '''
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
    '''
    Generate an adjacency matrix of a random graph with n nodes.
    '''

    A = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            if i > j:
                A[i,j] = np.abs(np.random.random())
                A[j,i] = A[i,j]
    return A

def erdos_reyni(n,m, seed):
    '''
    Generate an adjacency matrix of an unweighted Erdos Reyni graph 
    with n nodes and m edges with seed
    '''

    G = nx.gnm_random_graph(n, m, seed=seed)
    A = nx.to_numpy_array(G, nodelist=list(np.arange(n)))

    return A



def laplace_partition(A, fiedler = True, k = 1):
    '''
    A function that partitions the data using the graph Laplacian

    CAUTION--- NETWORK OF A MUST BE CONNECTED to do this calculation!!!

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
    #make graph laplacian
    m = A.shape[0]
    
    incident_edges = np.sum(A,axis = 1)
    #degree matrix
    D = np.diag(incident_edges)*np.eye(m)

    if fiedler:
        L = D-A
    else:
        #normalized
        snD = np.diag(1/np.sqrt(incident_edges))*np.eye(m)
        L = snD @ (D-A) @ snD

    #speed this up?
    #tried calculating one eigenvalue and using lsqsolve to find evec, it's slower
    evals, evecs = np.linalg.eigh(L)

    #find fiedler vector as smallest non-zero eigenvalue
    if len(evals) == 1:
        fiedler_eval = evals[0]
    else:
        fiedler_eval = np.sort(evals)[1]

    #find fiedler vector
    fiedler_vec = evecs[:,np.where(evals == fiedler_eval)[0]]
    
    if fiedler_vec.shape[1]>1:
        fiedler_vec = fiedler_vec[:,[0]]

    #sort nodes into two banks using sign of fiedler vector
    class_labels = np.zeros(m)
    class_labels[np.where(fiedler_vec < 0)[0]] = 1
    
    #return the indices of the nodes in these banks in two lists
    classA = np.argwhere(class_labels==0)
    classB = np.argwhere(class_labels==1)

    return classA, classB



def cluster_laplace(A, clst_adj, nodes, min_clust_sz, clst_node, all_clusters_node, fiedler_switch =True, stop_criteria = 'size'):
    '''
    A recursive function that clusters the graph using laplace partitiions
    
    CAUTION--- NETWORK OF A MUST BE COMPLETELY CONNECTED to do this calculation!!!

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
             7) fiedler_switch
             8) stop_criteria
    outputs: none
    '''
    #partition the data using the fiedler vector
    N1,N2 = laplace_partition(A,fiedler_switch,1)
    #sizes of the clusters
    s1 = N1.size
    s2 = N2.size
    #nodes in each cluser
    nodes1 = np.zeros(s1)
    nodes2 = np.zeros(s2)
    if s1 > 0:
        for i in range(s1):
            nodes1[i] = nodes[N1[i]]
    if s2 > 0:
        for i in range(s2):
            nodes2[i] = nodes[N2[i]]
    #adjacency matrix for each cluster
    A1 = np.zeros((s1,s1))
    A2 = np.zeros((s2,s2))
    for i in range(s1):
        for j in range(s1):
            A1[i,j] = A[N1[i],N1[j]]
    for i in range(s2):
        for j in range(s2):
            A2[i,j] = A[N2[i],N2[j]]
    #add this cluster of nodes to the list of nodes
    all_clusters_node.append(np.array([int(node) for node in nodes]))

    if stop_criteria == 'size':

        keep_going = s1 >= min_clust_sz and s2 >= min_clust_sz

        #store the final clusters and their adjacency matrices
        if not keep_going:
            clst_adj.append(A)
            clst_node.append(nodes)
        #if we are not done, recurse
        else :
            cluster_laplace(A1, clst_adj, nodes1, min_clust_sz, clst_node, all_clusters_node)
            cluster_laplace(A2, clst_adj, nodes2, min_clust_sz, clst_node, all_clusters_node)

    elif stop_criteria == 'weight':

        mx = np.ma.masked_array(A, mask=np.eye(A.shape[0]))
        if s1 > 1:
            mx1 = np.ma.masked_array(A1, mask=np.eye(s1))
        else:
            mx1 = A1.copy()
        if s2 > 1:
            mx2 = np.ma.masked_array(A2, mask=np.eye(s2))
        else:
            mx2 = A2.copy()

        med = np.ma.mean(mx)
        med1 = np.ma.mean(mx1)
        med2 = np.ma.mean(mx2)

        keep_going1 = med < med1
        keep_going2 = med < med2
        keep_going = keep_going1 or keep_going2

        #store the final clusters and their adjacency matrices
        if not keep_going:
            clst_adj.append(A)
            clst_node.append(nodes)
        #if we are not done, recurse
        elif keep_going1:
            cluster_laplace(A1, clst_adj, nodes1, min_clust_sz, clst_node, all_clusters_node)
        elif keep_going2:
            cluster_laplace(A2, clst_adj, nodes2, min_clust_sz, clst_node, all_clusters_node)
        
    elif stop_criteria == 'loo_svm':
        print('in construction')

    else:
        print('stop_criteria not recognized')
    
    


def embedgraph(A):
    '''
    Embedding using second and third smallest eigenvectors of the graph laplacian.

    Inputs:
        Numpy array of adjacency matrix
    '''
    m = A.shape[0]
    
    incident_edges = np.sum(A,axis = 1)
    #degree matrix
    D = np.diag(incident_edges)*np.eye(m)

    L = D-A

    _, evecs = np.linalg.eigh(L)

    plt.figure()
    plt.scatter(evecs[:,1], evecs[:,2], marker = 'x')
    plt.xlabel('Second Smallest Eigenvector')
    plt.ylabel('Third Smallest Eigenvector')
    plt.savefig('graph_embedding.png')



def displaygraph(small_A, labels, scores = [],  
                layout = 'spring', save_name = 'new_graph.png',
                remove_edges = True):
    '''
    A function that plots the graph

    inputs: 1) small_A
                    an adjacency matrix that represents a directed weighted graph of the data
             2) labels
                    dictionary with node numbers as keys and strings for node labels
             3) layout
                    shell- plots the graph in a circle, only plots largest connected component
                    circular- plots the graph in a circle, only plots largest connected component
                    spectral- plots the graph using two eigenvectors of laplacian as coordinates
                    spring- plots graph so we have the smallest number of crossing edges
             4) plt_name
                    a string for the filename and location of saved graph as png
    outputs: plots the graph (no return values)
    '''

    #get network centrality scores
    if len(scores) == 0:
        scores = centrality_scores(small_A, centrality = 'degree')
        scores = scores / np.max(scores)

    if remove_edges:
        #remove nodes less than median weight
        small_A[np.arange(len(scores)), np.arange(len(scores))] = 0
        # small_A = small_A/np.max(small_A)
        idx =  np.where(small_A < np.median(small_A))
        small_A[idx] = 0

    #make networkx graph object
    G = nx.from_numpy_matrix(np.round(np.matrix(small_A),2))

    #node labels using labels
    mapping = {}
    for i in range(len(labels)):
        mapping[i] = labels[i]

    print('hi')

    #relabel the nodes in the network
    G = nx.relabel_nodes(G, mapping)

    #use edge weights
    edges = G.edges()
    # weights = [G[u][v]['weight']*10 for u,v in edges]
    weights = [G[u][v]['weight'] for u,v in edges]

    # draw graph
    plt.figure()
    
    #choose node orientation
    if layout =='shell':
        pos = nx.shell_layout(G)
    elif layout =='circular':
        pos = nx.circular_layout(G)
    elif layout == 'spring':
        pos = nx.spring_layout(G)
    elif layout == 'spectral':
        pos = nx.spectral_layout(G)

    #draw edges and nodes
    edges = nx.draw_networkx_edges(G, pos, edge_color=weights, width=4,
                                edge_cmap=plt.cm.Oranges)
    nodes = nx.draw_networkx_nodes(G, pos, node_color='antiquewhite', 
                                node_size = scores*100)
    nx.draw_networkx_labels(G, pos, font_size=3, font_color='k')

    plt.colorbar(edges)
    plt.axis('off')

    #save the figure
    plt.savefig(save_name, dpi=1000)



def connected_components(A):
    '''
    A function that returns the number of connected components

    inputs: adjacency matrix
                    represents a directed weighted graph of the data
    outputs: number of connected components
    '''

    #calculate the laplacian
    m = A.shape[1]
    incident_edges = np.sum(A,axis = 1)
    #degree matrix
    D = np.diag(incident_edges)*np.eye(m)
    L = (D-A)

    #generate eigenvalues and eigenvectors
    # if L.size > 20:
    # 	Evals, Evecs= linalg.eigsh(L,2,which = 'SM')
    # else:
    Evals, Evecs= np.linalg.eigh(L)

    #the number of connected components is the number of evals that are close to 0
    components = np.sum(np.abs(Evals)<= .000000000001)

    return components


def plot_spectrum(A, lbl = 'line'):
    '''
    A function that plots spectrum of the laplacian of the graph

    inputs: adjacency matrix
                    represents a directed weighted graph of the data
    outputs: none
    '''

    #calculate the laplacian
    n,m = A.shape[0]
    incident_edges = np.sum(A,axis = 1)
    #degree matrix
    D = np.diag(incident_edges)*np.eye(m)
    L = D-A

    #generate eigenvalues and eigenvectors
    Evals, Evecs= np.linalg.eig(L)

    sorted_evals = np.sort(Evals)

    plt.plot(sorted_evals,label=lbl)
    plt.legend()



def cluster_centers(A, clst_adj, clst_node, centrality = 'degree'):
    '''
    A function that finds the center of each cluster using degree centrality

    inputs: 1) A
                    adjacency matrix represents a directed weighted graph of the data
            2) clst_adj
                    a list of the adjacency matrices of each cluster
            3) clst_node
                    a list of the nodes in each cluster
            4) centrality
                    a string for the centrality measure to use
    outputs:1) newA
                    the new center adjacancy matrix
             2) newN
                    the new center nodes
    '''

    nAsz = len(clst_adj)
    newN= np.zeros(nAsz)
    newA = np.zeros((nAsz,nAsz))

    # #count the weighted degree of each node in each cluster
    # for ii in range(nAsz):
    #     Aclass = clst_adj[ii]
    #     Nclass = clst_node[ii]
    #     Asz = clst_node[ii].size
    #     score = np.zeros(Asz)
    #     for i in range(Asz):
    #         for j in range(Asz):
    #             if j != i:
    #                 score[i] += Aclass[i,j]

    #calculate centrality score of each node in each cluster
    for ii in range(nAsz):
        score = centrality_scores(clst_adj[ii], centrality)    

        #store the winning node in the ii-th cluster
        newN[ii] = clst_node[ii][np.argmax(score)]

    newN.sort()

    #store the winning node adjacency matrix
    for i in range(nAsz):
        for j in range(nAsz):
            newA[i,j]= A[int(newN[i]),int(newN[j])]

    return newA, newN





def sim2dist(S):
    '''
    A function that converts similarity to distance

    inputs: similarity matrix
    outputs: distance matrix
    '''
    if not np.any(S <=1) and np.any(S >=0):
        print('similarity is not between zero and one')
        D = None
    else:
        D = np.sqrt(2*(1-S))
    return D



def linkage_matrix(all_clusters_node, A, clst_dst):
    '''
    A function that generates a linkage matrix similar to scipy method

    inputs: node clusters
                generally the output of cluster laplace
            A
                adjacency matrix
            clst_dst
                the string for the distance between clusters (default dumb) eventually implement others
                options are 'dumb' or 'avg_cut' or 'norm_cut'
                    'avg_cut' based off average cut problem but with a distance matrix. this results in a poorly structured dendrogram
                    'norm_cut' is unfinished
    outputs: (np.array) linkage matrix
    '''
    m = A.shape[0]
    all_clusters_node.sort(key=len)


    num_clusters = len(all_clusters_node)

    #find which clusters are subsets of other clusters
    #label each cluster using numbers beginning with number of nodes
    subsets = np.zeros(num_clusters-1)
    for i in range(num_clusters):  
        for j in range(i+1,num_clusters):
            small_set = set(all_clusters_node[i].tolist())
            big_set = set(all_clusters_node[j].tolist())
            if small_set.issubset(big_set) == True:
    #             print(all_clusters_node[i])
    #             print(all_clusters_node[j])
    #             print(j)
    #             print('------------------')
                subsets[i] = j
                break #removing this break statement would be great!

    
    Z_dim = len(all_clusters_node)-m
    
        
    Z = -1*np.ones((Z_dim,4))

    #this might be able to be done better
    for j in range(m,num_clusters,1):
        #current cluster row is at index m-j in linkage matrix

        #the indices of the smaller clusters that belong to cluster j
        cluster_idx = np.where(subsets == j)[0]

        #for the joining cluster
        joining_cluster = all_clusters_node[cluster_idx[0]]
        if joining_cluster.size == 1:
            #if the joining clutser is of size 1
            Z[j-m,0] = joining_cluster[0]
        else:
            #is the joining cluster is of size bigger than 1
            Z[j-m,0] = cluster_idx[0]

        #for the current cluster
        current_cluster = all_clusters_node[cluster_idx[1]]
        if current_cluster.size == 1:
            #if the current clutser is of size 1
            Z[j-m,1] = current_cluster[0]
        else:
            Z[j-m,1] = cluster_idx[1]


        #add the cluster size
        Z[j-m,3] = len(all_clusters_node[j])

    
    if clst_dst == 'dumb':
        Z[:,2] = Z[:,3].copy()
    
    #average cut size
    elif clst_dst == 'avg_cut':
        Dist = sim2dist(A)
        for ii in range(m-1):
            jj = m + ii
            idx = np.where(subsets == jj)[0]
            cl1 = all_clusters_node[idx[0]]
            cl2 = all_clusters_node[idx[1]]
            sz1 = cl1.size
            sz2 = cl2.size
            dist = 0
            for nodeA in cl1:
                for nodeB in cl2:
                    dist += Dist[nodeA,nodeB]
            Z[ii,2] = (dist/sz1 + dist/sz2)

    #normalized cut size
    elif clst_dst == 'norm_cut':
        print('in construction') 
    else:
        print('clst_dst not recognized')

    return Z



def cut_tree(Z, n_clusters = None, height = None):
    '''
    Cut a linkage matrix and return the clustering.

    inputs: Z
                a numpy array of a linkage matrix (see scipy.cluster.hierarchy.linkage_matrix for the format)
            n_clusters
                an integer or list of integers for the number of clusters
            height
                the height or lists of heights of the cut
    outputs:
            the_clustering
                numpy array of the cluster labels of the nodes in Z.  
                ith column corresponds with the ith entry of n_clusters or height
    '''
    the_clustering = sch.cut_tree(Z, n_clusters, height)
    
    return the_clustering
    
              
              
             
def plot_dendrogram(all_clusters_node, A, X, clst_dst = 'dumb', fname = 'generated_dendrogram.png', title='Dendrogram', just_dendrogram = False, split = 0):

    '''
    A function that generates a dendrogram

    inputs: all_clusters_node
                a list of indices for the node clustering
            A
                a numpy array of the adjacency matrix
            X
                a numpy array of the dataset, rows correspond to entries in A
            clst_dst
                the distance between clusters (default dumb)
            fname
                a string for the file path to save the plot
            title
                a string for the plot title
            just_dendrogram
                boolean, False for adding heatmap and horizontally oriented dendrogram
            split
                an integer, a horizontal line will be drawn between the split-1 and 
                split indexed pixels. this is useful for distinguishing between two 
                different classes in the node features
    outputs: (none) plots the dendrogram
            Saves said plot in the current directory as generated_dendrogram.png
    '''

    
    Z = linkage_matrix(all_clusters_node, A, clst_dst)
    
    
    if just_dendrogram:
        fig = pylab.figure(figsize=(8,8))
        Z_den = sch.dendrogram(Z, color_threshold=0)
    
    else:
        fig = pylab.figure(figsize=(8,8))
        ax1 = fig.add_axes([0.07,0.03,0.26,0.88])
        Z_den = sch.dendrogram(Z,orientation='left', color_threshold =0)
    #     ax1.set_xticks([])
        ax1.set_yticks([])

        axmatrix = fig.add_axes([0.34,0.03,0.6,0.88])
        fig.suptitle(title, fontsize=35)
        idx1 = Z_den['leaves']
        X = X[:,idx1].T
        im = axmatrix.matshow(X, aspect='auto', origin='lower', cmap=pylab.cm.YlGnBu)
        # if split > 0:
        #     axmatrix.axvline(split -.5, color = 'white')
        cbar = fig.colorbar(im)
        axmatrix.set_xticks([])
        axmatrix.set_yticks([])

    #fig.show()

    pylab.savefig(fname)
    
    return Z_den
    

    
    
def supra_adjacency(dataset, time_weight = 'mean', msr = 'parcor', epsilon = 0, h_k_param = 2, negative = False, weighted = True):
    '''
    Generates a supre-adjacency matrix from a list of data matrices.

    inputs: dataset
                a list of numpy arrays that are data matrices for the same dataset at times 1,2,3,...
            time_weight
                a number for type of edge weight to connect the same nodes between adjacent time steps
            msr
                a string for method for calculating distance between data points 
                corrolation or heatkernel or partial correlation
            epsilon
                a number that is a user-parameter that determines will disconnect 
                all points that are further away or less corrolated than epsilon
            h_k_param
                a number for the heat parameter for the heat kernel similarity measure
            weighted
                a boolean that creates a weighted matrix if true
            negative 
                a boolean to include negative correlations? (default is False)
                
    outputs: sA a numpy array that represents the supra-adjacency matrix
    '''
    node_list = []
    A = []
    for X in dataset:
        A1 = adjacency_matrix(X,msr,epsilon)
        node_list.append(np.arange(A1[0].shape[0]))
        A.append(A1)


    n,m = A[0].shape

    N = m*len(A)
    sA = np.zeros((N,N))
    for i in range(len(A)):
        for j in range(len(A)):
            if i == j:
                sA[i*m:(i+1)*m, i*m:(i+1)*m] = A[i]
            if i == j+1:
                #add in other time_weight schemes here
                if time_weight == 'mean':
                    time_weight1 = np.mean(A)
                else:
                    print('time weight not recognized')

                sA[i*m:(i+1)*m,j*m:(j+1)*m] = time_weight1*np.eye(m)

    return sA



def centrality_scores(A, centrality = 'large_evec', pagerank_d = .85, pagerank_seed = 1, stochastic = False, in_rank = False):
    '''
    A method for computing the centrality of the nodes in a network

    Note: a node has degree 5 if it has 5 edges coming out of it. 
    We are interested in out edges rather than in edges! 
    Page rank ranks nodes with out edges higher than nodes with in-edges.
    
    Inputs:
        A - a numpy array that is the adjacency matrix
        centrality - a string for the type of centrality
                     options are:
                         'largest_evec'
                         'page_rank'
                         'degree'
        pagerank_d - float, parameter for pagerank 
    Outputs:
        scores - a numpy array of the centrality scores for the nodes in the network
                 index in scores corresponds to index in A
    
    '''
    
    if centrality == 'large_evec':
        W,V = np.linalg.eig(A)
        scores = np.real(V[:,W.argmax()])

    elif centrality == 'degree':
        #sum by out edges
        degrees = np.sum(A,axis = 0)
        if A.shape[0] > 1:
            scores = degrees
        else:
            scores = np.array([0])
        
    elif centrality == 'page_rank':
        if not in_rank:
            A = A.T
        n = A.shape[0]
        if n == 1:
            scores = np.array([0])
        else:
            #in connections
            connected_idx = np.where(np.sum(A, axis = 0) != 0)[0]
            #connected_idx_out = np.where(np.sum(A, axis = 1) != 0)[0]
            #connected_idx = np.union1d(connected_idx_in, connected_idx_out)
            connected_A = A[:,connected_idx][connected_idx,:]
            n = len(connected_idx)
            if n <= 1:
                scores = np.array([0])
            else:
                M = np.zeros((n,n))
                for i in range(n): 
                    A_sum = np.sum(connected_A[:,i])
                    if A_sum == 0:
                        M[:,i] = connected_A[:,i]
                        # print('dangling nodes for page rank')
                    else:
                        M[:,i] = connected_A[:,i]/A_sum

                if stochastic:
                    #taken from da wikipedia
                    eps = 0.001

                    #new and fast
                    np.random.seed(pagerank_seed)
                    
                    v = np.random.rand(n, 1)
                    v = v / np.linalg.norm(v, 1)
                    err = 1
                    while err > eps:
                        v0 = v.copy()
                        v = (pagerank_d * M) @ v0 + (1 - pagerank_d) / n
                        err = np.linalg.norm(v - v0, 2)

                    #sanity check
                    big_M = (pagerank_d * M)  + np.ones((n,n))*(1 - pagerank_d) / n
                    v_check = big_M @ v
                    if not np.allclose(v_check, v, rtol=1e-05, atol=1e-08):
                        print('page rank not converged')
                else:
                    big_M = (pagerank_d * M)  + np.ones((n,n))*(1 - pagerank_d) / n
                    evals, evecs = np.linalg.eig(big_M)
                    dist_from_1 = np.abs(evals - 1)
                    idx = dist_from_1.argmin()
                    v = evecs[:,idx]
                    v = v/np.sum(v)
                    if np.abs(evals[idx]-1) > 1e-08:
                        print('page rank not converged')
                    
                connected_scores = v.flatten()

                scores = np.zeros(A.shape[0])
                scores[connected_idx] = connected_scores


        
    else:
        print('centrality type not recognized')
        
    return scores



def supra_adjacency_scores(sA, centrality, n_times, n_nodes): 
    '''
    A method for computing the centrality of the nodes in a time series network
    
    Inputs:
        sA - a numpy array that is a supra adjacency matrix
        centrality - a string for the type of centrality
                     options are:
                         'largest_evec'
                         'page_rank'
        n_times - the number of times points in the dataset
        n_nodes - the number of nodes at one time
    Outputs:
        scores - a numpy array of the centrality scores for the nodes in the network
                 index in scores corresponds to index in A
    
    '''
    N = sA.shape[0]

    scores = centrality_scores(sA, centrality)
       

    scores1 = np.zeros(n_nodes)
    for i in range(n_nodes):
        for j in range(n_times):
            scores1[i] += np.abs(scores[i+n_nodes*j])
            
    return scores1


def cluster_module_reps(A: np.array, labels: list, MEDissThres: float = .2):
    # A undirected adjacency matrix

    # divide A by max entry
    if len(np.where(A > 1)[0])> 0:
        A = A/np.amax(A)

    # Calculate dissimilarity of module eigengenes
    MEDiss = pd.DataFrame(1 - A, columns = labels, index = labels)

    # Cluster module eigengenes
    d = squareform(MEDiss, checks=False)
    METree = sch.linkage(d, method='average')

    plt.figure(figsize=(max(20, round(MEDiss.shape[1] / 20)), 10), facecolor='white')
    sch.dendrogram(METree, color_threshold=MEDissThres, labels=MEDiss.columns, leaf_rotation=90,
                leaf_font_size=8)


