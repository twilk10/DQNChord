from pprint import pprint

def binarySearch(alist, item):
    """
    binary search given alist and item to search for
    returns is element was found, otherwise return the last midpoint before search failed
    last midpoint is always either the predessecor or succesor to item (if the item is not found)
    """
    first = 0
    last = len(alist)-1
    found = False
    
    while first<=last and not found:
        midpoint = (first + last)//2
        if alist[midpoint] == item:
            found = True
        else:
            if item < alist[midpoint]:
                last = midpoint-1
            else:
                first = midpoint+1
    
    return found, midpoint

def successor(L, n):
    """
    Given a sorted list of ints L, and an integer n, return the smallest element of L that
    is larger than or equal to n. If none exist, return None
    """
    found, midpoint = binarySearch(L,n)
    if L[midpoint] >= n:
        return L[midpoint]
    if L[midpoint] < n:
        if midpoint==len(L)-1:
            return None
        return L[midpoint+1]


def get_fingers(nodes, m):
    """
    build finger table, given a list of nodes, and m
    """
    nodes_augment = nodes.copy()
    nodes_augment = sorted(nodes_augment)
    maxnum = pow(2,m)
    # wraparound first node to accomodate nodes clockwise of n when n is larger than the largest valued node
    nodes_augment.append(nodes_augment[0]+maxnum)
    
    finger_table = {}
    for node in nodes: 
        fingers = [None]*m
        for i in range(m):
            j = (node + pow(2,i))%maxnum
            fingers[i] = successor(nodes_augment,j)%maxnum
        finger_table[node] = fingers.copy()
        
    return finger_table

def get_key_loc(nodes, k, m):
    """
    given a key location k, and ring of size 2^m, and list of nodes, give the location of the key in the DHT
    """
    nodes_augment = nodes.copy()
    nodes_augment = sorted(nodes_augment)
    maxnum = pow(2,m)
    # wraparound first node to accomodate nodes clockwise of n when n is larger than the largest valued node
    nodes_augment.append(nodes_augment[0]+maxnum)
    return successor(nodes_augment, k)%maxnum


def get_query_path(nodes, k, n, m):
    """
    compute the quey path, given a list of nodes in the DHT, the key k to be found, origin node n, and ring of size 2^m
    """
    nodes_copy = nodes.copy()
    nodes_copy = sorted(nodes_copy)
    
    k_loc = get_key_loc(nodes_copy, k, m) # location of key k
    fingers = get_fingers(nodes_copy, m) # finger table
    query_path = [n] # start the path with origin node
    
    while(True): #keep going until the correct key location is found
        
        if k_loc==n: # return the path if correct location found
            return query_path

        f = fingers[n]
        
        i = 0
        x = f[i]
        
        if k_loc > n: 
            # if the clockwise path between the origin node (n) and the key location (k_loc) DOES NOT include 0, all the nodes that
            # lie in that path (x) must satisfy k_loc> x >n
            while (x < k_loc) and (x > n):
                i += 1
                if i == len(f):
                    break
                x = f[i]
                
        if k_loc < n: 
            # if the clockwise path between the origin node and the key location includes 0, all the nodes that
            # lie in that path (x) must satisfy (k_loc < x) && (n < x) OR (k_loc > x) && (n > x)
            while ( ((x > k_loc) and (x > n)) or ((x < k_loc) and (x < n)) ):
                i += 1
                if i == len(f):
                    break
                x = f[i]
        
        # update next node to query, and append to path
        if i==0:
            n = f[0]
        else:
            n = f[i-1]
            
        query_path.append(n)
    
if __name__=="__main__":
    
    # Chord DHT as described in Lecture 5 of week 3 of the Coursera Cloud Computing Concepts Part 1 Course
    
    nodes = [112, 96, 80, 16, 32, 45] # nodes in the P2P system
    m = 7 # Ring is of size 2^m
    print('Chord DHT with peers {nodes}, and m = {m} \n'.format(nodes=nodes, m=7))
    
    print('The finger table for this DHT is: ')
    pprint(get_fingers(nodes, m))
    print()
    
    file_key = 42
    print('File location for file key {file_key} is at node {file_key_location} \n'.format(
        file_key=file_key, file_key_location=get_key_loc(nodes, file_key, m)
    ))
    query_node = 80
    print('Query path for file {file_key} with origin node {query_node} : {query_path}'.format(
        file_key=file_key, query_node=query_node, query_path=get_query_path(nodes, file_key, query_node, m)
    ))
    query_node = 96
    print('Query path for file {file_key} with origin node {query_node} : {query_path}'.format(
        file_key=file_key, query_node=query_node, query_path=get_query_path(nodes, file_key, query_node, m)
    ))
    query_node = 45
    print('Query path for file {file_key} with origin node {query_node} : {query_path}'.format(
        file_key=file_key, query_node=query_node, query_path=get_query_path(nodes, file_key, query_node, m)
    ))
    