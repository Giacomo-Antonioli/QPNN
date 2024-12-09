import numpy as np
import heapq
from collections import defaultdict

def MWM_process(graph, phi_0, iterations, alpha,beta,gamma, vert_decay,link_decay):
    # https://web.eecs.umich.edu/~pettie/papers/ApproxMWM-JACM.pdf
    import networkx as nx

    G=nx.Graph(graph)
    for u, v in G.edges():
        G[u][v]['weight']=0.0

    def Ren2(p):
        return -np.log(np.sum([pi**2 for pi in p.values()]))

    num_edges=G.number_of_edges()
    phi_t = phi_0.copy()
    vert_use_t={vt:0.0 for vt in G.nodes()}
    link_use_t={(ed if ed[1]>ed[0] else (ed[1],ed[0])):0.0 for ed in G.edges()}
    #link_t={ed:1./num_edges for ed in G.edges()}

    selected_edges_list = []
    probability_distributions=[phi_t.copy()]
    entropy_list = [Ren2(phi_t)]  # second Rényi entropy
    #probability_distributions=[(phi_t.copy(),link_t.copy())]
    #entropy_list = [(Ren2(phi_t),Ren2(link_t))]  # second Rényi entropy

    for it in range(iterations):
        #print(probability_distributions[-1])
        # Compute pairwise differences and create a max-heap
        edge_diffs = []
        min_w=+np.inf
        #last_verts=set() if len(selected_edges_list)==0 else {a for el in selected_edges_list[-1] for a in el}
        #print(last_verts)
        for ed in G.edges():
            u,v = ed
            if u>v:
                u,v=v,u
#            link_presence_penalty=1.0 if (len(selected_edges_list)>=1 and (u,v) in selected_edges_list[-1]) else 0.0
#            vert_presence_penalty=1.0 if (len(selected_edges_list)>=1 and ((u in last_verts) or (v in last_verts))) else 0.0
            #G[u][v]['weight']=1+np.log(1e-10+np.abs(phi_t[u]-phi_t[v]))-beta*link_use_t[ed]
            #G[u][v]['weight']=(phi_t[u]-phi_t[v])**2-(beta*(link_use_t[ed]+0.5*(vert_use_t[u]+vert_use_t[v])) if np.abs(phi_t[u]-phi_t[v])>1e-8 else 0.0)
            #G[u][v]['weight']=(phi_t[u]-phi_t[v])**2*(1-beta*0.5*link_use_t[(u,v)]-0.5*gamma*(vert_use_t[u]+vert_use_t[v]))
            G[u][v]['weight']=(phi_t[u]-phi_t[v])**2 
            if not (phi_t[u]==0.0 and phi_t[v]==0.0):
                #G[u][v]['weight']+=beta*(1.0-link_presence_penalty)+gamma*(1-vert_presence_penalty) #-beta*link_use_t[(u,v)]-gamma*(vert_use_t[u]+vert_use_t[v]))
                G[u][v]['weight']+=beta*(1.0-link_use_t[(u,v)])+gamma*(2-vert_use_t[u]-vert_use_t[v]) #-beta*link_use_t[(u,v)]-gamma*(vert_use_t[u]+vert_use_t[v]))
            #G[u][v]['weight']=(phi_t[u]-phi_t[v])**2-beta*(link_t[ed]-1./num_edges)
            #G[u][v]['weight']=max(0.,2+np.log(np.abs(phi_t[u]-phi_t[v])+1e-10)) #+np.exp(-(beta*(link_use_t[ed]+0.5*(vert_use_t[u]+vert_use_t[v])))) if np.abs(phi_t[u]-phi_t[v])>1e-8 else 0.0
            
#            if min_w>G[u][v]['weight']:
#                min_w=G[u][v]['weight']
#
#        for ed in G.edges():
#            u,v = ed
#            #G[u][v]['weight']=1+np.log(1e-10+np.abs(phi_t[u]-phi_t[v]))-beta*link_use_t[ed]
#            G[u][v]['weight']-=min_w
            
            #print(it, u, v, G[u][v]['weight'])
            # np.log(1e-10+np.abs(phi_t[u]-phi_t[v]))

        selected_edges = nx.algorithms.matching.max_weight_matching(G, maxcardinality=False)
        selected_edges={((el[1],el[0]) if el[1]<el[0] else (el[0],el[1]))for el in selected_edges}

        if len(selected_edges_list)>=1:
            selected_edges={el for el in selected_edges if el not in selected_edges_list[-1]}

        selected_edges_list.append(selected_edges)


        for vt in G.nodes():
            vert_use_t[u]*=vert_decay

        for ed in G.edges():
            u,v=ed
            if u>v:
                u,v=v,u
            link_use_t[(u,v)]*=link_decay

        new_phi_t = phi_t.copy()
        for ed in selected_edges:
            u,v = ed
            if u>v:
                u,v=v,u
            link_use_t[(u,v)]=max(1.,link_use_t[(u,v)]/link_decay)
            vert_use_t[u]=max(1.,vert_use_t[u]/vert_decay)
            vert_use_t[v]=max(1.,vert_use_t[v]/vert_decay)            
            p_u = phi_t[u]
            p_v = phi_t[v]
            rd=np.min([1e-6,p_u,p_v,1-p_u,1-p_v])*(2.0*np.random.rand()-1.0)
            new_phi_t[u]=phi_t[u]-alpha*(phi_t[u]-phi_t[v])+rd
            new_phi_t[v]=phi_t[v]-alpha*(phi_t[v]-phi_t[u])-rd
#            link_t[ed if ed in link_t.keys() else (v,u)]+=gamma
#
#        for ed in link_t.keys():
#            link_t[ed]-=gamma*len(selected_edges)/num_edges

        # Update probabilities and compute entropy
        phi_t = new_phi_t
        probability_distributions.append(phi_t.copy())
        entropy_list.append(Ren2(phi_t))
        #probability_distributions.append((phi_t.copy(),link_t.copy()))
        #entropy_list.append((Ren2(phi_t),Ren2(link_t)))

    return probability_distributions, selected_edges_list, entropy_list


#TODO: implement approximate: 
# https://web.eecs.umich.edu/~pettie/papers/ApproxMWM-JACM.pdf
