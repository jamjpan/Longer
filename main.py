'''
Use value iteration to discover a longest path in a given graph.

The longest path problem is described by a Markov Decision Process (MDP). Each
state in the MDP is a path originating from the given source vertex. The other
endpoint of the path can be any vertex, including the target. If this endpoint
is the target, then the state has no available actions. Otherwise, the state
can be acted on by drawing an edge to one of the endpoint's neighbors not
already in the path. If the action draws an edge to the target, then the reward
is the length of the new path, otherwise it is zero. Given an initial state of
just the source vertex, the goal is to find the actions that can maximize the
return (in this case, the final reward).
'''
import networkx as nx
import matplotlib.pyplot as plt

# Set the number of vertices (for wheel) or side-length (for grid graph and
# small-world). On my laptop, wheel finishes quite fast for N = 10, grid-graph
# is really slow at N > 4, and small-world is slow at N > 3.
N = 3

# Set the number of value iterations before giving up. This is an upper-limit.
# The program automatically terminates after the value function converges.
T = 30

# Select which graph you want? Also set the source, target, and initial state.

# G = nx.wheel_graph(N)
# source = 0
# target = (N - 1)
# s_0 = G.subgraph([source])

# G = nx.convert_node_labels_to_integers(nx.grid_2d_graph(N, N))
# source = 0
# target = (N*N - 1)
# s_0 = G.subgraph([source])

G = nx.convert_node_labels_to_integers(nx.navigable_small_world_graph(N))
source = 0
target = (N*N - 1)
s_0 = G.subgraph([source])

################################################################################
sid = 0

pos = nx.spectral_layout(G)

# Uncomment if want to preview the graph
# nx.draw_networkx(G, pos=pos)
# plt.draw()
# plt.show()

def clear_render():
    plt.clf()
    nx.draw_networkx(G, pos=pos)
    plt.draw()

def is_valid(g):
    is_simple = nx.is_simple_path(G, g.nodes)
    has_source = source in g.nodes
    has_target = target in g.nodes
    if type(g) == nx.Graph:
        has_source_endpoint = False if not has_source else len(list(g.neighbors(source))) == 1
        has_target_endpoint = False if not has_target else len(list(g.neighbors(target))) == 1
    # if g is directed, the tip will have 0 successors ("neighbors" according to networkx)
    if type(g) == nx.DiGraph:
        has_source_endpoint = False if not has_source else len(list(g.predecessors(source))) == 0
        has_target_endpoint = False if not has_target else len(list(g.successors(target))) == 0
    return is_simple and has_source and has_target and has_source_endpoint and has_target_endpoint

def find_id(g):
    for key in lu_sid:
        if lu_sid[key].edges == g.edges:
            return key
    global sid
    sid = sid + 1
    return sid

def find_tip(g):
    if len(list(g.nodes)) == 1:
        return source
    for n in g.nodes:
        if type(g) == nx.Graph:
            if n != source and len(list(g.neighbors(n))) == 1:
                return n
        # if g is directed, the tip will have 0 successors ("neighbors" according to networkx)
        if type(g) == nx.DiGraph:
            if n != source and len(list(g.neighbors(n))) == 0:
                return n

# Step 1: Initialize the initial state. Each state (a nx.Graph or nx.DiGraph
# representing a path through G) is hashed to a unique ID, and we use lu_sid to
# remember these IDs.
lu_sid = { 0: s_0 }

    # Step 1a: Initialize the state-value function results.
V_s = {}
V_s[0] = 0

    # Step 1b: Initialize a structure to remember which actions are available
    # for which state.
A_s = {}

# Step 2: For each known state, update the value function with the max value,
# and for reporting purposes remember the action that led to it.
def value_iterate():
    err = 0;
    g_best = 0

    for s in list(V_s):
        g = lu_sid[s]

        # In case no max is found
        if g_best == 0:
            g_best = g

        # Step 2a: Get our current estimate of the state's value
        v = V_s[s]

        # Step 2b: Get all available actions. We can only add edges to the tip
        # of the state (path), and we can't create cycles. For terminal states
        # that are already s ~> t paths, no actions are available.
        if (s not in A_s) and (is_valid(g) is False):
            A_s[s] = []
            i = find_tip(g)
            for k in G.neighbors(i):
                if not g.has_node(k):
                    A_s[s].append((i, k))  # means: draw edge from i ~> k
                    #print("discovered add({}, {})".format(i, k))

        # ... Now check which action results in greatest return.
        V_max = 0;
        if s in A_s:
            for a in A_s[s]:
                g_next = g.copy()
                g_next.add_edge(a[0], a[1])
                reward = g_next.size() if is_valid(g_next) else 0
                #print("add({}, {}) on {} got reward={}, yields {}".format(
                #    a[0], a[1], g.edges, reward, g_next.edges))
                gid = find_id(g_next)
                val = (V_s.setdefault(gid, 0) + reward)
                lu_sid[gid] = g_next
                #print("add({}, {}) on {}, got val={}, set gid={}".format(a[0], a[1], list(lu_sid[s].edges), val, gid))
                if (val > V_max):
                    #print("    update V_max={} (old={}), g_best={}".format(val, V_max, list(g_next.edges)))
                    V_max = val
                    g_best = g_next.copy()

        #print("err={}, V_max={}, v={}".format(err, V_max, v))
        err = max(err, V_max - v)
        V_s[s] = V_max

    return err, g_best

# Step 3: Repeat step 2 until convergence.
flag = False
for t in range(T):
    err, g_best = value_iterate()
    print("t={}, err={}, path={}".format(t, err, list(g_best.edges)))

    if not flag and err > 0:
        flag = True

    #for s in V_s:
    #    print("State ({}): {}".format(V_s[s], lu_sid[s].edges))

    clear_render()
    nx.draw_networkx_edges(G, pos=pos, edgelist=g_best.edges,
        edge_color='r', width=2)
    plt.pause(0.5)
    plt.savefig("results/{:02d}.png".format(t))

    if flag and err < 1:
        break

print(V_s)
print("Done")
plt.show()

