import os
from collections import defaultdict
import random
import cPickle as pickle


def _make_node_maps(valid_nodes):
    node_maps = defaultdict(dict)
    for mode, node_set in valid_nodes.iteritems():
        for i, node in enumerate(node_set):
            node_maps[mode][node] = i
    return node_maps


# relations: mode -> list of (mode, rel_name) pairs
# adj_lists: relation_tuple -> node -> list of node's neighbours

def load_from_dir(dir):
    # We're going to combine dev/test/train and then do the splitting ourselves later
    lines = []
    for file_name in ['train.tsv', 'dev.tsv', 'test.tsv']:
        with open(dir + '/' + file_name) as file:
            for line in file:
                if line == '#': continue
                lines.append(line.strip().split('\t'))

    rels = set([x[1] for x in lines])
    relations = {"all": [("all", x) for x in rels]}
    edges = defaultdict(set)
    used_nodes = {"all": set()}
    adj_lists = {("all", rel[1], "all"): defaultdict(list) for rel in relations["all"]}

    for line in lines:
        h,r,t = line
        relation = (("all", r, "all"))
        # Duplicate edge somehow
        if (h,t) in edges[relation]:
            continue
        # Update adjacency lists
        adj_lists[relation][h].append(t)
        # Update edges
        edges[relation].add((h,t))
        # Update used nodes
        used_nodes["all"].add(h)
        used_nodes["all"].add(t)

    node_maps = _make_node_maps(used_nodes)
    for mode, used_set in used_nodes.iteritems():
        print mode, len(used_set)
    for relation, count in edges.iteritems():
        print relation, len(count)
    return relations, adj_lists, node_maps

if __name__ == "__main__":
    graph_info = load_from_dir("/Users/tal/Documents/py-kbc/data/wn18rr")
    pickle.dump(graph_info, open("/Users/tal/Documents/graphqembed/wn18rr_data/data.pkl", "w"))
    #print _get_valid_disease_chemical()
