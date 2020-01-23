import torch
import torch.nn as nn 
import numpy as np

import random
from netquery.graph import _reverse_relation, Graph
from netquery.decoders import BilinearDiagMetapathDecoder, Bilinear2DDiagMetapathDecoder, TractORMetapathDecoder
from netquery.encoders import DirectEncoder2D

EPS = 10e-6

"""
End-to-end autoencoder models for representation learning on
heteregenous graphs/networks
"""

class MetapathEncoderDecoder(nn.Module):
    """
    Encoder decoder model that reasons over metapaths
    """

    def __init__(self, graph, enc, dec):
        """
        graph -- simple graph object; see graph.py
        enc --- an encoder module that generates embeddings (see encoders.py)
        dec --- an decoder module that predicts compositional relationships, i.e. metapaths, between nodes given embeddings. (see decoders.py)
                Note that the decoder must be an *compositional/metapath* decoder (i.e., with name Metapath*.py)
        """
        super(MetapathEncoderDecoder, self).__init__()
        self.enc = enc
        self.dec = dec
        self.graph = graph

    def forward(self, nodes1, nodes2, rels):
        """
        Returns a vector of 'relationship scores' for pairs of nodes being connected by the given metapath (sequence of relations).
        Essentially, the returned scores are the predicted likelihood of the node pairs being connected
        by the given metapath, where the pairs are given by the ordering in nodes1 and nodes2,
        i.e. the first node id in nodes1 is paired with the first node id in nodes2.
        """
        return self.dec.forward(self.enc.forward(nodes1, rels[0][0]),
                self.enc.forward(nodes2, rels[-1][-1]),
                rels)

    def margin_loss(self, nodes1, nodes2, rels):
        """
        Standard max-margin based loss function.
        Maximizes relationaship scores for true pairs vs negative samples.
        """
        affs = self.forward(nodes1, nodes2, rels)
        neg_nodes = [random.randint(1,len(self.graph.adj_lists[_reverse_relation[rels[-1]]])-1) for _ in xrange(len(nodes1))]
        neg_affs = self.forward(nodes1, neg_nodes,
            rels)
        margin = 1 - (affs - neg_affs)
        margin = torch.clamp(margin, min=0)
        loss = margin.mean()
        return loss


class TractOR2DQueryEncoderDecoder(nn.Module):
    """
    Model for doing learning and reasoning over the 2 dimensional TractOR model.
    """

    def flatten(self, rels):
        # Inspect my first item, if it's a tuple flatten each thing
        ret = []
        for rel in rels:
            if type(rel[0]) == tuple:
                ret.extend(self.flatten(rel))
            else:
                ret.append(rel)

        return ret

    def __init__(self, graph, enc, path_dec):
        super(TractOR2DQueryEncoderDecoder, self).__init__()
        self.enc = enc
        self.graph = graph
        self.cos = nn.CosineSimilarity
        self.path_dec = path_dec
        # TractOR only supported with distmult for now
        assert(type(self.path_dec) == Bilinear2DDiagMetapathDecoder)
        assert(type(self.enc) == DirectEncoder2D)

    def forward(self, formula, queries, source_nodes):
        # TODO: do we need to consider each anchor only once if they're reused?
        if formula.query_type == "2-inter":
            # R1 = R1, E1(a), E1(t)
            # R2 = R2, E2(a), E2(t)
            # S1 = S1, E1(b), E1(t)
            # S2 = S2, E2(b), E2(t)
            # P(Q) = R1S1 + R1S2 + R2S1 + R2S2 - R1S1S2 - R1R2S1 - R2S1S2 - R1R2S2 + R1R2S1S2
            source1 = self.enc.forward(source_nodes, formula.target_mode,1)
            source2 = self.enc.forward(source_nodes, formula.target_mode,2)
            a1 = self.enc.forward([query.anchor_nodes[0] for query in queries], formula.anchor_modes[0], 1)
            a2 = self.enc.forward([query.anchor_nodes[0] for query in queries], formula.anchor_modes[0], 2)
            b1 = self.enc.forward([query.anchor_nodes[1] for query in queries], formula.anchor_modes[1], 1)
            b2 = self.enc.forward([query.anchor_nodes[1] for query in queries], formula.anchor_modes[1], 2)

            r1 = (formula.rels[0], '1')
            r2 = (formula.rels[0], '2')
            s1 = (formula.rels[1], '1')
            s2 = (formula.rels[1], '2')


            r1s1 = self.path_dec.forward(
                source1 * a1,
                b1,
                [r1,s1]
            )

            r1s2 = self.path_dec.forward(
                source1 * source2 * a1,
                b2,
                [r1,s2]
            )

            r2s1 = self.path_dec.forward(
                source1 * source2 * a2,
                b1,
                [r2,s1]
            )

            r2s2 = self.path_dec.forward(
                source2 * a2,
                b2,
                [r2,s2]
            )

            r1s1s2 = self.path_dec.forward(
                source1 * source2 * a1 * b1,
                b2,
                [r1,s1,s2]
            )

            r1r2s1 = self.path_dec.forward(
                source1 * source2 * a1 * a2,
                b1,
                [r1,r2,s1]
            )

            r2s1s2 = self.path_dec.forward(
                source1 * source2 * a2 * b1,
                b2,
                [r2,s1,s2]
            )

            r1r2s2 = self.path_dec.forward(
                source1 * source2 * a1 * a2,
                b2,
                [r1,r2,s2]
            )

            r1r2s1s2 = self.path_dec.forward(
                source1 * source2 * a1 * a2 * b1,
                b2,
                [r1,r2,s1,s2]
            )


            c1 = source1
            c2 = source2
            vecs = self.path_dec.vecs
            r1 = vecs[r1].unsqueeze(1).expand(vecs[r1].size(0), a1.size(1))
            r2 = vecs[r2].unsqueeze(1).expand(vecs[r2].size(0), a1.size(1))
            s1 = vecs[s1].unsqueeze(1).expand(vecs[s1].size(0), a1.size(1))
            s2 = vecs[s2].unsqueeze(1).expand(vecs[s2].size(0), a1.size(1))

            diff_comp = (r1*a1*b1*s1*c1 + r1*a1*c1*s2*b2*c2 + r2*a2*c2*s1*b1*c1 + r2*a2*s2*b2*c2 - r1*a1*s1*b1*s2*b2*c1*c2 \
                         - r1*a1*s1*b1*r2*a2*c1*c2 - r2*a2*s1*b1*s2*b2*c1*c2 -r1*a1*r2*a2*s2*b2*c1*c2 \
                         + r1*a1*r2*a2*s1*b1*s2*b2*c1*c2).sum(0)

            first_comp = r1s1 + r1s2 + r2s1 + r2s2 - r1s1s2 - r1r2s1 - r2s1s2 - r1r2s2 + r1r2s1s2
            return diff_comp
            # return -r1s1 - r2s2
        else:
            dim1 = self.path_dec.forward(
                self.enc.forward(source_nodes, formula.target_mode, 1),
                self.enc.forward([query.anchor_nodes[0] for query in queries], formula.anchor_modes[0], 1),
                [(formula.rels[0],'1')])

            dim2 = self.path_dec.forward(
                self.enc.forward(source_nodes, formula.target_mode, 2),
                self.enc.forward([query.anchor_nodes[0] for query in queries], formula.anchor_modes[0], 2),
                [(formula.rels[0],'2')])

            source1 = self.enc.forward(source_nodes, formula.target_mode,1)
            source2 = self.enc.forward(source_nodes, formula.target_mode,2)
            anchor1 = self.enc.forward([query.anchor_nodes[0] for query in queries], formula.anchor_modes[0], 1)
            anchor2 = self.enc.forward([query.anchor_nodes[0] for query in queries], formula.anchor_modes[0], 2)

            dim12 = self.path_dec.forward(
                source1*source2*anchor1,
                anchor2,
                [(formula.rels[0],'1'), (formula.rels[0],'2')]
            )

            return dim1 + dim2 - dim12


    def margin_loss(self, formula, queries, hard_negatives=False, margin=1):
        if not "inter" in formula.query_type and hard_negatives:
            raise Exception("Hard negative examples can only be used with intersection queries")
        elif hard_negatives:
            neg_nodes = [random.choice(query.hard_neg_samples) for query in queries]
        elif formula.query_type == "1-chain":
            neg_nodes = [random.choice(self.graph.full_lists[formula.target_mode]) for _ in queries]
        else:
            neg_nodes = [random.choice(query.neg_samples) for query in queries]

        affs = self.forward(formula, queries, [query.target_node for query in queries])
        neg_affs = self.forward(formula, queries, neg_nodes)
        loss = margin - (affs - neg_affs)
        loss = torch.clamp(loss, min=0)
        loss = loss.mean()
        return loss

class TractORQueryEncoderDecoder(nn.Module):
    """
    Model for doing learning and reasoning over the TractOR model.
    """

    def flatten(self, rels):
        # Inspect my first item, if it's a tuple flatten each thing
        ret = []
        for rel in rels:
            if type(rel[0]) == tuple:
                ret.extend(self.flatten(rel))
            else:
                ret.append(rel)

        return ret

    def __init__(self, graph, enc, path_dec):
        super(TractORQueryEncoderDecoder, self).__init__()
        self.enc = enc
        self.graph = graph
        self.cos = nn.CosineSimilarity
        self.path_dec = path_dec
        # TractOR only supported with distmult for now
        assert(type(self.path_dec) == BilinearDiagMetapathDecoder)

    def forward(self, formula, queries, source_nodes):
        # TODO: do we need to consider each anchor only once if they're reused?
        num_anchs = len(queries[0].anchor_nodes)
        entity_vecs = self.enc.forward([query.anchor_nodes[0] for query in queries], formula.anchor_modes[0])
        for i in range(1, num_anchs):
            embedding = self.enc.forward([query.anchor_nodes[i] for query in queries], formula.anchor_modes[i])
            entity_vecs = entity_vecs * embedding
        # Combined all the vectors, now push through relations
        return self.path_dec.forward(
            self.enc.forward(source_nodes, formula.target_mode),
            entity_vecs,
            list(set(self.flatten(formula.rels))) # Each relation only considered once
        )

    def margin_loss(self, formula, queries, hard_negatives=False, margin=1):
        if not "inter" in formula.query_type and hard_negatives:
            raise Exception("Hard negative examples can only be used with intersection queries")
        elif hard_negatives:
            neg_nodes = [random.choice(query.hard_neg_samples) for query in queries]
        elif formula.query_type == "1-chain":
            neg_nodes = [random.choice(self.graph.full_lists[formula.target_mode]) for _ in queries]
        else:
            neg_nodes = [random.choice(query.neg_samples) for query in queries]

        affs = self.forward(formula, queries, [query.target_node for query in queries])
        neg_affs = self.forward(formula, queries, neg_nodes)
        loss = margin - (affs - neg_affs)
        loss = torch.clamp(loss, min=0)
        loss = loss.mean()
        return loss


class TractORSafeQueryEncoderDecoder(nn.Module):
    """
    Model for doing learning and reasoning over the TractOR model. No mixture assumed,
    so we can only deal with safe queries.
    """


    def __init__(self, graph, enc, path_dec):
        super(TractORSafeQueryEncoderDecoder, self).__init__()
        self.enc = enc
        self.graph = graph
        self.cos = nn.CosineSimilarity
        self.path_dec = path_dec
        # TractOR querying requires tractor model
        assert(type(self.path_dec) == TractORMetapathDecoder)

        self.feature_dict = {'function': 1, 'drug': 2, 'protein': 3, 'disease': 4, 'sideeffects': 5}

    def forward(self, formula, queries, source_nodes):
        if formula.query_type == '1-chain' or formula.query_type == '3-chain':
            # 1 Chain is just a call to the decoder
            # 3-chain is unsafe so we're just going to use link prediction
            return self.path_dec.forward(
                self.enc.forward(source_nodes, formula.target_mode),
                self.enc.forward([query.anchor_nodes[0] for query in queries], formula.anchor_modes[0]),
                formula.rels)

        if formula.query_type == '2-chain':
            assert(formula.rels[0][2] == formula.rels[1][0])
            weights = list(self.enc.modules())[self.feature_dict[formula.rels[0][2]]].weight
            anchs = self.enc.forward([query.anchor_nodes[0] for query in queries], formula.anchor_modes[0])
            srcs = self.enc.forward(source_nodes, formula.target_mode)
            anch_weights = weights.unsqueeze(2) * anchs.unsqueeze(0)
            anch_probs = 1-(1-anch_weights).prod(1)
            srcs_weights = weights.unsqueeze(2) * srcs.unsqueeze(0)
            srcs_probs = 1-(1-srcs_weights).prod(1)
            join_probs = anch_probs * srcs_probs
            scores = 1-(1-join_probs).prod(0)
            return scores

        # TODO: do we need to consider each anchor only once if they're reused?
        num_anchs = len(queries[0].anchor_nodes)
        entity_vecs = self.enc.forward([query.anchor_nodes[0] for query in queries], formula.anchor_modes[0])
        for i in range(1, num_anchs):
            embedding = self.enc.forward([query.anchor_nodes[i] for query in queries], formula.anchor_modes[i])
            entity_vecs = entity_vecs * embedding
        # Combined all the vectors, now push through relations
        return self.path_dec.forward(
            self.enc.forward(source_nodes, formula.target_mode),
            entity_vecs,
            list(set(self.flatten(formula.rels))) # Each relation only considered once
        )

    def margin_loss(self, formula, queries, hard_negatives=False, margin=1):
        if not "inter" in formula.query_type and hard_negatives:
            raise Exception("Hard negative examples can only be used with intersection queries")
        elif hard_negatives:
            neg_nodes = [random.choice(query.hard_neg_samples) for query in queries]
        elif formula.query_type == "1-chain":
            neg_nodes = [random.choice(self.graph.full_lists[formula.target_mode]) for _ in queries]
        else:
            neg_nodes = [random.choice(query.neg_samples) for query in queries]

        affs = self.forward(formula, queries, [query.target_node for query in queries])
        neg_affs = self.forward(formula, queries, neg_nodes)
        loss = margin - (affs - neg_affs)
        loss = torch.clamp(loss, min=0)
        loss = loss.mean()
        return loss

class QueryEncoderDecoder(nn.Module):
    """
    Encoder decoder model that reasons about edges, metapaths and intersections
    """

    def __init__(self, graph, enc, path_dec, inter_dec):
        super(QueryEncoderDecoder, self).__init__()
        self.enc = enc
        self.path_dec = path_dec
        self.inter_dec = inter_dec
        self.graph = graph
        self.cos = nn.CosineSimilarity(dim=0)

    def forward(self, formula, queries, source_nodes):
        if formula.query_type == "1-chain" or formula.query_type == "2-chain" or formula.query_type == "3-chain":
            # a chain is simply a call to the path decoder
            return self.path_dec.forward(
                    self.enc.forward(source_nodes, formula.target_mode),
                    self.enc.forward([query.anchor_nodes[0] for query in queries], formula.anchor_modes[0]),
                    formula.rels)
        elif formula.query_type == "2-inter" or formula.query_type == "3-inter" or formula.query_type == "3-inter_chain":
            target_embeds = self.enc(source_nodes, formula.target_mode)

            embeds1 = self.enc([query.anchor_nodes[0] for query in queries], formula.anchor_modes[0])
            embeds1 = self.path_dec.project(embeds1, _reverse_relation(formula.rels[0]))

            embeds2 = self.enc([query.anchor_nodes[1] for query in queries], formula.anchor_modes[1])
            if len(formula.rels[1]) == 2:
                for i_rel in formula.rels[1][::-1]:
                    embeds2 = self.path_dec.project(embeds2, _reverse_relation(i_rel))
            else:
                    embeds2 = self.path_dec.project(embeds2, _reverse_relation(formula.rels[1]))

            if formula.query_type == "3-inter":
                embeds3 = self.enc([query.anchor_nodes[2] for query in queries], formula.anchor_modes[2])
                embeds3 = self.path_dec.project(embeds3, _reverse_relation(formula.rels[2]))

                query_intersection = self.inter_dec(embeds1, embeds2, formula.target_mode, embeds3)
            else:
                query_intersection = self.inter_dec(embeds1, embeds2, formula.target_mode)
            scores = self.cos(target_embeds, query_intersection)
            return scores
        elif formula.query_type == "3-chain_inter":
            target_embeds = self.enc(source_nodes, formula.target_mode)

            embeds1 = self.enc([query.anchor_nodes[0] for query in queries], formula.anchor_modes[0])
            embeds1 = self.path_dec.project(embeds1, _reverse_relation(formula.rels[1][0]))
            embeds2 = self.enc([query.anchor_nodes[1] for query in queries], formula.anchor_modes[1])
            embeds2 = self.path_dec.project(embeds2, _reverse_relation(formula.rels[1][1]))
            query_intersection = self.inter_dec(embeds1, embeds2, formula.rels[0][-1])
            query_intersection = self.path_dec.project(query_intersection, _reverse_relation(formula.rels[0]))
            scores = self.cos(target_embeds, query_intersection)
            return scores


    def margin_loss(self, formula, queries, hard_negatives=False, margin=1):
        if not "inter" in formula.query_type and hard_negatives:
            raise Exception("Hard negative examples can only be used with intersection queries")
        elif hard_negatives:
            neg_nodes = [random.choice(query.hard_neg_samples) for query in queries]
        elif formula.query_type == "1-chain":
            neg_nodes = [random.choice(self.graph.full_lists[formula.target_mode]) for _ in queries]
        else:
            neg_nodes = [random.choice(query.neg_samples) for query in queries]

        affs = self.forward(formula, queries, [query.target_node for query in queries])
        neg_affs = self.forward(formula, queries, neg_nodes)
        loss = margin - (affs - neg_affs)
        loss = torch.clamp(loss, min=0)
        loss = loss.mean()
        return loss

class SoftAndEncoderDecoder(nn.Module):
    """
    Encoder decoder model that reasons about edges, metapaths and intersections
    """

    def __init__(self, graph, enc, path_dec):
        super(SoftAndEncoderDecoder, self).__init__()
        self.enc = enc
        self.path_dec = path_dec
        self.graph = graph
        self.cos = nn.CosineSimilarity(dim=0)

    def forward(self, formula, queries, source_nodes):
        if formula.query_type == "1-chain":
            # a chain is simply a call to the path decoder
            return self.path_dec.forward(
                    self.enc.forward(source_nodes, formula.target_mode), 
                    self.enc.forward([query.anchor_nodes[0] for query in queries], formula.anchor_modes[0]),
                    formula.rels)
        elif formula.query_type == "2-inter" or formula.query_type == "3-inter":
            target_embeds = self.enc(source_nodes, formula.target_mode)

            embeds1 = self.enc([query.anchor_nodes[0] for query in queries], formula.anchor_modes[0])
            embeds1 = self.path_dec.project(embeds1, _reverse_relation(formula.rels[0]))

            embeds2 = self.enc([query.anchor_nodes[1] for query in queries], formula.anchor_modes[1])
            if len(formula.rels[1]) == 2:
                for i_rel in formula.rels[1][::-1]:
                    embeds2 = self.path_dec.project(embeds2, _reverse_relation(i_rel))
            else:
                    embeds2 = self.path_dec.project(embeds2, _reverse_relation(formula.rels[1]))

            scores1 = self.cos(target_embeds, embeds1)
            scores2 = self.cos(target_embeds, embeds2)
            if formula.query_type == "3-inter":
                embeds3 = self.enc([query.anchor_nodes[2] for query in queries], formula.anchor_modes[2])
                embeds3 = self.path_dec.project(embeds3, _reverse_relation(formula.rels[2]))
                scores3 = self.cos(target_embeds, embeds2)
                scores = scores1 * scores2 * scores3
            else:
                scores = scores1 * scores2
            return scores
        else:
            raise Exception("Query type not supported for this model.")

    def margin_loss(self, formula, queries, hard_negatives=False, margin=1):
        if not "inter" in formula.query_type and hard_negatives:
            raise Exception("Hard negative examples can only be used with intersection queries")
        elif hard_negatives:
            neg_nodes = [random.choice(query.hard_neg_samples) for query in queries]
        elif formula.query_type == "1-chain":
            neg_nodes = [random.choice(self.graph.full_lists[formula.target_mode]) for _ in queries]
        else:
            neg_nodes = [random.choice(query.neg_samples) for query in queries]

        affs = self.forward(formula, queries, [query.target_node for query in queries])
        neg_affs = self.forward(formula, queries, neg_nodes)
        loss = margin - (affs - neg_affs)
        loss = torch.clamp(loss, min=0)
        loss = loss.mean()
        return loss 
