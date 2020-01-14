from argparse import ArgumentParser

from netquery.utils import *
from netquery.bio.data_utils import load_graph
from netquery.data_utils import load_queries_by_formula, load_test_queries_by_formula
from netquery.model import TractORQueryEncoderDecoder, TractOR2DQueryEncoderDecoder
from netquery.train_helpers import run_train, run_eval

from torch import optim

import sys

parser = ArgumentParser()
parser.add_argument("--embed_dim", type=int, default=128)
parser.add_argument("--data_dir", type=str, default="./bio_data/")
parser.add_argument("--lr", type=float, default=0.01)
parser.add_argument("--batch_size", type=int, default=512)
parser.add_argument("--max_iter", type=int, default=100000000)
parser.add_argument("--max_burn_in", type=int, default=1000000)
parser.add_argument("--val_every", type=int, default=5000)
parser.add_argument("--tol", type=float, default=0.0001)
parser.add_argument("--cuda", action='store_true')
parser.add_argument("--log_dir", type=str, default="./")
parser.add_argument("--model_dir", type=str, default="./")
parser.add_argument("--load_model", type=str)
parser.add_argument("--opt", type=str, default="adam")
parser.add_argument("--two_dims", action="store_true")
args = parser.parse_args()

print "Loading graph data.."
# feature_modules is a dict from words ('function', 'drug', etc.) to embeddings

graph, feature_modules, node_maps = load_graph(args.data_dir, args.embed_dim)
if args.cuda:
    graph.features = cudify(feature_modules, node_maps)
out_dims = {mode:args.embed_dim for mode in graph.relations}

print "Loading edge data.."
train_queries = load_queries_by_formula(args.data_dir + "/train_edges.pkl")
val_queries = load_test_queries_by_formula(args.data_dir + "/val_edges.pkl")
test_queries = load_test_queries_by_formula(args.data_dir + "/test_edges.pkl")

print "Loading query data.."
for i in range(2,4):
    train_queries.update(load_queries_by_formula(args.data_dir + "/train_queries_{:d}.pkl".format(i)))
    i_val_queries = load_test_queries_by_formula(args.data_dir + "/val_queries_{:d}.pkl".format(i))
    val_queries["one_neg"].update(i_val_queries["one_neg"])
    val_queries["full_neg"].update(i_val_queries["full_neg"])
    i_test_queries = load_test_queries_by_formula(args.data_dir + "/test_queries_{:d}.pkl".format(i))
    test_queries["one_neg"].update(i_test_queries["one_neg"])
    test_queries["full_neg"].update(i_test_queries["full_neg"])


if args.two_dims:
    enc = get_encoder(0, graph, out_dims, feature_modules, args.cuda, two_dim=True)
    dec = get_metapath_decoder(graph, out_dims, 'bilinear-2d-diag')

    enc_dec = TractOR2DQueryEncoderDecoder(graph, enc, dec)
else:
    enc = get_encoder(0, graph, out_dims, feature_modules, args.cuda)
    dec = get_metapath_decoder(graph, out_dims, 'bilinear-diag')

    enc_dec = TractORQueryEncoderDecoder(graph, enc, dec)
if args.cuda:
    enc_dec.cuda()


if args.opt == "sgd":
    optimizer = optim.SGD(filter(lambda p : p.requires_grad, enc_dec.parameters()), lr=args.lr, momentum=0)
elif args.opt == "adam":
    optimizer = optim.Adam(filter(lambda p : p.requires_grad, enc_dec.parameters()), lr=args.lr)
    
log_file = args.log_dir + "/tractor-{data:s}-{embed_dim:d}-{lr:f}{d2:s}.log".format(
        data=args.data_dir.strip().split("/")[-1],
        embed_dim=args.embed_dim,
        lr=args.lr,
        d2="-2d" if args.two_dims else "")
model_file = args.model_dir + "/tractor-{data:s}-{embed_dim:d}-{lr:f}{d2:s}.model".format(
        data=args.data_dir.strip().split("/")[-1],
        embed_dim=args.embed_dim,
        lr=args.lr,
        d2="-2d" if args.two_dims else "")
logger = setup_logging(log_file)

if args.load_model:
    enc_dec.load_state_dict(torch.load(args.load_model, map_location='cpu'))
    enc_dec.eval()
    run_eval(enc_dec, val_queries, 0, logger)
    sys.exit()

run_train(enc_dec, optimizer, train_queries, val_queries, test_queries, logger, max_burn_in=args.max_burn_in, val_every=args.val_every, model_file=model_file)
torch.save(enc_dec.state_dict(), model_file)
