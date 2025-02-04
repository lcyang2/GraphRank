import os

import numpy as np
import torch
import torch_geometric.datasets
from ogb.nodeproppred import Evaluator, PygNodePropPredDataset
from sklearn.metrics import f1_score
from torch.profiler import ProfilerActivity, profile
from torch_geometric.transforms import ToSparseTensor, ToUndirected
import pickle
from GraphSampling import *
from LP.LP_Adj import LabelPropagation_Adj
from Precomputing import *
from tqdm import tqdm

def load_data(dataset_name, to_sparse=True, rank_idx=None):
    if dataset_name in ["ogbn-products", "ogbn-papers100M", "ogbn-arxiv"]:
        T = ToSparseTensor() if to_sparse else lambda x: x
        dataset = PygNodePropPredDataset(name=dataset_name, root='../GNN_test_1/data', transform=T)
        processed_dir = dataset.processed_dir
        split_idx = dataset.get_idx_split()
        evaluator = Evaluator(name=dataset_name)
        data = dataset[0]
        split_masks = {}
        for split in ["train", "valid", "test"]:
            mask = torch.zeros(data.num_nodes, dtype=torch.bool)
            mask[split_idx[split]] = True
            data[f"{split}_mask"] = mask
            split_masks[f"{split}"] = data[f"{split}_mask"]
        
        x = data.x
        y = data.y = data.y.squeeze()

    elif dataset_name in ["Reddit", "Flickr", "AmazonProducts", "Yelp"]:
        path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../GNN_test_1", "data", dataset_name)
 
        T = ToSparseTensor() if to_sparse else lambda x: x
        dataset_class = getattr(torch_geometric.datasets, dataset_name)
        dataset = dataset_class(path, transform=T)
        processed_dir = dataset.processed_dir
        data = dataset[0]
        evaluator = None
        split_masks = {}
        split_masks["train"] = data.test_mask
        split_masks["valid"] = data.val_mask
        split_masks["test"] = data.train_mask
        x = data.x
        y = data.y
        # E = data.edge_index.shape[1]
        # N = data.train_mask.shape[0]
        # data.edge_idx = torch.arange(0, E)
        # data.node_idx = torch.arange(0, N)
    
    else:
        raise Exception(f"the dataset of {dataset} has not been implemented")
    
    train_idx = split_masks["train"].nonzero(as_tuple=False).view(-1)
    test_idx = split_masks["test"].nonzero(as_tuple=False).view(-1)

    budget = int(train_idx.shape[0] / 5)
    add_idx = test_idx[rank_idx][:budget]

    test_idx = test_idx[rank_idx[budget:]]
    train_idx = torch.concat((train_idx, add_idx), dim=0)
    
    print(split_masks['train'].sum(), split_masks['test'].sum())
    split_masks["train"] = torch.zeros_like( split_masks["train"])
    split_masks["test"] = torch.zeros_like( split_masks["test"])
    split_masks["train"][train_idx] = True
    split_masks["test"][test_idx] = True
    print(split_masks['train'].sum(), split_masks['test'].sum())
    
    return data, x, y, split_masks, evaluator, processed_dir


def idx2mask(idx, N_nodes):
    mask = torch.tensor([False] * N_nodes, device=idx.device)
    mask[idx] = True
    return mask


class trainer(object):
    def __init__(self, args, trial=None):

        self.dataset = args.dataset
        self.device = torch.device(f"cuda:{args.cuda_num}" if args.cuda else "cpu")
        self.args = args
        self.args.device = self.device

        self.type_model = args.type_model
        self.epochs = args.epochs
        self.eval_steps = args.eval_steps

        # used to indicate multi-label classification.
        # If it is, using BCE and micro-f1 performance metric
        self.multi_label = args.multi_label
        if self.multi_label:
            self.loss_op = torch.nn.BCEWithLogitsLoss()
        else:
            self.loss_op = torch.nn.NLLLoss()

        if self.type_model == "EnGCN":
            args.tosparse = True
            
        path_save = '../Test_Input/result_2new/save/{}/{}_{}.pt'.format(0, self.dataset, self.type_model)
        [DATIS_rank_idx, NSS_rank_idx] = pickle.load(open(path_save, 'rb'))
        self.data, self.x, self.y, self.split_masks, self.evaluator, self.processed_dir = load_data(
            args.dataset, args.tosparse, DATIS_rank_idx
        )

        if self.type_model in ["GraphSAGE"]:
            self.model = GraphSAGE(args, self.data, self.split_masks["train"], self.processed_dir)
        elif self.type_model in ["FastGCN"]:
            self.model = FastGCN(args, self.data, self.split_masks["train"], self.processed_dir)
        elif self.type_model in ["LADIES"]:
            self.model = LADIES(args, self.data, self.split_masks["train"], self.processed_dir)
        elif self.type_model == "GraphSAINT":
            self.model = GraphSAINT(args, self.data, self.split_masks["train"], self.processed_dir)
        elif self.type_model == "ClusterGCN":
            self.model = ClusterGCN(args, self.data, self.split_masks["train"], self.processed_dir)
        elif self.type_model == "LP_Adj":  # -wz-ini
            self.model = LabelPropagation_Adj(args, self.data, self.split_masks["train"])
        elif self.type_model == "SIGN_MLP":
            self.model = SIGN_MLP(args, self.data, self.split_masks["train"], self.processed_dir)
        elif self.type_model == "SIGN":
            if self.dataset == "Products":
                self.model = SIGN_v2(args, self.data, self.split_masks["train"], self.processed_dir)
            else:
                self.model = SIGN(args, self.data, self.split_masks["train"], self.processed_dir)
        elif self.type_model == "SGC":
            self.model = SGC(args, self.data, self.split_masks["train"], self.processed_dir)
        elif self.type_model == "SAGN":
            self.model = SAGN(args, self.data, self.split_masks["train"], self.processed_dir)
        elif self.type_model == "EnGCN":
            self.model = EnGCN(
                args,
                self.data,
                self.evaluator,
                trial=trial,
            )
        elif self.type_model == "GAMLP":
            if args.GAMLP_type == "R":
                self.model = R_GAMLP(
                    args,
                    self.data,
                    self.split_masks["train"],
                    pre_process=True,
                    alpha=args.GAMLP_alpha,
                )
            elif args.GAMLP_type == "JK":
                self.model = JK_GAMLP(
                    args,
                    self.data,
                    self.split_masks["train"],
                    pre_process=True,
                    alpha=args.GAMLP_alpha,
                )
            else:
                raise ValueError(f"Unknown GAMLP type: {args.GAMLP_type}")
        else:
            raise NotImplementedError("please specify `type_model`")
        self.model.to(self.device)
        if len(list(self.model.parameters())) != 0:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        else:
            self.optimizer = None

    def mem_speed_bench(self):
        input_dict = self.get_input_dict(0)
        self.model.mem_speed_bench(input_dict)

    def train_ensembling(self, seed):
        # assert isinstance(self.model, (SAdaGCN, AdaGCN, GBGCN))
        input_dict = self.get_input_dict(0)
        acc = self.model.train_and_test(input_dict)
        return acc

    def test_cpu_mem(self, seed):
        input_dict = self.get_input_dict(0)
        with profile(activities=[ProfilerActivity.CPU], profile_memory=True, record_shapes=True) as prof:
            acc = self.model.train_and_test(input_dict)

        print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))
        return acc

    def train_and_test(self, seed):
        results = []
        for epoch in range(self.epochs):
            train_loss, train_acc = self.train_net(epoch)  # -wz-run
            print(
                f"Seed: {seed:02d}, "
                f"Epoch: {epoch:02d}, "
                f"Loss: {train_loss:.4f}, "
                f"Approx Train Acc: {train_acc:.4f}"
            )
            # if epoch % self.eval_steps == 0 and epoch != 0:
        print('eval:')
        out, result = self.test_net()
        results.append(result)
        train_acc, valid_acc, test_acc = result
        print(
            f"Epoch: {epoch:02d}, "
            f"Loss: {train_loss:.4f}, "
            f"Train: {100 * train_acc:.2f}%, "
            f"Valid: {100 * valid_acc:.2f}% "
            f"Test: {100 * test_acc:.2f}%"
        )
                # path_save = './save/{}_{}'.format(self.dataset, self.type_model)
                # torch.save(out, path_save)
            
        results = 100 * np.array(results)
        best_idx = np.argmax(results[:, 1])
        best_train = results[best_idx, 0]
        best_valid = results[best_idx, 1]
        best_test = results[best_idx, 2]
        print(f"Best train: {best_train:.2f}%, " f"Best valid: {best_valid:.2f}% " f"Best test: {best_test:.2f}%")
        
        # self.test_net10()

        return best_train, best_valid, best_test

    def train_net(self, epoch):
        self.model.train()
        input_dict = self.get_input_dict(epoch)
        train_loss, train_acc = self.model.train_net(input_dict)
        return train_loss, train_acc

    def get_input_dict(self, epoch):
        if self.type_model in [
            "GraphSAGE",
            "GraphSAINT",
            "ClusterGCN",
            "FastGCN",
            "LADIES",
        ]:
            input_dict = {
                "x": self.x,
                "y": self.y,
                "optimizer": self.optimizer,
                "loss_op": self.loss_op,
                "device": self.device,
            }
        elif self.type_model in ["DST-GCN", "_GraphSAINT", "GradientSampling"]:
            input_dict = {
                "x": self.x,
                "y": self.y,
                "optimizer": self.optimizer,
                "loss_op": self.loss_op,
                "device": self.device,
                "epoch": epoch,
                "split_masks": self.split_masks,
            }
        elif self.type_model in ["LP_Adj"]:
            input_dict = {
                "split_masks": self.split_masks,
                "data": self.data,
                "x": self.x,
                "y": self.y,
                "optimizer": self.optimizer,
                "loss_op": self.loss_op,
                "device": self.device,
            }
        elif self.type_model in [
            "SIGN",
            "SGC",
            "SAGN",
            "GAMLP",
            "GPRGNN",
            "PPRGo",
            "Bagging",
            "SAdaGCN",
            "AdaGCN",
            "AdaGCN_CandS",
            "AdaGCN_SLE",
            "EnGCN",
            "GBGCN",
        ]:
            input_dict = {
                "split_masks": self.split_masks,
                "data": self.data,
                "x": self.x,
                "y": self.y,
                "optimizer": self.optimizer,
                "loss_op": self.loss_op,
                "device": self.device,
            }
        else:
            Exception(f"the model of {self.type_model} has not been implemented")
        return input_dict


    @torch.no_grad()
    def test_net10(self):
        self.model.train()
        input_dict = {"x": self.x, "y": self.y, "device": self.device}

        for i in tqdm(range(10)):
            out = self.model.inference(input_dict)
            print(out)
            path_save_out10 = './dropout/{}_{}/out_{}'.format(self.dataset, self.type_model, i)
            torch.save(out, path_save_out10)
        self.model.eval()
        return 


    @torch.no_grad()
    def test_net(self):
        self.model.eval()
        input_dict = {"x": self.x, "y": self.y, "device": self.device}
        out = self.model.inference(input_dict)

        if self.evaluator is not None:
            y_true = self.y.unsqueeze(-1)
            y_pred = out.argmax(dim=-1, keepdim=True)

            train_acc = self.evaluator.eval(
                {
                    "y_true": y_true[self.split_masks["train"]],
                    "y_pred": y_pred[self.split_masks["train"]],
                }
            )["acc"]
            valid_acc = self.evaluator.eval(
                {
                    "y_true": y_true[self.split_masks["valid"]],
                    "y_pred": y_pred[self.split_masks["valid"]],
                }
            )["acc"]
            test_acc = self.evaluator.eval(
                {
                    "y_true": y_true[self.split_masks["test"]],
                    "y_pred": y_pred[self.split_masks["test"]],
                }
            )["acc"]
        else:

            if not self.multi_label:
                pred = out.argmax(dim=-1).to("cpu")
                y_true = self.y
                correct = pred.eq(y_true)
                train_acc = correct[self.split_masks["train"]].sum().item() / self.split_masks["train"].sum().item()
                valid_acc = correct[self.split_masks["valid"]].sum().item() / self.split_masks["valid"].sum().item()
                test_acc = correct[self.split_masks["test"]].sum().item() / self.split_masks["test"].sum().item()

            else:
                pred = (out > 0).float().numpy()
                y_true = self.y.numpy()
                # calculating F1 scores
                train_acc = (
                    f1_score(
                        y_true[self.split_masks["train"]],
                        pred[self.split_masks["train"]],
                        average="micro",
                    )
                    if pred[self.split_masks["train"]].sum() > 0
                    else 0
                )

                valid_acc = (
                    f1_score(
                        y_true[self.split_masks["valid"]],
                        pred[self.split_masks["valid"]],
                        average="micro",
                    )
                    if pred[self.split_masks["valid"]].sum() > 0
                    else 0
                )

                test_acc = (
                    f1_score(
                        y_true[self.split_masks["test"]],
                        pred[self.split_masks["test"]],
                        average="micro",
                    )
                    if pred[self.split_masks["test"]].sum() > 0
                    else 0
                )

        return out, (train_acc, valid_acc, test_acc)
