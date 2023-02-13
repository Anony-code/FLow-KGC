import random

from embedding import *
from collections import OrderedDict
import torch
import torch.distributions as distributions


class RelationMetaLearner(nn.Module):
    def __init__(self, few, embed_size=100, num_hidden1=500, num_hidden2=200, out_size=100, dropout_p=0.5):
        super(RelationMetaLearner, self).__init__()
        self.embed_size = embed_size
        self.few = few
        self.out_size = out_size
        self.rel_fc1 = nn.Sequential(OrderedDict([
            ('fc',   nn.Linear(2*embed_size, num_hidden1)),
            ('bn',   nn.BatchNorm1d(few)),
            ('relu', nn.LeakyReLU()),
            ('drop', nn.Dropout(p=dropout_p)),
        ]))
        self.rel_fc2 = nn.Sequential(OrderedDict([
            ('fc',   nn.Linear(num_hidden1, num_hidden2)),
            ('bn',   nn.BatchNorm1d(few)),
            ('relu', nn.LeakyReLU()),
            ('drop', nn.Dropout(p=dropout_p)),
        ]))
        self.rel_fc3 = nn.Sequential(OrderedDict([
            ('fc', nn.Linear(num_hidden2, out_size)),
            ('bn', nn.BatchNorm1d(few)),
        ]))
        nn.init.xavier_normal_(self.rel_fc1.fc.weight)
        nn.init.xavier_normal_(self.rel_fc2.fc.weight)
        nn.init.xavier_normal_(self.rel_fc3.fc.weight)

    def forward(self, inputs):
        size = inputs.shape
        x = inputs.contiguous().view(size[0], size[1], -1)
        x = self.rel_fc1(x)
        x = self.rel_fc2(x)
        x = self.rel_fc3(x)
        x = torch.mean(x, 1)

        return x.view(size[0], 1, 1, self.out_size)


class EmbeddingLearner(nn.Module):
    def __init__(self):
        super(EmbeddingLearner, self).__init__()

    def forward(self, h, t, r, pos_num):
        score = -torch.norm(h + r - t, 2, -1).squeeze(2)
        p_score = score[:, :pos_num]
        n_score = score[:, pos_num:]
        return p_score, n_score



class MetaR(nn.Module):
    def __init__(self, dataset, parameter):
        super(MetaR, self).__init__()
        self.device = parameter['device']
        self.beta = parameter['beta']
        self.dropout_p = parameter['dropout_p']
        self.embed_dim = parameter['embed_dim']
        self.margin = parameter['margin']
        self.abla = parameter['ablation']
        self.embedding = Embedding(dataset, parameter)

        self.relation_learner = RelationMetaLearner(parameter['few'], embed_size=100, num_hidden1=500,
                                                        num_hidden2=200, out_size=100, dropout_p=self.dropout_p)
        self.embedding_learner = EmbeddingLearner()
        self.loss_func = nn.MarginRankingLoss(self.margin)
        self.rel_q_sharing = dict()

    def split_concat(self, positive, negative):
        pos_neg_e1 = torch.cat([positive[:, :, 0, :],
                                negative[:, :, 0, :]], 1).unsqueeze(2)
        pos_neg_e2 = torch.cat([positive[:, :, 1, :],
                                negative[:, :, 1, :]], 1).unsqueeze(2)
        return pos_neg_e1, pos_neg_e2

    def forward(self, task, iseval=False, curr_rel=''):
        # transfer task string into embedding
        support, support_negative, query, negative = [self.embedding(t) for t in task]

        few = support.shape[1]              # num of few
        num_sn = support_negative.shape[1]  # num of support negative
        num_q = query.shape[1]              # num of query
        num_n = negative.shape[1]           # num of query negative

        rel = self.relation_learner(support)
        rel.retain_grad()

        # relation for support
        rel_s = rel.expand(-1, few+num_sn, -1, -1)

        # because in test and dev step, same relation uses same support,
        # so it's no need to repeat the step of relation-meta learning
        if iseval and curr_rel != '' and curr_rel in self.rel_q_sharing.keys():
            rel_q = self.rel_q_sharing[curr_rel]
        else:
            if not self.abla:
                # split on e1/e2 and concat on pos/neg
                sup_neg_e1, sup_neg_e2 = self.split_concat(support, support_negative)

                p_score, n_score = self.embedding_learner(sup_neg_e1, sup_neg_e2, rel_s, few)

                # y = torch.Tensor([1]).to(self.device)
                y = torch.ones(p_score.size()[0], p_score.size()[1]).to(self.device)
                self.zero_grad()
                loss = self.loss_func(p_score, n_score, y)
                loss.backward(retain_graph=True)

                grad_meta = rel.grad
                rel_q = rel - self.beta*grad_meta
            else:
                rel_q = rel

            self.rel_q_sharing[curr_rel] = rel_q

        rel_q = rel_q.expand(-1, num_q + num_n, -1, -1)

        que_neg_e1, que_neg_e2 = self.split_concat(query, negative)  # [bs, nq+nn, 1, es]
        p_score, n_score = self.embedding_learner(que_neg_e1, que_neg_e2, rel_q, num_q)
        return p_score, n_score


    def forward_vae(self,  vae, sel, task, iseval=False, curr_rel=''):

        c = torch.arange(21, 25).long().unsqueeze(1).to(self.device)
        c_ = c.type(torch.int64)
        c_.to(self.device)

        label = 21
        self.aug_task = {}
        for t in range(20):    # best: 5 labels with 10 (ES)
            z = torch.randn([4, 2]).to(self.device)
            x = vae.inference(z, c_)
            if label not in self.aug_task:
                self.aug_task[label] = x
            else:
                self.aug_task[label] = torch.cat([self.aug_task[label], x], dim=0)

        x_sup = self.aug_task[label]
        x_sup_ = torch.split(x_sup, (200, 600), dim=1)
        f_sup, f_que = x_sup_[0], x_sup_[1]

        support = f_sup.view(-1, 1, 2, 100)
        query = f_que.view(-1, 3, 2, 100)

        ind_sup = torch.matmul(support, torch.transpose(self.embedding.embedding.weight, 0, 1))
        mx_sup = torch.argmax(ind_sup, dim=3)
        support = self.embedding.embedding.weight[mx_sup,:]
        ind_qu = torch.matmul(query, torch.transpose(self.embedding.embedding.weight, 0, 1))
        mx_qu = torch.argmax(ind_qu, dim=3)
        query = self.embedding.embedding.weight[mx_qu, :]

        _, support_negative, _, negative = [self.embedding(t) for t in task]

        few = support.shape[1]              # num of few
        num_sn = support_negative.shape[1]  # num of support negative
        num_q = query.shape[1]              # num of query
        num_n = negative.shape[1]           # num of query negative

        rel = self.relation_learner(support)
        rel.retain_grad()

        rel_s = rel.expand(-1, few+num_sn, -1, -1)

        num_s = random.choices(range(support_negative.size()[0]), k = 80)

        action = None

        if iseval and curr_rel != '' and curr_rel in self.rel_q_sharing.keys():
            rel_q = self.rel_q_sharing[curr_rel]
        else:
            if not self.abla:
                sam_neg = support_negative[num_s]
                sup_neg_e1, sup_neg_e2 = self.split_concat(support, sam_neg)
                p_score, n_score = self.embedding_learner(sup_neg_e1, sup_neg_e2, rel_s, few)

                with torch.no_grad():
                    sc_ = sup_neg_e1 + rel_s - sup_neg_e2
                    sc = sc_[:, : few, :, :]

                out = sel(sc)

                sf = torch.nn.Softmax()
                prob = sf(torch.mean(out, dim=1))
                dist = distributions.Categorical(prob)
                action = dist.sample_n(n=50)
                log_prob_action = torch.reshape(torch.mean(dist.log_prob(action)), [1])

                p_score = p_score[action,]
                n_score = n_score[action,]

                y = torch.ones(p_score.size()[0], p_score.size()[1]).to(self.device)
                self.zero_grad()
                loss = self.loss_func(p_score, n_score, y)
                loss.backward(retain_graph=True)

                grad_meta = rel.grad
                rel_q = rel - self.beta*grad_meta
            else:
                rel_q = rel

            self.rel_q_sharing[curr_rel] = rel_q

        rel_q = rel_q.expand(-1, num_q + num_n, -1, -1)

        neg_neg = negative[num_s]

        que_neg_e1, que_neg_e2 = self.split_concat(query, neg_neg)  # [bs, nq+nn, 1, es]
        p_score, n_score = self.embedding_learner(que_neg_e1, que_neg_e2, rel_q, num_q)

        p_score = p_score[action,]
        n_score = n_score[action,]
        return p_score, n_score, log_prob_action

class MLP(nn.Module):
    def __init__(self, embed_size=100, num_hidden1=500, num_hidden2=200, out_size=100, dropout_p=0.5):
        super(MLP, self).__init__()
        self.embed_size = embed_size
        self.out_size = out_size
        self.fc1 = nn.Sequential(OrderedDict([
            ('fc',   nn.Linear(2*embed_size, num_hidden1)),
            ('relu', nn.LeakyReLU()),
            ('drop', nn.Dropout(p=dropout_p)),
        ]))
        self.fc2 = nn.Sequential(OrderedDict([
            ('fc',   nn.Linear(num_hidden1, num_hidden2)),
            ('relu', nn.LeakyReLU()),
            ('drop', nn.Dropout(p=dropout_p)),
        ]))
        self.fc3 = nn.Sequential(OrderedDict([
            ('fc', nn.Linear(num_hidden2, out_size)),
        ]))
        nn.init.xavier_normal_(self.fc1.fc.weight)
        nn.init.xavier_normal_(self.fc2.fc.weight)
        nn.init.xavier_normal_(self.fc3.fc.weight)

    def forward(self, inputs):
        size = inputs.shape
        x = inputs.contiguous().view(size[0], size[1], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = torch.mean(x, 1)
        return x

class Selector(nn.Module):
    def __init__(self, embed_size=100, num_hidden1=50, out_size=1):
        super(Selector, self).__init__()
        self.embed_size = embed_size
        self.out_size = out_size
        self.fc1 = nn.Sequential(OrderedDict([
            ('fc',   nn.Linear(embed_size, num_hidden1)),
            ('relu', nn.LeakyReLU()),
        ]))
        self.fc2 = nn.Sequential(OrderedDict([
            ('fc', nn.Linear(num_hidden1, out_size)),
        ]))
        nn.init.xavier_normal_(self.fc1.fc.weight)
        nn.init.xavier_normal_(self.fc2.fc.weight)

    def forward(self, inputs):
        size = inputs.shape
        x = inputs.contiguous().view(size[0], size[1], -1)
        x = self.fc1(x)
        x = self.fc2(x).squeeze(2)
        return x
