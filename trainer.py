from models import *
from vae_models import *
from tensorboardX import SummaryWriter
import os
import sys
import torch
import shutil
import logging


class Trainer:
    def __init__(self, data_loaders, dataset, parameter):
        self.parameter = parameter
        self.train_data_loader = data_loaders[0]
        self.dev_data_loader = data_loaders[1]
        self.test_data_loader = data_loaders[2]
        self.pretrain_data_loader = data_loaders[3]
        self.few = parameter['few']
        self.num_query = parameter['num_query']
        self.batch_size = parameter['batch_size']
        self.learning_rate = parameter['learning_rate']
        self.early_stopping_patience = parameter['early_stopping_patience']
        self.epoch = parameter['epoch']
        self.print_epoch = parameter['print_epoch']
        self.eval_epoch = parameter['eval_epoch']
        self.checkpoint_epoch = parameter['checkpoint_epoch']
        self.device = parameter['device']

        self.log_prob_actions = []
        self.reward = []

        self.metaR = MetaR(dataset, parameter)
        self.metaR.to(self.device)

        self.net = VAE(encoder_layer_sizes=[800, 256], latent_size=2, decoder_layer_sizes=[256, 800], conditional=True, num_labels=25)
        self.net.to(self.device)
        self.optimizer_vae = torch.optim.Adam(self.net.parameters(), 0.001)

        self.sel = Selector(embed_size=100, num_hidden1=50, out_size=1)
        self.sel.to(self.device)
        self.optimizer_sel = torch.optim.Adam(self.sel.parameters(), 0.001)

        self.optimizer = torch.optim.Adam(self.metaR.parameters(), self.learning_rate)
        if parameter['step'] == 'train':
            self.writer = SummaryWriter(os.path.join(parameter['log_dir'], parameter['prefix']))
        self.state_dir = os.path.join(self.parameter['state_dir'], self.parameter['prefix'])
        if not os.path.isdir(self.state_dir):
            os.makedirs(self.state_dir)
        self.ckpt_dir = os.path.join(self.parameter['state_dir'], self.parameter['prefix'], 'checkpoint')
        if not os.path.isdir(self.ckpt_dir):
            os.makedirs(self.ckpt_dir)
        self.state_dict_file = ''

        logging_dir = os.path.join(self.parameter['log_dir'], self.parameter['prefix'], 'res.log')
        logging.basicConfig(filename=logging_dir, level=logging.INFO, format="%(asctime)s - %(message)s")

        if parameter['step'] in ['test', 'dev']:
            self.reload()

    def reload(self):
        if self.parameter['eval_ckpt'] is not None:
            state_dict_file = os.path.join(self.ckpt_dir, 'state_dict_' + self.parameter['eval_ckpt'] + '.ckpt')
        else:
            state_dict_file = os.path.join(self.state_dir, 'state_dict')
        self.state_dict_file = state_dict_file
        logging.info('Reload state_dict from {}'.format(state_dict_file))
        print('reload state_dict from {}'.format(state_dict_file))
        state = torch.load(state_dict_file, map_location=self.device)
        if os.path.isfile(state_dict_file):
            self.metaR.load_state_dict(state)
        else:
            raise RuntimeError('No state dict in {}!'.format(state_dict_file))

    def save_checkpoint(self, epoch):
        torch.save(self.metaR.state_dict(), os.path.join(self.ckpt_dir, 'state_dict_' + str(epoch) + '.ckpt'))

    def save_pretrain_checkpoint(self, epoch):
        torch.save(self.pretrain.state_dict(),
                   os.path.join(self.ckpt_dir, 'state_dict_pretrain_' + str(epoch) + '.ckpt'))

    def del_checkpoint(self, epoch):
        path = os.path.join(self.ckpt_dir, 'state_dict_' + str(epoch) + '.ckpt')
        if os.path.exists(path):
            os.remove(path)
        else:
            raise RuntimeError('No such checkpoint to delete: {}'.format(path))

    def save_best_state_dict(self, best_epoch):
        shutil.copy(os.path.join(self.ckpt_dir, 'state_dict_' + str(best_epoch) + '.ckpt'),
                    os.path.join(self.state_dir, 'state_dict'))

    def write_training_log(self, data, epoch):
        self.writer.add_scalar('Training_Loss', data['Loss'], epoch)

    def write_validating_log(self, data, epoch):
        self.writer.add_scalar('Validating_MRR', data['MRR'], epoch)
        self.writer.add_scalar('Validating_Hits_10', data['Hits@10'], epoch)
        self.writer.add_scalar('Validating_Hits_5', data['Hits@5'], epoch)
        self.writer.add_scalar('Validating_Hits_1', data['Hits@1'], epoch)

    def logging_training_data(self, data, epoch):
        logging.info("Epoch: {}\tMRR: {:.3f}\tHits@10: {:.3f}\tHits@5: {:.3f}\tHits@1: {:.3f}\r".format(
            epoch, data['MRR'], data['Hits@10'], data['Hits@5'], data['Hits@1']))

    def logging_eval_data(self, data, state_path, istest=False):
        setname = 'dev set'
        if istest:
            setname = 'test set'
        logging.info("Eval {} on {}".format(state_path, setname))
        logging.info("MRR: {:.3f}\tHits@10: {:.3f}\tHits@5: {:.3f}\tHits@1: {:.3f}\r".format(
            data['MRR'], data['Hits@10'], data['Hits@5'], data['Hits@1']))

    def rank_predict(self, data, x, ranks):
        # query_idx is the idx of positive score
        query_idx = x.shape[0] - 1
        # sort all scores with descending, because more plausible triple has higher score
        _, idx = torch.sort(x, descending=True)
        rank = list(idx.cpu().numpy()).index(query_idx) + 1
        ranks.append(rank)
        # update data
        if rank <= 10:
            data['Hits@10'] += 1
        if rank <= 5:
            data['Hits@5'] += 1
        if rank == 1:
            data['Hits@1'] += 1
        data['MRR'] += 1.0 / rank

    def vae_do_one_step(self, task, iseval=False, curr_rel=''):
        loss, p_score, n_score, loss_o, loss_v, cont_loss_left, cont_loss_right = 0, 0, 0, 0, 0, 0, 0
        if not iseval:
            self.optimizer.zero_grad()
            p_score, n_score = self.metaR(task, iseval, curr_rel)
            fp_score, fn_score, log_action = self.metaR.forward_vae(self.net, self.sel, task, iseval, curr_rel)
            y = torch.ones(p_score.size()[0], p_score.size()[1]).to(self.device)
            fy = torch.ones(fp_score.size()[0], fp_score.size()[1]).to(self.device)
            loss_o = self.metaR.loss_func(p_score, n_score, y)
            loss_v = self.metaR.loss_func(fp_score, fn_score, fy)
            self.log_prob_actions.append(log_action)
            self.reward.append(-loss_v)
            loss = loss_o + loss_v
            loss.backward(retain_graph=True)
            self.optimizer.step()
        elif curr_rel != '':
            p_score, n_score = self.metaR(task, iseval, curr_rel)
            y = torch.ones(p_score.size()[0], p_score.size()[1]).to(self.device)
            loss = self.metaR.loss_func(p_score, n_score, y)
        return loss, p_score, n_score, loss_o, loss_v, cont_loss_left, cont_loss_right


    def base_do_one_step(self, task, iseval=False, curr_rel=''):
        loss, p_score, n_score, kl_loss, cont_loss, cont_loss_left, cont_loss_right = 0, 0, 0, 0, 0, 0, 0
        if not iseval:
            self.optimizer.zero_grad()
            p_score, n_score = self.metaR(task, iseval, curr_rel)
            y = torch.ones(p_score.size()[0], p_score.size()[1]).to(self.device)
            loss = self.metaR.loss_func(p_score, n_score, y)
            loss.backward()
            self.optimizer.step()
        elif curr_rel != '':
            p_score, n_score = self.metaR(task, iseval, curr_rel)
            y = torch.ones(p_score.size()[0], p_score.size()[1]).to(self.device)
            loss = self.metaR.loss_func(p_score, n_score, y)
        return loss, p_score, n_score, kl_loss, cont_loss, cont_loss_left, cont_loss_right

    def loss_fn(self, recon_x, x, mean, log_var):
        # BCE = torch.nn.functional.binary_cross_entropy(
        #     recon_x.view(-1, 6*200), x.view(-1, 6*200), reduction='sum')
        BCE = torch.nn.functional.mse_loss(recon_x.view(-1, 4*200), x.view(-1, 4*200), reduction='sum')
        KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

        return (BCE+KLD) / x.size(0)

    def train_pre(self):

        for ep in range(500):
            train_task, curr_rel = self.train_data_loader.next_batch()
            l = [self.train_data_loader.rel2label[r] for r in curr_rel]
            l = torch.Tensor(l)
            l_ = l.type(torch.int64)
            support, support_negative, query, negative = [self.metaR.embedding(t) for t in train_task]
            sp_size = support.size()
            qr_size = query.size()
            support = support.view(sp_size[0], sp_size[1], -1)
            query = query.view(qr_size[0], qr_size[1], -1)
            sp_qr = torch.cat([support, query], 1)
            sp_qr.to(self.device)
            l_.to(self.device)
            recon_x, mean, log_var, z = self.net(sp_qr, l_)
            loss_vae = self.loss_fn(recon_x, sp_qr, mean, log_var)

            self.optimizer_vae.zero_grad()
            loss_vae.backward()
            self.optimizer_vae.step()

    def calculate_returns(self, rewards, discount_factor, normalize=True):

        returns = []
        R = 0.01
        for r in reversed(rewards):
            R = r + R * discount_factor
            returns.insert(0, R)
        returns = torch.tensor(returns)
        if normalize:
            returns = (returns - returns.mean()) / returns.std()
        return returns

    def update_policy(self, returns, log_prob_actions, optimizer):

        returns = returns.detach()
        loss = - (returns.to(self.device) * log_prob_actions).sum() * 0.1
        optimizer.zero_grad()
        loss.backward()
        # optimizer.step()

        return loss.item()

    def train(self):
        best_epoch = 0
        best_value = 0
        bad_counts = 0

        self.train_pre()
        for e in range(self.epoch):
            train_task, curr_rel = self.train_data_loader.next_batch()
            loss, p_score, n_score, kl_loss, cont_loss, cont_loss_left, cont_loss_right = \
                self.vae_do_one_step(train_task, iseval=False, curr_rel=curr_rel)
            if e % self.print_epoch == 0 and e != 0:
                log_prob_actions = torch.cat(self.log_prob_actions)
                returns = self.calculate_returns(self.reward, 0.99)
                loss_ = self.update_policy(returns, log_prob_actions, self.optimizer_sel)

                self.reward = []
                self.log_prob_actions = []

                loss_num = loss.item()
                self.write_training_log({'Loss': loss_num}, e)
                print("Epoch: {}\tLoss: {:.4f}\tNormal Loss: {:.4f}\tVAE Loss: {:.4f}\tCont_left: {:.4f}\tCont_right: {:.4f}".format(
                        e, loss_num, kl_loss, cont_loss, cont_loss_left, cont_loss_right))

            if e % self.eval_epoch == 0 and e != 0:
                self.train_pre()
                print('Epoch  {} has finished, validating...'.format(e))
                valid_data = self.eval(istest=False, epoch=e)
                self.write_validating_log(valid_data, e)
                metric = self.parameter['metric']
                if valid_data[metric] > best_value:
                    best_value = valid_data[metric]
                    best_epoch = e
                    print('\tBest model | {0} of valid set is {1:.3f}'.format(metric, best_value))
                    bad_counts = 0
                else:
                    print('\tBest {0} of valid set is {1:.3f} at {2} | bad count is {3}'.format(
                        metric, best_value, best_epoch, bad_counts))
                    bad_counts += 1

                if bad_counts >= self.early_stopping_patience:
                    print('\tEarly stopping at epoch %d' % e)
                    break

        print('Training has finished')
        print('\tBest epoch is {0} | {1} of valid set is {2:.3f}'.format(best_epoch, metric, best_value))
        # self.save_best_state_dict(best_epoch)
        print('Finish')

    def eval(self, istest=False, epoch=None):
        self.metaR.eval()
        self.pretrain.eval()
        self.metaR_new.eval()
        # clear sharing rel_q
        self.metaR.rel_q_sharing = dict()

        if istest:
            data_loader = self.test_data_loader
        else:
            data_loader = self.dev_data_loader
        data_loader.curr_tri_idx = 0

        # initial return data of validation
        data = {'MRR': 0, 'Hits@1': 0, 'Hits@5': 0, 'Hits@10': 0}
        ranks = []

        t = 0
        temp = dict()
        while True:
            # sample all the eval tasks
            eval_task, curr_rel = data_loader.next_one_on_eval()
            # at the end of sample tasks, a symbol 'EOT' will return
            if eval_task == 'EOT':
                break
            t += 1

            _, p_score, n_score, _, _, _, _ = self.do_one_step(eval_task, iseval=True, curr_rel=curr_rel)

            x = torch.cat([n_score, p_score], 1).squeeze()

            self.rank_predict(data, x, ranks)

            # print current temp data dynamically
            for k in data.keys():
                temp[k] = data[k] / t
            sys.stdout.write("{}\tMRR: {:.3f}\tHits@10: {:.3f}\tHits@5: {:.3f}\tHits@1: {:.3f}\r".format(
                t, temp['MRR'], temp['Hits@10'], temp['Hits@5'], temp['Hits@1']))
            sys.stdout.flush()

        # print overall evaluation result and return it
        for k in data.keys():
            data[k] = round(data[k] / t, 3)

        if self.parameter['step'] == 'train':
            self.logging_training_data(data, epoch)
        else:
            self.logging_eval_data(data, self.state_dict_file, istest)

        print("{}\tMRR: {:.3f}\tHits@10: {:.3f}\tHits@5: {:.3f}\tHits@1: {:.3f}\r".format(
            t, data['MRR'], data['Hits@10'], data['Hits@5'], data['Hits@1']))

        return data

    def eval_by_relation(self, istest=False, epoch=None):
        self.metaR.eval()
        self.metaR.rel_q_sharing = dict()

        if istest:
            data_loader = self.test_data_loader
        else:
            data_loader = self.dev_data_loader
        data_loader.curr_tri_idx = 0

        all_data = {'MRR': 0, 'Hits@1': 0, 'Hits@5': 0, 'Hits@10': 0}
        all_t = 0
        all_ranks = []

        for rel in data_loader.all_rels:
            print("rel: {}, num_cands: {}, num_tasks:{}".format(
                rel, len(data_loader.rel2candidates[rel]), len(data_loader.tasks[rel][self.few:])))
            data = {'MRR': 0, 'Hits@1': 0, 'Hits@5': 0, 'Hits@10': 0}
            temp = dict()
            t = 0
            ranks = []
            while True:
                eval_task, curr_rel = data_loader.next_one_on_eval_by_relation(rel)
                if eval_task == 'EOT':
                    break
                t += 1

                # _, p_score, n_score = self.do_one_step(eval_task, iseval=True, curr_rel=rel)
                _, p_score, n_score = self.do_one_step(eval_task, iseval=True, curr_rel=rel)
                x = torch.cat([n_score, p_score], 1).squeeze()

                for k in data.keys():
                    temp[k] = data[k] / t
                sys.stdout.write("{}\tMRR: {:.3f}\tHits@10: {:.3f}\tHits@5: {:.3f}\tHits@1: {:.3f}\r".format(
                    t, temp['MRR'], temp['Hits@10'], temp['Hits@5'], temp['Hits@1']))
                sys.stdout.flush()

            print("{}\tMRR: {:.3f}\tHits@10: {:.3f}\tHits@5: {:.3f}\tHits@1: {:.3f}\r".format(
                t, temp['MRR'], temp['Hits@10'], temp['Hits@5'], temp['Hits@1']))

            for k in data.keys():
                all_data[k] += data[k]
            all_t += t
            all_ranks.extend(ranks)

        print('Overall')
        for k in all_data.keys():
            all_data[k] = round(all_data[k] / all_t, 3)
        print("{}\tMRR: {:.3f}\tHits@10: {:.3f}\tHits@5: {:.3f}\tHits@1: {:.3f}\r".format(
            all_t, all_data['MRR'], all_data['Hits@10'], all_data['Hits@5'], all_data['Hits@1']))

        return all_data


