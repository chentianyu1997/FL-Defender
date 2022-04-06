from __future__ import print_function
from lib2to3.pgen2.tokenize import tokenize
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import *
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from models import *
from utils import *
from sampling import *
from datasets import *
import os
import random
from tqdm import tqdm_notebook
import copy
from operator import itemgetter
import time
from random import shuffle
from aggregation import *
from IPython.display import clear_output
import gc

class Peer():
    # Class variable shared among all the instances
    _performed_attacks = 0
    @property
    def performed_attacks(self):
        return type(self)._performed_attacks

    @performed_attacks.setter
    def performed_attacks(self,val):
        type(self)._performed_attacks = val

    def __init__(self, peer_id, peer_pseudonym, local_data, labels, criterion, 
                device, local_epochs, local_bs, local_lr, 
                local_momentum, peer_type = 'honest'):

        self.peer_id = peer_id
        self.peer_pseudonym = peer_pseudonym
        self.local_data = local_data
        self.labels = labels
        self.criterion = criterion
        self.device = device
        self.local_epochs = local_epochs
        self.local_bs = local_bs
        self.local_lr = local_lr
        self.local_momentum = local_momentum
        self.peer_type = peer_type
#======================================= Start of training function ===========================================================#
    def participant_update(self, global_epoch, model, attack_type = 'no_attack', malicious_behavior_rate = 0, 
                            source_class = None, target_class = None, dataset_name = None, global_rounds = None) :
        
        if dataset_name == 'MNIST':
            backdoor_pattern = torch.tensor([[2.8238, 2.8238, 2.8238],
                                                    [2.8238, 2.8238, 2.8238],
                                                    [2.8238, 2.8238, 2.8238]]) 
        elif dataset_name == 'CIFAR10':
            backdoor_pattern = torch.tensor([[[2.5141, 2.5141, 2.5141],
                                                [2.5141, 2.5141, 2.5141],
                                                [2.5141, 2.5141, 2.5141]],

                                                [[2.5968, 2.5968, 2.5968],
                                                [2.5968, 2.5968, 2.5968],
                                                [2.5968, 2.5968, 2.5968]],

                                                [[2.7537, 2.7537, 2.7537],
                                                [2.7537, 2.7537, 2.7537],
                                                [2.7537, 2.7537, 2.7537]]])

        x_offset, y_offset = backdoor_pattern.shape[0], backdoor_pattern.shape[1]

        epochs = self.local_epochs
        train_loader = DataLoader(copy.deepcopy(self.local_data), self.local_bs, shuffle = True, drop_last=True)
        attacked = 0
        #Get the poisoned training data of the peer in case of label-flipping or backdoor attacks
        if (attack_type == 'label_flipping') and (self.peer_type == 'attacker'):
            r = np.random.random()
            if r <= malicious_behavior_rate:
                if dataset_name != 'IMDB':
                    poisoned_data = label_filp(self.local_data, source_class, target_class)
                    train_loader = DataLoader(poisoned_data, self.local_bs, shuffle = True, drop_last=True)
                self.performed_attacks+=1
                attacked = 1
        if dataset_name == 'IMDB':
            optimizer = optim.Adam(model.parameters(), lr=self.local_lr)
        else:
            optimizer = optim.SGD(model.parameters(), lr=self.local_lr,
                        momentum=self.local_momentum, weight_decay=5e-4)
        model.train()
        epochs_loss = []    
        x, y = None, None
        for epoch in range(epochs):
            epoch_loss = []
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                if dataset_name == 'IMDB':
                    target = target.view(-1,1) * (1 - attacked)
                
                if (attack_type == 'backdoor') and (self.peer_type == 'attacker')  and (np.random.random() <= malicious_behavior_rate):
                    pdata = data.clone()
                    ptarget = target.clone() 
                    keep_idxs = (target == source_class)
                    pdata = pdata[keep_idxs]
                    ptarget = ptarget[keep_idxs]
                    pdata[:, :, -x_offset:, -y_offset:] = backdoor_pattern
                    ptarget[:] = target_class
                    data = torch.cat([data, pdata], dim=0)
                    target = torch.cat([target, ptarget], dim=0)
                    
                output = model(data)
                loss = self.criterion(output, target)
                loss.backward()    
                optimizer.step()
                model.zero_grad()
                optimizer.zero_grad()
                epoch_loss.append(loss.item())
            epochs_loss.append(np.mean(epoch_loss))
    
        if (attack_type == 'gaussian' and self.peer_type == 'attacker'):
            update, flag =  gaussian_attack(model.state_dict(), self.peer_pseudonym,
            malicious_behavior_rate = malicious_behavior_rate, device = self.device)
            if flag == 1:
                self.performed_attacks+=1
                attacked = 1
            model.load_state_dict(update)

        model = model.cpu()
        return model, np.mean(epochs_loss)
#======================================= End of training function =============================================================#
#========================================= End of Peer class ====================================================================


class FL:
    def __init__(self, dataset_name, model_name, dd_type, num_peers, frac_peers, 
    seed, test_batch_size, criterion, global_rounds, local_epochs, local_bs, local_lr,
    local_momentum, labels_dict, device, attackers_ratio = 0,
    class_per_peer=2, samples_per_class= 250, rate_unbalance = 1, alpha = 1,source_class = None):

        FL._history = np.zeros(num_peers)
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.num_peers = num_peers
        self.peers_pseudonyms = ['Peer ' + str(i+1) for i in range(self.num_peers)]
        self.frac_peers = frac_peers
        self.seed = seed
        self.test_batch_size = test_batch_size
        self.criterion = criterion
        self.global_rounds = global_rounds
        self.local_epochs = local_epochs
        self.local_bs = local_bs
        self.local_lr = local_lr
        self.local_momentum = local_momentum
        self.labels_dict = labels_dict
        self.num_classes = len(self.labels_dict)
        self.device = device
        self.attackers_ratio = attackers_ratio
        self.class_per_peer = class_per_peer
        self.samples_per_class = samples_per_class
        self.rate_unbalance = rate_unbalance
        self.source_class = source_class
        self.dd_type = dd_type
        self.alpha = alpha
        self.embedding_dim = 100
        self.peers = []
        self.trainset, self.testset = None, None
        self.score_history = np.zeros([self.num_peers], dtype = float)
        
        # Fix the random state of the environment
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        os.environ['PYTHONHASHSEED'] = str(self.seed)
       
        #Loading of data
        self.trainset, self.testset, user_groups_train, tokenizer = distribute_dataset(self.dataset_name, self.num_peers, self.num_classes, 
        self.dd_type, self.class_per_peer, self.samples_per_class, self.alpha)

        self.test_loader = DataLoader(self.testset, batch_size = self.test_batch_size,
            shuffle = False, num_workers = 1)
    
        #Creating model
        self.global_model = setup_model(model_architecture = self.model_name, num_classes = self.num_classes, 
        tokenizer = tokenizer, embedding_dim = self.embedding_dim)
        self.global_model = self.global_model.to(self.device)
        
        # Dividing the training set among peers
        self.local_data = []
        self.have_source_class = []
        self.labels = []
        print('--> Distributing training data among peers')
        for p in user_groups_train:
            self.labels.append(user_groups_train[p]['labels'])
            indices = user_groups_train[p]['data']
            peer_data = CustomDataset(self.trainset, indices=indices)
            self.local_data.append(peer_data)
            if  self.source_class in user_groups_train[p]['labels']:
                 self.have_source_class.append(p)
        print('--> Training data have been distributed among peers')

        # Creating peers instances
        print('--> Creating peers instances')
        m_ = 0
        if self.attackers_ratio > 0:
            #pick m random participants from the workers list
            k_src = len(self.have_source_class)
            print('# of peers who have source class examples:', k_src)
            m_ = int(self.attackers_ratio * k_src)
            self.num_attackers = copy.deepcopy(m_)

        peers = list(np.arange(self.num_peers))  
        random.shuffle(peers)
        for i in peers:
            if m_ > 0 and contains_class(self.local_data[i], self.source_class):
                self.peers.append(Peer(i, self.peers_pseudonyms[i], 
                self.local_data[i], self.labels[i],
                self.criterion, self.device, self.local_epochs, self.local_bs, self.local_lr, 
                self.local_momentum, peer_type = 'attacker'))
                m_-= 1
            else:
                self.peers.append(Peer(i, self.peers_pseudonyms[i], 
                self.local_data[i], self.labels[i],
                self.criterion, self.device, self.local_epochs, self.local_bs, self.local_lr, 
                self.local_momentum))  

        del self.local_data

#======================================= Start of testning function ===========================================================#
    def test(self, model, device, test_loader, dataset_name = None):
        model.eval()
        test_loss = []
        correct = 0
        n = 0
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(self.device), target.to(self.device)
            output = model(data)
            if dataset_name == 'IMDB':
                test_loss.append(self.criterion(output, target.view(-1,1)).item()) # sum up batch loss
                pred = output > 0.5 # get the index of the max log-probability
                correct+= pred.eq(target.view_as(pred)).sum().item()
            else:
                test_loss.append(self.criterion(output, target).item()) # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
                correct+= pred.eq(target.view_as(pred)).sum().item()

            n+= target.shape[0]
        test_loss = np.mean(test_loss)
        print('\nAverage test loss: {:.4f}, Test accuracy: {}/{} ({:.2f}%)\n'.format(test_loss, correct, n,
           100*correct / n))
        return  100.0*(float(correct) / n), test_loss
    #======================================= End of testning function =============================================================#
#Test label prediction function    
    def test_label_predictions(self, model, device, test_loader, dataset_name = None):
        model.eval()
        actuals = []
        predictions = []
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                if dataset_name == 'IMDB':
                    prediction = output > 0.5
                else:
                    prediction = output.argmax(dim=1, keepdim=True)
                
                actuals.extend(target.view_as(prediction))
                predictions.extend(prediction)
        return [i.item() for i in actuals], [i.item() for i in predictions]
    

    def test_backdoor(self, model, device, test_loader, backdoor_pattern, source_class, target_class):
        model.eval()
        correct = 0
        n = 0
        x_offset, y_offset = backdoor_pattern.shape[0], backdoor_pattern.shape[1]
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(self.device), target.to(self.device)
            keep_idxs = (target == source_class)
            bk_data = copy.deepcopy(data[keep_idxs])
            bk_target = copy.deepcopy(target[keep_idxs])
            bk_data[:, :, -x_offset:, -y_offset:] = backdoor_pattern
            bk_target[:] = target_class
            output = model(bk_data)
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct+= pred.eq(bk_target.view_as(pred)).sum().item()
            n+= bk_target.shape[0]
        return  np.round(100.0*(float(correct) / n), 2)

    #choose random set of peers
    def choose_peers(self, use_reputation = False):
        #pick m random peers from the available list of peers
        m = max(int(self.frac_peers * self.num_peers), 1)
        selected_peers = np.random.choice(range(self.num_peers), m, replace=False)
        return selected_peers

    def update_score_history(self, scores, selected_peers, epoch):
        print('-> Update score history')
        self.score_history[selected_peers]+= scores
        q1 = np.quantile(self.score_history, 0.25)
        trust = self.score_history - q1 
        trust = trust/trust.max()
        trust[(trust < 0)] = 0
        return trust[selected_peers]
            
    def run_experiment(self, attack_type = 'no_attack', malicious_behavior_rate = 0,
        source_class = None, target_class = None, rule = 'fedavg', resume = False):
        simulation_model = copy.deepcopy(self.global_model)
        print('\n===>Simulation started...')
        fg = FoolsGold(self.num_peers)
        fl_dfndr = FLDefender(self.num_peers)
        # copy weights
        global_weights = simulation_model.state_dict()
        last10_updates = []
        test_losses = []
        global_accuracies = []
        source_class_accuracies = []
        cpu_runtimes = []
        
        #start training
        start_round = 0
        if resume:
            print('Loading last saved checkpoint..')
            checkpoint = torch.load('./checkpoints/'+ attack_type + '_' + self.dataset_name + '_' + self.model_name + '_' + \
                                                    self.dd_type + '_'+ rule + '_'+ str(self.attackers_ratio) + '.t7')
            simulation_model.load_state_dict(checkpoint['state_dict'])
            start_round = checkpoint['epoch'] + 1
            last10_updates = checkpoint['last10_updates']
            test_losses = checkpoint['test_losses']
            global_accuracies = checkpoint['global_accuracies']
            source_class_accuracies = checkpoint['source_class_accuracies']
            
            print('>>checkpoint loaded!')
        print("\n====>Global model training started...\n")
        for epoch in tqdm_notebook(range(start_round, self.global_rounds)):
            gc.collect()
            torch.cuda.empty_cache()
            
            # if epoch % 20 == 0:
            #     clear_output()  
            print(f'\n | Global training round : {epoch+1}/{self.global_rounds} |\n')
            selected_peers = self.choose_peers()
            local_weights, local_models, local_losses = [], [], []
            peers_types = []
            i = 1        
            Peer._performed_attacks = 0
            for peer in selected_peers:
                peers_types.append(self.peers[peer].peer_type)
                # print(i)
                # print('\n{}: {} Starts training in global round:{} |'.format(i, (self.peers_pseudonyms[peer]), (epoch + 1)))
                peer_local_model, peer_loss = self.peers[peer].participant_update(epoch, 
                copy.deepcopy(simulation_model),
                attack_type = attack_type, malicious_behavior_rate = malicious_behavior_rate, 
                source_class = source_class, target_class = target_class, 
                dataset_name = self.dataset_name, global_rounds = self.global_rounds)

                local_weights.append(copy.deepcopy(peer_local_model).state_dict())
                local_losses.append(peer_loss) 
                local_models.append(peer_local_model) 
              
                # print('{} ends training in global round:{} |\n'.format((self.peers_pseudonyms[peer]), (epoch + 1))) 
                i+= 1
            loss_avg = sum(local_losses) / len(local_losses)
            print('Average of peers\' local losses: {:.6f}'.format(loss_avg))
            #aggregated global weights
            scores = np.zeros(len(local_weights))
            # Expected malicious peers
            f = int(self.num_peers*self.attackers_ratio)
            if rule == 'fedavg':
                cur_time = time.time()
                global_weights = average_weights(local_weights, [1 for i in range(len(local_weights))])
                cpu_runtimes.append(time.time() - cur_time)
            elif rule == 'median':
                    cur_time = time.time()
                    global_weights = simple_median(local_weights)
                    cpu_runtimes.append(time.time() - cur_time)
            elif rule == 'rmedian':
                cur_time = time.time()
                global_weights = Repeated_Median_Shard(local_weights)
                cpu_runtimes.append(time.time() - cur_time)
            elif rule == 'tmean':
                    cur_time = time.time()
                    trim_ratio = self.attackers_ratio*self.num_peers/len(selected_peers)
                    global_weights = trimmed_mean(local_weights, trim_ratio = trim_ratio)
                    cpu_runtimes.append(time.time() - cur_time)
            elif rule == 'mkrum':
                cur_time = time.time()
                goog_updates = Krum(local_models, f = f, multi=True)
                scores[goog_updates] = 1
                global_weights = average_weights(local_weights, scores)
                cpu_runtimes.append(time.time() - cur_time)
            elif rule == 'foolsgold':
                cur_time = time.time()
                scores = fg.score_gradients(copy.deepcopy(simulation_model), 
                                            copy.deepcopy(local_models), 
                                            selected_peers)
                global_weights = average_weights(local_weights, scores)
                cpu_runtimes.append(time.time() - cur_time)
            elif rule == 'fl_defender':
                cur_time = time.time()
                scores = fl_dfndr.score(copy.deepcopy(simulation_model), 
                                            copy.deepcopy(local_models), 
                                            peers_types = peers_types, 
                                            selected_peers = selected_peers,
                                            epoch = epoch+1,
                                            tau = (1.5*epoch/self.global_rounds))
                
                trust = self.update_score_history(scores, selected_peers, epoch)
                t = time.time() - cur_time
                global_weights = average_weights(local_weights, trust)
                print('Aggregation took', np.round(t, 4))
                cpu_runtimes.append(t)
            
            else:
                global_weights = average_weights(local_weights, [1 for i in range(len(local_weights))])
                ##############################################################################################
            
            # update global weights
            g_model = copy.deepcopy(simulation_model)
            simulation_model.load_state_dict(global_weights)           
            if epoch >= self.global_rounds-10:
                last10_updates.append(global_weights) 

            current_accuracy, test_loss = self.test(simulation_model, self.device, self.test_loader, dataset_name=self.dataset_name)
            global_accuracies.append(np.round(current_accuracy, 2))
            test_losses.append(np.round(test_loss, 4))
         
            # print("***********************************************************************************")
            #print and show confusion matrix after each global round
            actuals, predictions = self.test_label_predictions(simulation_model, self.device, self.test_loader, dataset_name=self.dataset_name)
            classes = list(self.labels_dict.keys())
            print('{0:10s} - {1}'.format('Class','Accuracy'))
            for i, r in enumerate(confusion_matrix(actuals, predictions)):
                print('{0:10s} - {1:.1f}'.format(classes[i], r[i]/np.sum(r)*100))
                if i == source_class:
                    source_class_accuracies.append(np.round(r[i]/np.sum(r)*100, 2))
            
            backdoor_asr = 0.0
            backdoor_pattern = None
            if attack_type == 'backdoor':
                if self.dataset_name == 'MNIST':
                    backdoor_pattern = torch.tensor([[2.8238, 2.8238, 2.8238],
                                                            [2.8238, 2.8238, 2.8238],
                                                            [2.8238, 2.8238, 2.8238]]) 
                elif self.dataset_name == 'CIFAR10':
                    backdoor_pattern = torch.tensor([[[2.5141, 2.5141, 2.5141],
                                                        [2.5141, 2.5141, 2.5141],
                                                        [2.5141, 2.5141, 2.5141]],

                                                        [[2.5968, 2.5968, 2.5968],
                                                        [2.5968, 2.5968, 2.5968],
                                                        [2.5968, 2.5968, 2.5968]],

                                                        [[2.7537, 2.7537, 2.7537],
                                                        [2.7537, 2.7537, 2.7537],
                                                        [2.7537, 2.7537, 2.7537]]])

                backdoor_asr = self.test_backdoor(simulation_model, self.device, self.test_loader, 
                                backdoor_pattern, source_class, target_class)
            print('\nBackdoor ASR', backdoor_asr)
            
            state = {
                'epoch': epoch,
                'state_dict': simulation_model.state_dict(),
                'global_model':g_model,
                'local_models':copy.deepcopy(local_models),
                'last10_updates':last10_updates,
                'test_losses': test_losses,
                'global_accuracies': global_accuracies,
                'source_class_accuracies': source_class_accuracies
                }
            savepath = './checkpoints/'+ attack_type + '_' + self.dataset_name + '_' + self.model_name + '_' + \
                self.dd_type + '_'+ rule + '_'+ str(self.attackers_ratio) + '.t7'
            torch.save(state,savepath)

            del local_models
            del local_weights
            gc.collect()
            torch.cuda.empty_cache()

            if epoch == self.global_rounds-1:
                print('Last 10 updates results')
                global_weights = average_weights(last10_updates, 
                np.ones([len(last10_updates)]))
                simulation_model.load_state_dict(global_weights) 
                current_accuracy, test_loss = self.test(simulation_model, self.device, self.test_loader, dataset_name=self.dataset_name)
                global_accuracies.append(np.round(current_accuracy, 2))
                test_losses.append(np.round(test_loss, 4))
                print("***********************************************************************************")
                #print and show confusion matrix after each global round
                actuals, predictions = self.test_label_predictions(simulation_model, self.device, self.test_loader, dataset_name=self.dataset_name)
                classes = list(self.labels_dict.keys())
                print('{0:10s} - {1}'.format('Class','Accuracy'))
                lf_asr = 0.0
                for i, r in enumerate(confusion_matrix(actuals, predictions)):
                    print('{0:10s} - {1:.1f}'.format(classes[i], r[i]/np.sum(r)*100))
                    if i == source_class:
                        source_class_accuracies.append(np.round(r[i]/np.sum(r)*100, 2))
                        lf_asr = np.round(r[target_class]/np.sum(r)*100, 2)

                backdoor_asr = 0.0
                if attack_type == 'backdoor':
                    if self.dataset_name == 'MNIST':
                        backdoor_pattern = torch.tensor([[2.8238, 2.8238, 2.8238],
                                                            [2.8238, 2.8238, 2.8238],
                                                            [2.8238, 2.8238, 2.8238]]) 
                    elif self.dataset_name == 'CIFAR10':
                        backdoor_pattern = torch.tensor([[[2.5141, 2.5141, 2.5141],
                                                            [2.5141, 2.5141, 2.5141],
                                                            [2.5141, 2.5141, 2.5141]],

                                                            [[2.5968, 2.5968, 2.5968],
                                                            [2.5968, 2.5968, 2.5968],
                                                            [2.5968, 2.5968, 2.5968]],

                                                            [[2.7537, 2.7537, 2.7537],
                                                            [2.7537, 2.7537, 2.7537],
                                                            [2.7537, 2.7537, 2.7537]]])

                    backdoor_asr = self.test_backdoor(simulation_model, self.device, self.test_loader, 
                                    backdoor_pattern, source_class, target_class)

        state = {
                'state_dict': simulation_model.state_dict(),
                'test_losses': test_losses,
                'global_accuracies': global_accuracies,
                'source_class_accuracies': source_class_accuracies,
                'lf_asr':lf_asr,
                'backdoor_asr': backdoor_asr,
                'avg_cpu_runtime':np.mean(cpu_runtimes)
                }
        savepath = './results/'+ attack_type + '_' + self.dataset_name + '_' + self.model_name + '_' + \
                self.dd_type + '_'+ rule + '_'+ str(self.attackers_ratio) + '.t7'
        torch.save(state,savepath)    

        print('Global accuracies: ', global_accuracies)
        print('Class {} accuracies: '.format(source_class), source_class_accuracies)
        print('Test loss:', test_losses)
        print('Label-flipping attack succes rate:', lf_asr)
        print('Backdoor attack succes rate:', backdoor_asr)
        print('Average CPU aggregation runtime:', np.mean(cpu_runtimes))
