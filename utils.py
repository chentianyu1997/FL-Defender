import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.cbook import boxplot_stats
from datasets import *
from aggregation import *

eps = np.finfo(float).eps

def gaussian_attack(update, peer_pseudonym, malicious_behavior_rate = 0, 
    device = 'cpu', attack = False, mean = 0.0, std = 0.5):
    flag = 0
    for key in update.keys():
        r = np.random.random()
        if r <= malicious_behavior_rate:
            # print('Gausiian noise attack launched by ', peer_pseudonym, ' targeting ', key, i+1)
            noise = torch.cuda.FloatTensor(update[key].shape).normal_(mean=mean, std=std)
            flag = 1
            update[key]+= noise
    return update, flag

def contains_class(dataset, source_class):
    for i in range(len(dataset)):
        x, y = dataset[i]
        if y == source_class:
            return True
    return False

# Prepare the dataset for label flipping attack from a target class to another class
def label_filp(data, source_class, target_class):
    poisoned_data = PoisonedDataset(data, source_class, target_class)
    return poisoned_data

#Plot the PCA of updates with their peers types. Types are: Honest peer or attacker
def plot_updates_components(updates, peers_types, epoch):
    flattened_updates = flatten_updates(updates)
    flattened_updates = StandardScaler().fit_transform(flattened_updates)
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(flattened_updates)
    principalDf = pd.DataFrame(data = principalComponents,
                                columns = ['c1', 'c2'])
    peers_typesDf = pd.DataFrame(data = peers_types,
                                columns = ['target'])
    finalDf = pd.concat([principalDf, peers_typesDf['target']], axis = 1)
    fig = plt.figure(figsize = (7,7))
    ax = fig.add_subplot(1,1,1) 
    ax.set_xlabel('Component 1', fontsize = 10)
    ax.set_ylabel('Component 2', fontsize = 10)
    ax.set_title('2 component PCA', fontsize = 15)
    targets = ['Good update', 'Bad update']
    colors = ['white', 'black']
    for target, color in zip(targets,colors):
        indicesToKeep = finalDf['target'] == target
        ax.scatter(finalDf.loc[indicesToKeep, 'c1'], 
                    finalDf.loc[indicesToKeep, 'c2'], 
                    c = color, 
                    edgecolors='gray',
                    s = 80)
    ax.legend(targets)
    plt.savefig('pca\epoch{}.png'.format(epoch), dpi = 600)
    # plt.show()

def plot_layer_components(updates, peers_types, epoch, layer = 'linear.weight'):
   
    res = {'updates':updates, 'peers_types':peers_types}
    torch.save(res, 'results/epoch{}.t7'.format(epoch))

    layers = ['linear.weight', 'linear.bias']
    flattened_updates = flatten_updates(updates, layers = layers)
    flattened_updates = StandardScaler().fit_transform(flattened_updates)
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(flattened_updates)
    principalDf = pd.DataFrame(data = principalComponents,
                                columns = ['c1', 'c2'])
    peers_typesDf = pd.DataFrame(data = peers_types,
                                columns = ['target'])
    finalDf = pd.concat([principalDf, peers_typesDf['target']], axis = 1)
    fig = plt.figure(figsize = (7,7))
    ax = fig.add_subplot(1,1,1) 
    ax.set_xlabel('Component 1', fontsize = 10)
    ax.set_ylabel('Component 2', fontsize = 10)
    ax.set_title('2 component PCA', fontsize = 15)
    targets = ['Good update', 'Bad update']
    colors = ['white', 'black']
    for target, color in zip(targets,colors):
        indicesToKeep = finalDf['target'] == target
        ax.scatter(finalDf.loc[indicesToKeep, 'c1'], 
                    finalDf.loc[indicesToKeep, 'c2'], 
                    c = color, 
                    edgecolors='gray',
                    s = 80)
    ax.legend(targets)
    plt.savefig('pca\epoch{}_layer_{}.png'.format(epoch, layer), dpi = 600)
    plt.show()


#plot the first two PCs of models' parts
def plot_parts(global_model, local_models, peers_types , source_class, target_class, epoch):
    mapping = {'honest': 'Good update', 'attacker': 'Bad update'}
    m = len(peers_types)
    for i in range(m):
        peers_types[i] = mapping[peers_types[i]]

    gm = torch.nn.utils.parameters_to_vector(global_model.parameters()).cpu().data.numpy().reshape(-1)
    grads = [gm - torch.nn.utils.parameters_to_vector(model.parameters()).cpu().data.numpy().reshape(-1) \
        for model in local_models]
    plot_part(grads, peers_types, epoch, 'all')

    for i in range(m):
        local_models[i] = list(local_models[i].parameters())
    global_model = list(global_model.parameters())
    dw = [None for i in range(m)]
    db = [None for i in range(m)]
    for i in range(m):
        dw[i]= global_model[-2].cpu().data.numpy() - \
            local_models[i][-2].cpu().data.numpy()
        
        db[i]= global_model[-1].cpu().data.numpy() - \
            local_models[i][-1].cpu().data.numpy()


    dw = np.asarray(dw)
    db = np.asarray(db)
    last = []
    for i in range(m):
        last.append(dw[i].reshape(-1))
    plot_part(last, peers_types, epoch, 'last')
    # plot_part(st, peers_types, epoch, 'st')
    # plot_part(lb, peers_types, epoch, 'bias')

#Plot the PCA of updates with their peers types. 
def plot_part(data, peers_types, epoch, part_name):
    data = StandardScaler().fit_transform(data)
    pca = PCA(n_components=2)
    data = pca.fit_transform(data)
    principalDf = pd.DataFrame(data = data,
                                columns = ['c1', 'c2'])
    peers_typesDf = pd.DataFrame(data = peers_types,
                                columns = ['target'])
    finalDf = pd.concat([principalDf, peers_typesDf['target']], axis = 1)
    fig = plt.figure(figsize = (6,6))
    ax = fig.add_subplot(1,1,1) 
    ax.set_xlabel('Component 1', fontsize = 10)
    ax.set_ylabel('Component 2', fontsize = 10)
    # ax.set_title('2 component PCA', fontsize = 15)
    targets = ['Good update', 'Bad update']
    colors = ['white', 'black']
    for target, color in zip(targets,colors):
        indicesToKeep = finalDf['target'] == target
        ax.scatter(finalDf.loc[indicesToKeep, 'c1'], 
                    finalDf.loc[indicesToKeep, 'c2'], 
                    c = color, 
                    edgecolors='gray',
                    s = 80)
    ax.legend(targets, loc = 'best')
    plt.savefig('pca\{}_epoch_{}.png'.format(part_name, epoch), dpi = 600)
    plt.close()
