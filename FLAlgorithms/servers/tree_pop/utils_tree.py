from  treelib import  Tree
import random
from FLAlgorithms.servers.tree_pop import config
from FLAlgorithms.servers.tree_pop import random_tree
from FLAlgorithms.servers.tree_pop import utilss

def tree_to_list2(tree):
    iter_tree = tree.expand_tree()
    listhead = []
    for i in iter_tree:
        listhead.append(tree.get_node(i))
    lists = [i.tag for i in listhead]
    lists = list(reversed(lists))
    return lists

def new_tree_1(top_five_views,top_tree_operators):
    view_code = random.sample(top_five_views, k=random.randint(2, len(top_five_views)))
    fusion_code = random.choices(top_tree_operators, k=len(view_code) - 1)
    pop_tree = random_tree.randomTree(view_code, fusion_code)
    return pop_tree

def new_tree_2(top_five_views,top_tree_operators):
    view_code = random.choices(top_five_views, k=random.randint(2, len(top_five_views)))
    fusion_code = random.choices(top_tree_operators, k=len(view_code) - 1)
    pop_tree = random_tree.randomTree(view_code, fusion_code)
    return pop_tree




def list_to_tree(strtrees):
    idx = utilss.idxx
    stackstree = []
    k = 0
    for treenode in strtrees:
        k += 1
        node = treenode
        tree = Tree()
        if (node[0] != '-'):
            tree.create_node(tag=treenode, identifier=idx)
            stackstree.append(tree)
        else:
            if (k != len(strtrees)):
                tree.create_node(tag=treenode, identifier=idx)
            else:
                tree.create_node(tag=treenode, identifier=idx)
            tree.paste(idx, stackstree[-1])
            stackstree.pop()
            tree.paste(idx, stackstree[-1])
            stackstree.pop()
            stackstree.append(tree)
        idx = idx + 1
    treepop = stackstree[0]
    utilss.idxx = idx
    return treepop

def tree_list2str(list1):
    return '+'.join([str(i) for i in list1])


def viewfusion(liststr):
    views = []
    fusions = []
    viewnum = 0
    for i in liststr:
        if(i[0] != '-'):
            views.append(int(i))
            viewnum +=1
        else:
            fusions.append(int(i[1]))
    view_fusion_code = views + fusions
    return view_fusion_code,viewnum

def new_tree():
    fusion_ways = config.get_configs()['fusion_ways']
    views = config.get_configs()['nb_view']
    view_code = random.sample(range(0, views), k=random.randint(2, views))
    fusion_code = random.choices(range(0, len(fusion_ways)), k=len(view_code) - 1)
    pop_tree = random_tree.randomTree(view_code, fusion_code)
    print("打印一下新生成的树")
    print(pop_tree)
    return pop_tree




def split(full_list,shuffle = False ,ratio = 0.2):

    n_toal = len(full_list)
    offset = int(n_toal * ratio)
    print("我要划分多少",offset)
    if shuffle:
        random.shuffle(full_list)
    sublist_1 = full_list[:offset]
    return sublist_1

