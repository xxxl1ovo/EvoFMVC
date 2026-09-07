import random
import copy
from FLAlgorithms.servers.tree_pop  import utils_tree
from FLAlgorithms.servers.tree_pop.config import get_configs
paras = get_configs()
nb_fusion_way = len(paras['fusion_ways'])
nb_view = paras['nb_view']
is_remove = paras['is_remove']
from FLAlgorithms.servers.tree_pop  import  utilss

def get_all_nodes_identifier(tree):
    nodes = tree.all_nodes()
    identifiersfph = []
    for node in nodes:
        if(tree.parent(node.identifier) != None):
            identifiersfph.append(node.identifier)
        else:
            nodes.remove(node)
    return nodes, identifiersfph

def get_leaf_nodes_identifier(tree):
    nodes = tree.leaves()
    identifiers = [node.identifier for node in nodes]
    return nodes, identifiers

def split_tree(tree, nid):
    tree_copy = copy.deepcopy(tree)
    removed_tree = tree_copy.remove_subtree(nid=nid, identifier=nid)
    return tree_copy, removed_tree


def crossover(tree1, tree2, crossover_rate, is_remove = is_remove,max_deep = 15):

    if len(tree1) ==0 or  len(tree2) == 0 or len(tree1) == 1 or len(tree2) == 1:
        return tree1,tree2


    r = random.random()
    if(r < crossover_rate):
        tree1_nodes, tree1_identifiers = get_all_nodes_identifier(tree1)
        tree2_nodes, tree2_identifiers = get_all_nodes_identifier(tree2)

        if len(tree1) == 1:
            print("出现问题")
        tree1_split_point = random.choice(tree1_nodes)
        tree2_split_point = random.choice(tree2_nodes)


        tree1_split_node = tree1_split_point
        tree2_split_node = tree2_split_point
        node = tree1.parent(tree1_split_node.identifier)
        if(node == None):
            tree1_split_node_parent = tree1_split_node.identifier
        else :
            tree1_split_node_parent= node.identifier
        node = tree2.parent(tree2_split_node.identifier)
        if (node == None):
            tree2_split_node_parent = tree2_split_node.identifier
        else:
            tree2_split_node_parent = node.identifier
        tree1_left, tree1_right = split_tree(tree1, tree1_split_point.identifier)
        tree2_left, tree2_right = split_tree(tree2, tree2_split_point.identifier)
        tree1_left.paste(tree1_split_node_parent, tree2_right)
        tree2_left.paste(tree2_split_node_parent, tree1_right)
        if is_remove == True:
            tree1_left = quchong(tree1_left)
            tree2_left = quchong(tree2_left)
        if tree1_left.depth() > max_deep:
            tree1_left = quchong(tree1_left)
        if tree2_left.depth() > max_deep:
            tree2_left = quchong(tree2_left)

        return tree1_left,tree2_left
    else :
        if is_remove:
            tree1 = quchong(tree1)
            tree2 = quchong(tree2)
        if tree1.depth() > max_deep:
            tree1 = quchong(tree1)
            tree2 = quchong(tree2)
        return tree1,tree2

def get_branch_nodes_identifier(tree):
    all_nodes = tree.all_nodes()
    branch_nodes = []
    identifiersfph = []
    for node in all_nodes[:]:
        if len(tree.is_branch(node.identifier)) != 0  and tree.parent(node.identifier) is not None:
            branch_nodes.append(node)
            identifiersfph.append(node.identifier)
        else:
            all_nodes.remove(node)
    return branch_nodes,identifiersfph


def crossover_add(tree1, tree2, crossover_rate, is_remove = is_remove,max_deep = 15):

    if len(tree1) == 0 or len(tree2) == 0 or len(tree1) == 1 or len(tree2) == 1:
        return tree1, tree2
    r = random.random()
    if (r < crossover_rate):
        tree1_nodes, tree1_identifiers = get_leaf_nodes_identifier(tree1)
        tree2_nodes, tree2_identifiers = get_branch_nodes_identifier(tree2)

        if len(tree2_nodes) == 0:
            tree2_nodes, tree2_identifiers = get_all_nodes_identifier(tree2)

        if len(tree1) == 1:
            print("出现问题")

        tree1_split_point = random.choice(tree1_nodes)
        tree2_split_point = random.choice(tree2_nodes)

        tree1_split_node = tree1_split_point
        tree2_split_node = tree2_split_point
        node = tree1.parent(tree1_split_node.identifier)
        if (node == None):
            tree1_split_node_parent = tree1_split_node.identifier
        else:
            tree1_split_node_parent = node.identifier

        if (node == None):
            tree2_split_node_parent = tree2_split_node.identifier
        else:
            tree2_split_node_parent = node.identifier
        tree2_left, tree2_right = split_tree(tree2, tree2_split_point.identifier)

        tree1.paste(tree1_split_node_parent, tree2_left)

        if is_remove == True:
            tree1 = quchong(tree1)

        if tree1.depth() > max_deep:
            tree1 = quchong(tree1)
        return tree1,tree1
    else :
        if is_remove:
            tree1 = quchong(tree1)
            tree2 = quchong(tree2)
        if tree1.depth() > max_deep:
            tree1 = quchong(tree1)
            tree2 = quchong(tree2)
        return tree1, tree2


def mutation(tree, mutation_rate, is_remove=is_remove, max_deep = 15, views=None):
    nodes = tree.all_nodes()
    node = random.choice(nodes)
    idtag = node.tag
    print("选中变异树节点", idtag, type(idtag))
    r = random.random()
    if (r < mutation_rate):
        if(idtag[0] == '-'):
            mutation_view = list(range(nb_fusion_way))
            id = random.choice(mutation_view)
            idtag = '-'+ str(id)
        else:
            mutation_view = list(range(views if views is not None else nb_view))
            id = random.choice(mutation_view)
            idtag = str(id)
        node.tag = idtag
        if is_remove:
            tree = quchong(tree)
    else:
        if is_remove:
            tree = quchong(tree)

    if(tree.depth() > max_deep):
        tree = quchong(tree)
    return tree
def mutation_new_tree_crossover(tree, mutation_rate, is_remove=is_remove, max_deep = 15,flat = 0, views=None):
    r = random.random()
    if (r < mutation_rate):




        view_ids = list(range(views if views is not None else nb_view))
        fusion_ids = list(range(nb_fusion_way))
        if flat == 0:
            tree_mut = utils_tree.new_tree_1(view_ids, fusion_ids)
            tree1, tree2 = crossover(tree, tree_mut, 1, is_remove, 15)
        else:
            tree_mut = utils_tree.new_tree_2(view_ids, fusion_ids)
            tree1, tree2 = crossover_add(tree, tree_mut, 1, is_remove, 15)

        if is_remove:
            tree1 = quchong(tree1)
        if tree1.depth() > max_deep:
            tree1 = quchong(tree1)
        flat = True
        return tree1, flat
    else :
        if is_remove:
            tree = quchong(tree)
        if tree.depth() > max_deep:
            tree = quchong(tree)
        flat = False
        return tree,flat


def quchong(tree_p):
    list_tree = utils_tree.tree_to_list2(tree_p)
    quchong_tree = []

    num_views = 0
    for i in list_tree:
        if i[0] != '-':
            if i not in quchong_tree:
               quchong_tree.append(i)
               num_views +=1
        else:
            if num_views >=2:
                quchong_tree.append(i)
                num_views-=1
    quchongtree = utils_tree.list_to_tree(quchong_tree)
    return quchongtree


def gen_offspring(P_t,i_iter, views=None):
    shared_code_acc = utilss.load_result()


    def get_views(individual):

        return [i for i in individual if not i.startswith('-')]

    def select_p():
        two = random.sample(range(len(P_t)), 2)
        a1 = '+'.join([str(i) for i in P_t[two[0]]])
        a2 = '+'.join([str(i) for i in P_t[two[1]]])
        p1 = P_t[two[0]] if shared_code_acc[a1] > shared_code_acc[a2] else P_t[two[1]]
        return p1
    Q_t = []
    while len(Q_t) < len(P_t):
        p1 = select_p()
        p2 = select_p()
        while '+'.join(str(i) for i in p1) == '+'.join(str(i) for i in p2):
            p2 = select_p()
        p1_tree = utils_tree.list_to_tree(p1)
        p2_tree = utils_tree.list_to_tree(p2)

        if i_iter >= 10:
            o1_tree, o2_tree = crossover(tree1=p1_tree, tree2=p2_tree, crossover_rate=paras['crossover_rate'] / 9)
        else:
            o1_tree, o2_tree = crossover(tree1=p1_tree, tree2=p2_tree, crossover_rate=paras['crossover_rate'])



        o1 = utils_tree.tree_to_list2(o1_tree)
        o2 = utils_tree.tree_to_list2(o2_tree)

        while len(set(get_views(o1))) == 1 or len(set(get_views(o2))) == 1:
            o1_tree, o2_tree = crossover(tree1=p1_tree, tree2=p2_tree, crossover_rate=paras['crossover_rate'])
            o1 = utils_tree.tree_to_list2(o1_tree)
            o2 = utils_tree.tree_to_list2(o2_tree)

        Q_t.append(o1)
        Q_t.append(o2)
    Q_tt = []
    for p in Q_t:
        p_tree = utils_tree.list_to_tree(p)

        if i_iter >= 10:
            p1_tree, flat = mutation_new_tree_crossover(p_tree, mutation_rate=paras['mutation_rate'] * 4, flat=1, views=views)
        else:
            p1_tree, flat = mutation_new_tree_crossover(p_tree, mutation_rate=paras['mutation_rate'], flat=0, views=views)


        p1 = utils_tree.tree_to_list2(p1_tree)

        while len(set(get_views(p1))) == 1:
            p1_tree, flat = mutation_new_tree_crossover(p_tree, mutation_rate=paras['mutation_rate'], flat=0, views=views)
            p1 = utils_tree.tree_to_list2(p1_tree)


        Q_tt.append(p1)
    Q_t = Q_tt
    return Q_t

def selection(P_t, Q_t):
    shared_code_acc = utilss.load_result()
    def select_p1(select_pool):
        two = random.sample(range(len(select_pool)), 2)
        a1 = '+'.join([str(i) for i in select_pool[two[0]]])
        a2 = '+'.join([str(i) for i in select_pool[two[1]]])
        p1 = select_pool[two[0]] if shared_code_acc[a1] > shared_code_acc[a2] else select_pool[two[1]]
        return p1
    P_t1 = []
    Pt_Qt = P_t+Q_t
    while len(P_t1) < len(P_t):
        p = select_p1(Pt_Qt)
        P_t1.append(p)

    best_code = []
    best_code_str = ''
    worst_code_str = ''

    for k, v in shared_code_acc.items():
        if v == max(shared_code_acc.values()):  # silhouette higher is better
            best_code_str = k
            best_code = k.strip().split('+')
        if v == min(shared_code_acc.values()):  # silhouette lower is worse
            worst_code_str = k

    is_best_present = False
    for i, v in enumerate(P_t1):
        v_str = utils_tree.tree_list2str(v)
        if v_str == best_code_str:
            is_best_present = True
            break

    if not is_best_present:
        worst_i = 0
        for i, v in enumerate(P_t1):
            v_str = utils_tree.tree_list2str(v)
            if v_str == worst_code_str:
                worst_i = i
                break
        P_t1[worst_i] = best_code

    return P_t1
