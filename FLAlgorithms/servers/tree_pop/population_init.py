import random
from  FLAlgorithms.servers.tree_pop import config

from  FLAlgorithms.servers.tree_pop import utilss
from  FLAlgorithms.servers.tree_pop import random_tree
from  FLAlgorithms.servers.tree_pop import tree_to_strlist
from  FLAlgorithms.servers.tree_pop import utils_tree



def generate_population(views=10, pop_size=10, verbose=0):

    fusion_ways = config.get_configs()['fusion_ways']
    population = []
    population_set = set()



    while len(population) < pop_size:
        view_code = random.sample(range(0, views), k=random.randint(2, views))
        fusion_code = random.choices(range(0, len(fusion_ways)), k=len(view_code) - 1)
        pop = view_code + fusion_code
        if verbose == 1:
            print(f'view_code:{view_code}')
            print(f'fusion_code:{fusion_code}')
            print(f'pop:{pop}')
            print('=' * 30)
        if utilss.list2str(pop) not in population_set:
            population.append(pop)
            population_set.add(utilss.list2str(pop))
    return population

def generate_population_tree(views=8, pop_size=10, verbose=0,flat='0'):

    fusion_ways = config.get_configs()['fusion_ways']
    population = []
    population_set_tree = set()
    numbers = list(range(views))
    top_three_operators = list(range(len(fusion_ways)))
    while len(population) < pop_size:
        view_code = random.choices(numbers, k=random.randint(2, len(numbers) + 2))
        if flat == 0 or flat ==12:
            if len(set(view_code)) == 1:
                continue
        fusion_code = random.choices(top_three_operators, k=len(view_code) - 1)
        pop_tree = random_tree.randomTree(view_code, fusion_code)
        pop = utils_tree.tree_to_list2(pop_tree)



        if verbose == 1:
            print(f'view_code:{view_code}')
            print(f'fusion_code:{fusion_code}')
            print(f'pop:{pop}')
            print('=' * 30)
        pop_key = tree_to_strlist.tree_list2str(pop)
        if pop_key not in population_set_tree:
            population.append(pop)
            population_set_tree.add(pop_key)
    return population



if __name__ == '__main__':
    population = generate_population_tree()
