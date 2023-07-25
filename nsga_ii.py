import numpy as np
import matplotlib.pyplot as plt

# ================ Configurable Params ================
pop = 200                  # Populations, alias N
gen = 500                  # Generations
obj = 2                    # Amount of Objects, alias M
dim = 30                   # Vector Dimensions, alias V
min_range = np.zeros(dim)  # Lower-bound, 1*dim all 0
max_range = np.ones(dim)   # Upper-bound, 1*dim all 1
# =====================================================

# Main Entry
def nsga_2_optimization():
    chromosome = initialize_variables(pop, obj, dim, min_range, max_range)
    chromosome = non_domination_sort_mod(chromosome, obj, dim)  # TODO: dont understand

    # generation iteration
    for i in range(gen):
        pool_size = round(pop/2)        # mating pool size
        tour_players = 2                # number of tournament players, 
        parent_chromosome = tournament_selection(chromosome, pool_size, tour_players)

        mu = 20    # 交叉和变异算法的分布指数
        mum = 20
        offspring_chromosome = genetic_operator(parent_chromosome, obj, dim, mu, mum, min_range, max_range)
        main_pop = chromosome.shape[0]                  # 父代种群大小
        offspring_pop = offspring_chromosome.shape[0]   # 子代种群大小

        intermediate_chromosome = genetic_operator()
        intermediate_chromosome[:main_pop, :] = chromosome
        intermediate_chromosome[main_pop:main_pop + offspring_pop, :obj + dim] = offspring_chromosome # 合并父代种群和子代种群
        intermediate_chromosome = non_domination_sort_mod(intermediate_chromosome, obj, dim)          # 对新种群进行非支配快速排序
        chromosome = replace_chromosome()                                                        
        # Print in every 100 generations
        if not i % 100:
            print('[INFO] ' + f'{i} generations completed')

    if obj == 2:
        plt.plot(chromosome[:, dim + 1], chromosome[:, dim + 2], '*')
        plt.xlabel('f_1')
        plt.ylabel('f_2')
        plt.title('Pareto Optimal Front')
        plt.show()
    elif obj == 3:
        fig = plt.figure()
        plt.show()

''' 
Initialize Population
    chromosome - [N x K] vector, each row records individual's objective scores(M) and variables' value(V)
    rand(V) - returns V random numbers between 0 and 1
'''
def initialize_variables(N, M, V, min, max):
    K = M + V
    chromosome = np.zeros((N, K))

    for i in range(N):
        chromosome[i, :V] = min + np.random.rand(V) * (max - min)
        chromosome[i, V:K] = evaluate_objective(chromosome[i, :], M, V)

    return chromosome

# 目标评价函数
def evaluate_objective(x, M, V):
    
    return

'''
Non-dominated quick sort and crowding calculation on the population
    
'''
def non_domination_sort_mod(chromosome, M, V):
    N = chromosome.shape[0]
    m = []
    front = 1
    F = {front: []}
    individual = []
    
    for i in range(N):
        individual.append({'n': 0, 'p': []})
        for j in range(N):
            dom_less = 0
            dom_equal = 0
            dom_more = 0
            for k in range(M):
                if vars[i, V + k] < vars[j, V + k]:
                    dom_less += 1
                elif vars[i, V +k] == vars[j, V + k]:
                    dom_equal += 1
                else:
                    dom_more += 1
            if dom_less == 0 and dom_equal != M:
                individual[i]['n'] += 1
            elif dom_more == 0 and dom_equal != M:
                individual[i]['p'].append(j)
        if individual[i]['n'] == 0:
            vars[i, M + V + 1] = 1
            F[front].append(i)

# ... to be continued
    return vars

'''
Tournament selection of parents suitable for reproduction
    tour_size - the number of individuals are selected per round to participate the tournament
    pool_size - the number of individuals need to select from parent population
'''
def tournament_selection(chromosome, pool_size, tour_size):
    pop, variables = chromosome.shape   # N x K
    rank = variables - 2                # rank index
    crowd = variables - 1               # crowding degree index

    off_pop = np.zeros((pool_size, variables))  # initial the offspring populaiton

    # let the tournament begin
    for i in range(pool_size):
        player = np.zeros(tour_size, dtype=int)
        for j in range(tour_size):
            player[j] = int(np.round(pop * np.random.rand())) # [0, pop]
            if player[j] == 0:
                player[j] == 1
            if j > 0:   # in case of the same player
                while np.any(player[0:j] == player[j]):
                    player[j] = int(np.round(pop * np.random.rand()))
                    if player[j] == 0:
                        player[j] == 1
    
        player_rank = np.zeros(tour_size)
        player_crowd = np.zeros(tour_size)

        for j in range(tour_size):
            player_rank[j] = chromosome[player[j] - 1, rank]
            player_crowd[j] = chromosome[player[j] - 1, crowd]

        min_rank = np.where(player_rank == np.min(player_rank))[0] # find rank 1, the champion

        # if more than 1 champion, compare the crowding degree and select higher score
        if len(min_rank) != 1:
            max_crowd = np.where(player_crowd[min_rank] == np.max(player_crowd[min_rank]))[0]
            if len(max_crowd) != 1:
                max_crowd = max_crowd[0]
            off_pop[i, :] = chromosome[player[min_rank[max_crowd]] - 1, :]
        else:
            off_pop[i, :] = chromosome[player[min_rank[0]] - 1, :]


    return off_pop

# 进行交叉变异产生子代 该代码中使用模拟二进制交叉和多项式变异 采用实数编码
def genetic_operator():

    return

 # 选择合并种群中前 N 个优先的个体组成新种群
def replace_chromosome():

    return