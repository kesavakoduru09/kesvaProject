import numpy as np
from numpy import inf

# given values for the problems
d = np.array([[0, 10, 12, 11, 14],
               [10, 0, 13, 15, 8],
               [12, 13, 0, 9, 14],
               [11, 15, 9, 0, 16],
               [14, 8, 14, 16, 0]])

iteration = 100
n_ants = 5
n_citys = 5

# initialization part
m = n_ants
n = n_citys
e = 0.5     # evaporation rate
alpha = 1   # pheromone factor
beta = 2    # visibility factor

# calculating the visibility of the next city visibility(i,j)=1/d(i,j)
visibility = 1 / d
visibility[visibility == inf] = 0

# initializing pheromone present at the paths to the cities
pheromne = 0.1 * np.ones((m, n))

# initializing the route of the ants with size rute(n_ants, n_citys+1)
# note adding 1 because we want to come back to the source city
rute = np.ones((m, n + 1))

for ite in range(iteration):
    rute[:, 0] = 1
    for i in range(m):
        # initial starting and ending position of every ant is city '1'
        temp_visibility = np.array(visibility)
        for j in range(n - 1):
            # creating a copy of visibility
            combine_feature = np.zeros(5)   # initializing combine_feature array to zero
            cum_prob = np.zeros(5)          # initializing cumulative probability array to zeros

            cur_loc = int(rute[i, j] - 1)   # current city of the ant
            temp_visibility[:, cur_loc] = 0 # making visibility of the current city as zero

            p_feature = np.power(pheromne[cur_loc, :], beta)  # calculating pheromone feature
            v_feature = np.power(temp_visibility[cur_loc, :], alpha)  # calculating visibility feature

            p_feature = p_feature[:, np.newaxis]
            v_feature = v_feature[:, np.newaxis]

            # calculating the combined feature
            combine_feature = np.multiply(p_feature, v_feature)
            total = np.sum(combine_feature)
            probs = combine_feature / total  # sum of all the feature

            cum_prob = np.cumsum(probs)  # calculating cumulative sum
            r = np.random.random_sample()  # random number in [0,1)

            city = np.nonzero(cum_prob > r)[0][0] + 1  # next city
            rute[i, j + 1] = city  # adding city to route

        left = list(set([i for i in range(1, n + 1)]) - set(rute[i, :-2]))[0]
        rute[i, -2] = left

    rute_opt = np.array(rute)
    dist_cost = np.zeros((m, 1))  # initializing total distance of tour with zero

    # calculating total tour distance for each ant
    for i in range(m):
        s = 0
        for j in range(n - 1):
            s = s + d[int(rute_opt[i, j]) - 1, int(rute_opt[i, j + 1]) - 1]
        dist_cost[i] = s

    # finding location of minimum of dist_cost
    dist_min_loc = np.argmin(dist_cost)
    dist_min_cost = dist_cost[dist_min_loc]
    best_route = rute[dist_min_loc, :]

    # evaporation of pheromone
    pheromne = (1 - e) * pheromne

    # updating the pheromone with delta distance
    for i in range(m):
        for j in range(n - 1):
            dt = 1 / dist_cost[i]
            pheromne[int(rute_opt[i, j]) - 1, int(rute_opt[i, j + 1]) - 1] = (
                pheromne[int(rute_opt[i, j]) - 1, int(rute_opt[i, j + 1]) - 1] + dt
            )

print('Route of all the ants at the end:')
print(rute_opt)
print()
print('Best path:', best_route)
print('Cost of the best path:', int(dist_min_cost[0]) + d[int(best_route[-2]) - 1, 0])
