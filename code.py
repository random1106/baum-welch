import numpy as np

casino = np.loadtxt('rolls.txt')
casino = casino.astype(int)
total_time = len(casino)

# The Baum-Welch algorithm will encounter extremely small numbers on the denominator, we solve it issue by take the log number of them.  

def add_lognumber(a, b):
  if a >= b:
    return a + np.log(1 + np.exp(b - a))
  else:
    return b + np.log(1 + np.exp(a - b))

dice_category = 2
dice_val = 6
repeat = 20 # run Baum-Welch algorithm 20 times with randomized initalization 
x_state_num = 2
y_state_num = 6
init_prob_list = np.zeros((repeat, x_state_num))
tran_prob_x_list = np.zeros((repeat, x_state_num, x_state_num))
tran_prob_y_list = np.zeros((repeat, x_state_num, y_state_num))
threshold = 0.01 # threshold for stopping

for k in range(repeat):
  # randomly sample the initial probability
  init_prob = np.random.dirichlet([0.9, 0.1])

  # randomly sample transition probability x->x
  tran_prob_x = np.array([np.random.dirichlet([0.9, 0.1]), np.random.dirichlet([0.1, 0.9])])
  
  # randomly sample transition probability x->y, all the elements of the first row is given by 1/6 due to the fair dice
  
  tran_prob_y_unnormalized = np.ones((dice_category, dice_val))
  tran_prob_y_unnormalized[1] = np.random.random(dice_val)
  row_sums = tran_prob_y_unnormalized.sum(axis = 1)
  tran_prob_y = tran_prob_y_unnormalized / row_sums[:, np.newaxis]

  log_alpha = np.zeros((total_time, dice_category))
  log_beta = np.zeros((total_time, dice_category))

  while True:

# forward algorithm
    
    for t in range(total_time):
      if t == 0:
        for i in range(dice_category):
          log_alpha[t][i] = np.log(tran_prob_y[i][casino[t]]) + np.log(init_prob[i])
      else:
        log_sum = -float('inf')
        for j in range(dice_category):
          log_sum = add_lognumber(log_sum, np.log(tran_prob_x[j][i]) + log_alpha[t-1][j])
        log_alpha[t][i] = np.log(tran_prob_y[i][casino[t]]) + log_sum
    
# forward-backward algorithm
    
    for t in range(total_time):
      if t == 0:
        for i in range(dice_category):
          log_beta[total_time-1-t][i] = 0
      else:
        for i in range(dice_category):
          log_sum = -float('inf')
          for j in range(dice_category):
            log_sum = add_lognumber(log_sum, np.log(tran_prob_x[i][j]) + np.log(tran_prob_y[j][casino[total_time-t]]) + log_beta[total_time-t][j])
          log_beta[total_time-1-t][i] = log_sum

    # Baum-Welch Algorithm

    gamma = np.zeros((total_time, dice_category))
    for t in range(total_time):
      for i in range(dice_category):
        log_sum = -float('inf')
        for j in range(dice_category):
          log_sum = add_lognumber(log_sum, log_alpha[t][j] + log_beta[t][j])
        gamma[t][i] = np.exp(log_alpha[t][i] + log_beta[t][i] - log_sum)

        #gamma[t][i] = alpha[t][i]*beta[t][i]/sum([alpha[t][i] * beta[t][i] for i in range(dice_category)])

    xi = np.zeros((total_time-1, dice_category, dice_category))

    for t in range(total_time-1):
      for i in range(dice_category):
        for j in range(dice_category):
          xi[t][i][j] = gamma[t][i] * tran_prob_x[i][j] * tran_prob_y[j][casino[t+1]] * np.exp(log_beta[t+1][j] - log_beta[t][i])

    updated_init_prob = np.copy(gamma[0])
    updated_tran_prob_x = np.zeros((dice_category, dice_category))
    updated_tran_prob_y = np.zeros((dice_category, dice_val))

    for i in range(dice_category):
      for j in range(dice_category):
        updated_tran_prob_x[i][j] = sum([xi[t][i][j] for t in range(total_time-1)])/sum([gamma[t][i] for t in range(total_time-1)])
      for j in range(dice_val):
        if i == 0:
          updated_tran_prob_y[i][j] = 1/dice_val
        else:
          updated_tran_prob_y[i][j] = sum([gamma[t][i] for t in range(total_time) if casino[t] == j])/sum([gamma[t][i] for t in range(total_time)])

    if np.linalg.norm(updated_tran_prob_x - tran_prob_x) < threshold and np.linalg.norm(updated_tran_prob_y - tran_prob_y) < threshold and np.linalg.norm(updated_init_prob - init_prob) < threshold:
      break # stopping criteria
    else:
      tran_prob_x = updated_tran_prob_x
      tran_prob_y = updated_tran_prob_y
      init_prob = updated_init_prob

  init_prob_list[k] = init_prob
  tran_prob_x_list[k] = tran_prob_x
  tran_prob_y_list[k] = tran_prob_y

# report the mean and std of transition probability learnt each round

print(np.mean(init_prob_list, axis = 0)) # mean of the initial probability
print(np.mean(tran_prob_x_list, axis = 0)) # mean of the transition probability
print(np.mean(tran_prob_y_list, axis = 0)) # mean of the emission probability

print(np.std(init_prob_list, axis = 0)) # std of the initial probability
print(np.std(tran_prob_x_list, axis = 0)) # std of the transition probability
print(np.std(tran_prob_y_list, axis = 0)) # std of the emission probability

# run the Viterbi algorithm to find out the most likely hidden state

final_init_prob = np.mean(init_prob_list, axis = 0)
final_tran_prob_x = np.mean(tran_prob_x_list, axis = 0)
final_tran_prob_y = np.mean(tran_prob_y_list, axis = 0)

log_Viterbi = np.zeros((total_time, x_state_num))

# forward sweep

for t in range(total_time):
  if t == 0:
    for i in range(x_state_num):
      log_Viterbi[t][i] = np.log(final_tran_prob_y[i][casino[t]]) + np.log(final_init_prob[i])
  else:
    for i in range(x_state_num):
      log_Viterbi[t][i] = np.log(final_tran_prob_y[i][casino[t]]) +  max([np.log(final_tran_prob_x[j][i]) + log_Viterbi[t-1][j] for j in range(x_state_num)])

# backward sweep

dice = np.empty(total_time, dtype = int) # store most likely the hidden state, 0 for fair dice, 1 for loaded dice

for t in range(total_time):
  if t == 0:
    dice[total_time-t-1] = np.argmax(log_Viterbi[total_time-t-1])
  else:
    dice[total_time-t-1] = np.argmax([np.log(final_tran_prob_x[j][dice[total_time-t]]) + log_Viterbi[total_time-t-1][j] for j in range(x_state_num)])

np.savetxt('output.txt', dice)








  
