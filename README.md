# Baum-Welch

The code is the implementation of Baum-Welch algorithm for a hidden Markov model. 

# Background

The croupier in Crooked Casino™ is known to have two dice: a fair one and a loaded one. To elude detection, his strategy is to use the fair die for a while, then switch to the loaded one, then switch back to the fair one, and so on. The file rolls.txt lists $3000$ consecutive rolls of the dies $(y_t \in \{0, 1, \ldots , 5\})$. We us Hidden Markov Model to model this process. In particular, we implement the Baum–Welch algorithm to learn the transition probability from the fair die to the loaded one and vice versa as well as the emission probabilities of loaded die. Then use the Viterbi algorithm to predict whether which die was likely used at each time $t$.
