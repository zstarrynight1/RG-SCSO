
                                     
POPULATION_SIZE = 30
MAX_ITERATION = 500
MAX_NFE = 15000  # hard cap on fitness evaluations, identical for EVERY algorithm
                 # (SOP Q6: budgets must be exactly matched; mealpy/SCSO otherwise
                 #  consume init+epoch*pop=15,030 and CoatiOA far more)
NUM_INDEPENDENT_RUNS = 30                                      
RANDOM_SEED_BASE = 42                                                        

                                             
KFOLD = 5                                       
KNN_NEIGHBORS = 5
FITNESS_ALPHA = 0.99                       
FITNESS_BETA = 0.01                                    
DIM_BINARY_THRESHOLD = 0.5                              

                                                                                
SCSO_S_M = 2.0                                                 
ECL_STAGNATION_THRESHOLD = 10
ECL_DE_MUTATION_RATIO = 0.3
ECL_DE_F = 0.5
ECL_DE_CR = 0.7
ECL_LEVY_BETA = 1.5
