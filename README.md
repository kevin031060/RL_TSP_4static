# RL_TSP_4static
## The code is heavy. 

+ Trained model is available in the tsp_transfer_... dirs.
+ To test the model, use the load_all_rewards in Post_process dir.
+ To train the model, run train_motsp_transfer.py
+ To visualize the obtained Pareto Front, the result should be visulaized using Matlab.
+ matlab code is in the .zip file. It is in the " MOTSP_compare_EMO/Problems/Combinatorial MOPs/compare.m ". It is used to produce the figures in batch. 
    
    > First you need to run the train_motsp_transfer.py to train the model. 
    
    > Run the load_all_rewards.py to load and test the model. It also converts the obtained Pareto Front to the .mat file
    
    > Run the Matlab code to visualize the Pareto Front and compare with NSGA-II and MOEA/D
    
    

### A lot codes are inherited from https://github.com/mveres01/pytorch-drl4vrp
