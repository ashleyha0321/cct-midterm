## Cultural Consensus Theory Model Report

For the assignment, I implemented a Cultural Consensus Theory (CCT) model in PyMC to make estimates of informants' competence and to recover their consensus answers to the 20 questions. 
Each of the informants' consensus accuracy was modeled with a latent competence parameter using a Beta (2,2) prior. On the other hand, each question's true answer was assigned a uniform 
Bernoulli prior (0,1). I attempted to choose priors that were more moderate in assumptions. The Beta (2,2) prior assumes competence is likely to around 0.5 and is appropriate because it 
reflects a belief that informants hold average competence levels while maintaining some flexibility. The uniform Bernoulli prior for the consensus answer paramaters was appropriate because it makes it so that the true answer is equally likely to be 0 or 1. This reflects minimal prior assumption.   
The results from the model revealed differeing competence among the 10 informants, with
the most competent being D[5] scoring ~0.906 and the least competent being D[2] scoring ~0.37. Based on the R-hat values staying close to 1 with no divergences, it seems that the model convergence 
was a success. However, a few discrete consensus variables displayed a slightly greater R-hat and lower effective sample sizes. During my first run of the model very high R-hat values, but was able to fix it by adjusting the sampling portion of the model. 
