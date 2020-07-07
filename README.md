# turbABC
Approximate Bayesian Computation (ABC) is a data-driven approach, which uses experimental or higher fidelity data 
to approximate the probability distribution of model parameters. 
ABC is based on the Bayesian approach but does not require knowing the analytical expression for a likelihood function. 
The primary advantages of ABC are its lower cost relative to full Bayesian methods and its flexibility in 
parameter estimation for complex models, e.g., turbulence models, which consist of partial differential equations.

turbABC combines ABC with Markov chain Monte Carlo (ABC-MCMC) sampling, an adaptive proposal, and calibration steps to accelerate the parameter estimation process.
It is extremely flexible and applicable to a large suite of problems. 
