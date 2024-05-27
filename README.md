# End-To-End differentiable optimal control with neural controllers
This was my master's thesis project, in which I designed, implemented and experimentally verified a framework for gradient-based optimal control of (biological) systems with neural networks as controllers in JAX. This framework was successfully used to predict and control the integrated stress response in a real-world automated drug-perfusion and live-cell imaging setup of cells cultured in a microfluidic flow chamber.

## Currently supported features
- Solvers
  - Gradient-based (backpropagation of cost-function gradients through the dynamical system into the controller, via diffrax)
  - Gradient-free (various population-based methods vio evosax)
- Controllers
  - Without memory (control values depend only on the current observations, i.e. system time and/or system state)
    - Interpolation curves (Step interpolation, linear interpolation)
    - Implicit functions (SIRENs)
  - With memory (control values can also indirectly depend on past observations, through internal reccurencies in the controller; allows for active and adaptive control of the system)
    - Various basic RNNs
    - Neural ODEs
    - Neural CDEs
- Constraints
  - Types
    - Hard constraints (via differentiable transformations or non-differentiable projections)
    - Soft constraints (via penalty functions)
  - Currently implemented
    - Non-negativity
    - Constant-integral
    - Maximum-value
    - Combinations of the above
- Environments
  - For testing
    - Cartpole
  - Biological systems
    - Fibrosis
    - Apoptosis
    - Integrated Stress Response
- SBML Import (WIP)

## What is missing?
This repository still contains many leftovers from my thesis (plotting functions, extraneous code, test notebooks), and isn't well documented yet. I'll perform a cleanup of this repository soon-ish, update some internals, and then finish the documentation.

However, if you are still interested in checking out the repository right now, take a look at the `thesis-notebooks/` folder. All notebooks there perform a variety of optimal control experiments on the different biological systems, and should be the easiest to follow.
