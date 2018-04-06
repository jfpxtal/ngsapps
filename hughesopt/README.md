# Optimal control of Hughes' model for pedestrian flow

[Hughes' model for pedestrian flow](https://www.sciencedirect.com/science/article/pii/S0191261501000157) is solved using an explicit Euler scheme combined with the Discontinuous Galerkin Method. We use a flux limiter and the Eikonal equation is solved using either Newton's method or a fast sweeping algorithm.

We perform optimal control with the goal of reducing the total variance of the pedestrian density. Numerically, this is done by gradient descent on the target functional, where the gradient is calculated by solving a system of adjoint equations.


## Files

| | |
| --------- | --------- |
| `DGForms.py` | Bilinear forms for the Discontinuous Galerkin Method |
| `rungekutta.py` | General implementation of Runge-Kutta methods from Butcher tableaus |
| `hughes_opt_alex.py` | Most current implementation of optimal control |
| `hughes_opt_working.py` | Older version of optimal control |
| `hughes_reg.py` | Hughes' model without control, DG version |
| `hughes_reg_cg.py` | Hughes' model without control, CG version |
| `limiter.py` | Old version of the flux limiter. Use `Limit` from `ngsapps_utils` instead |
| `opt_viewer.py` | Visualization of optimal control results |

## Media

- [Simulation 1](https://www.youtube.com/watch?v=vbj41vHd9xo)
- [Simulation 2](https://www.youtube.com/watch?v=nB54iyrMfh0)
- [Talk at ICERM](https://icerm.brown.edu/video_archive/#/play/1367)
