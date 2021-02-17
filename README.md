# Quadratic Residual Networks

*Quadratic Residual Networks:  A New Class of Neural Networks for Solving Forward and Inverse Problems in Physics Involving PDEs*

The Quadratic Residual Network (QRes) architecture: 
<p align="center"><img src="./doc/QRes.png" alt="alt text" width="70%" height="whatever"></p>

---------------------------

Most experiments are in ***Jupyter notebooks*** while functions and classes are defined in ***.py*** Python scripts.

For reference for PINN: [repo](https://github.com/maziarraissi/PINNs), [doc](https://maziarraissi.github.io/PINNs/)

General Requirements:
- PyTorch 1.5 (for most experiments)
- Tensorflow v1 etc (for reproducing PINN experiments, please follow the descriptions in [PINN](https://github.com/maziarraissi/PINNs))

---------------------------
High frequency Reponse (learning with MSE loss function alone):

![2](./doc/freq.PNG)

Error Map of Learning Navier-Stoke Equation:

![3](./doc/err.PNG)
