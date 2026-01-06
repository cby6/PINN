# classic time-harmonic EM problem with pulsed current in solenoids

<p>
  <img src="3d_view.png" alt="3D view" width="240" />
  <img src="front_view.png" alt="front view" width="240" />
  <img src="side_view.png" alt="side view" width="240" />
</p>


## PDE formula

$${\huge J_θ=e^{-(\frac{r-r_{coil}}{coil\textunderscore thickness})^2*[sigmoid(z-z1_{left})*sigmoid(z1_{right}-z)+sigmoid(z-z2_{left})*sigmoid(z2_{right}-z)]}}$$

$${\huge \frac{\partial ^2A_θ}{\partial r^2}+\frac{1}{r}*\frac{\partial A_θ}{\partial r}+\frac{\partial ^2A_θ}{\partial z^2}-\frac{1}{r^2}*A_θ+μ
*J_θ=0}$$

## Boundary condition

### outer boundary:

$${\huge \frac{\partial A_θ}{\partial r}=0}$$

### inner boundary

$${\huge A_θ = 0}$$

## Current density smooth approximation

### current density along z axis
<img src="current_density.png" alt="current density" width="240" />

### differentiable smooth approximation of current density 
<img src="current_density_smooth_approximation.png" alt="current density smooth approximation" width="240" />