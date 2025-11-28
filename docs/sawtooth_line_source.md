# classic 2-D time-harmonic EM problem with Gaussian pulse line source

![3D view](3d_view.png) ![2D view](2d_view.png)

## Sawtooth wave formula

$$J(t) = 2J_0\frac{π}arctan[βtan(πt\frac{T})]$$

## EM formula

$$\frac{\partial ^2Az}{\partial x^2}+\frac{\partial ^2Az}{\partial y^2}-με\frac{\partial ^2Az}{\partial t^2}+μ2J_0\frac{π}arctan[βtan(πt\frac{T})]$$

## Boundary condition (Dirichlet)

target_boundary_value:  $$A_z^bc(x,y,t) = - μJ(t)\frac{2π}ln\sqrt{x^2+y^2}\frac{r_0}$$

### left

$$A_z^NN(-1,y,t) - A_z^bc(-1,y,t) = 0$$

### right

$$A_z^NN(1,y,t) - A_z^bc(1,y,t) = 0$$

### bottom

$$A_z^NN(x,-1,t) - A_z^bc(x,-1,t) = 0$$

### top

$$A_z^NN(x,1,t) - A_z^bc(x,1,t) = 0$$

## Initial condition

$$A_z(x,y,0) = - μJ(0)\frac{2π}ln\sqrt{x^2+y^2}\frac{r_0}$$