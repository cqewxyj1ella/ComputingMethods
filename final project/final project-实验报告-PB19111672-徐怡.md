# final project
**姓名：** 徐怡
**学号：** PB19111672
## Q1
当$\alpha=0$时，对(1)采用积分格式：
$$
\begin{align*}
Z(u_1,...,u_n)&=min_{u_1,...,u_{n-1}}\sum_{i=1}^{n}\frac{1}{2}(\frac{u_i-u_{i-1}}{h})^2h+\sum_{i=1}^{n-1}f_iu_ih\\
&=min_{u_1,...,u_{n-1}}\sum_{i=1}^{n}\frac{1}{2h}(u_i^2+u_{i-1}^2-2u_iu_{i-1})+\sum_{i=1}^{n-1}f_iu_ih\\
&=min_{u_1,...,u_{n-1}}\sum_{i=1}^{n-1}\frac{1}{h}(u_i^2-u_iu_{i-1})+\sum_{i=1}^{n-1}f_iu_ih\\
&=min_{u_1,...,u_{n-1}}\sum_{i=1}^{n-1}\frac{1}{h}(u_i^2-u_iu_{i-1}+f_iu_ih^2)
\end{align*}
$$
上面的推导过程用到了题中所给的$u_0=u_n=0$这个条件。
对$Z$求偏导：
$$
\begin{align*}
&\frac{\partial Z(u_1,...,u_n)}{\partial u_i} = \frac{1}{h}(2u_i-u_{i-1}-u_{i+1}+f_ih^2)=0\\
&\frac{1}{h^2}(2u_i-u_{i-1}-u_{i+1})=f_i
\end{align*}
$$
所以线性方程组$A_hu_h=f_h$对应的系数矩阵$A_h$：
$$
A_h=\frac{1}{h^2}
\begin{pmatrix}  
2 & -1 & 0 & ...& 0 & 0\\  
-1 & 2 & -1 &...& 0 & 0\\
\vdots & \vdots & \vdots & \ddots & \vdots & \vdots\\
0 & 0 & 0 & \cdots & -1 & 2
\end{pmatrix}
$$
## Q2
