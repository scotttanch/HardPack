# Kalman Filter Note Sheet
I'll create 3 different Kalman Filters:
 1. IMU and Realsense
 2. IMU, Realsense, UWB (Position)
 3. IMU, Realsense, UWB (Landmark)

## State Transintion Function

The state function, specifically the state transition matrix, will be the same for all three and be based on the constant
acceleration kinematic equations. This gives nine state variables, being the x, y, and z components of position, velocity
and acceleration. The equations are given as:

$$
p_{i_{k+1}} = p_{i_{k}} + v_{i_{k}} \cdot \Delta t + \frac{1}{2} \cdot a_{i_{k}} \Delta t^2
$$

$$
v_{i_{k+1}} = v_{i_{k}} + a_{i_{k}} \cdot \Delta t
$$

$$
a_{i_{k+1}} = a_{i_{k}}
$$

where $i=x,y,z$

## Observation Models
### Case 1: IMU and RealSense (Standard Kalman Filter)
In this model, there will be 12 measured variables: position, velocity, and accleration as measured by the RealSense and
acceleration measured by the Bosch IMU. This first most simple case has a simple observation model. It is similar to 
the identity matrix with the additon of duplicate rows for the x,y,z components of acceleration
duplicate rows for linear acceleration leading to a [15 x 9] matrix

     H = [[1,0,0,0,0,0,0,0,0,0,0,0],
          [0,1,0,0,0,0,0,0,0,0,0,0],
          [0,0,1,0,0,0,0,0,0,0,0,0],
          [0,0,0,1,0,0,0,0,0,0,0,0],
          [0,0,0,0,1,0,0,0,0,0,0,0],
          [0,0,0,0,0,1,0,0,0,0,0,0],
          [0,0,0,0,0,0,1,0,0,0,0,0],
          [0,0,0,0,0,0,0,1,0,0,0,0],
          [0,0,0,0,0,0,0,0,1,0,0,0],
          [0,0,0,0,0,0,0,0,0,1,0,0],
          [0,0,0,0,0,0,0,0,0,1,0,0],
          [0,0,0,0,0,0,0,0,0,0,1,0],
          [0,0,0,0,0,0,0,0,0,0,1,0],
          [0,0,0,0,0,0,0,0,0,0,0,1],
          [0,0,0,0,0,0,0,0,0,0,0,1]]

### Case 2.1: IMU, RealSense, UWB Position System (Standard Kalman Filter)
This is an interesting case, which I expect to have poorer performance than the following. This system will integrate the
ftm capability of the ESP-32 UWB to measure distance to another UWB. With 4 devices one can triangulate (or trilaterate)
a position measurement in a manner similar to the GPS system. We will take the position calculated rather than the raw
measurements.

This system will now have 15 measured values and, similar to case 1, will use an observation model like the Identity matrix
but with duplicate rows for the x,y,z components of position coming from the RealSense and UWB system. This yeilds an [18 x 9] matrix

     H = [[1,0,0,0,0,0,0,0,0,0,0,0],
          [1,0,0,0,0,0,0,0,0,0,0,0],
          [0,1,0,0,0,0,0,0,0,0,0,0],
          [0,1,0,0,0,0,0,0,0,0,0,0],
          [0,0,1,0,0,0,0,0,0,0,0,0],
          [0,0,1,0,0,0,0,0,0,0,0,0],
          [0,0,0,1,0,0,0,0,0,0,0,0],
          [0,0,0,0,1,0,0,0,0,0,0,0],
          [0,0,0,0,0,1,0,0,0,0,0,0],
          [0,0,0,0,0,0,1,0,0,0,0,0],
          [0,0,0,0,0,0,0,1,0,0,0,0],
          [0,0,0,0,0,0,0,0,1,0,0,0],
          [0,0,0,0,0,0,0,0,0,1,0,0],
          [0,0,0,0,0,0,0,0,0,1,0,0],
          [0,0,0,0,0,0,0,0,0,0,1,0],
          [0,0,0,0,0,0,0,0,0,0,1,0],
          [0,0,0,0,0,0,0,0,0,0,0,1],
          [0,0,0,0,0,0,0,0,0,0,0,1]]

This model will have some complexity advantages due to H needing to only be created once, as well as being entirely linear.
However, the UWB system will be treated as a black box, either needing its own filtering or be subject to noise from one anchor
affecting all others. These tradeoffs will be addressed by case 2.2.

### Case 2.2: IMU, Realsene, UWB Landmark System (Extended Kalman Filter)
This last case extends from case 2.2, however it treats the distance measurement to each UWB anchor individually. I expect this
to have better perfomance than 2.1 as one bad anchor reading doesnt taint the rest. The tradeoff comes from this filter becoming
a non-linear Extened Kalman Filter. We shall see how the performance compares.

The observation model will be the same as Case 1 with additonal rows for each anchor added (another potential benefit, scalable anchors).
The measurement $D_n$ is the linear distance from the current position to anchor n. The equation is given as:
$$
D_n = \sqrt{(x_n - x_{k|k-1})^2 + (y_n - y_{k|k-1})^2 + (z_n - z_{k|k-1})^2}
$$
where $(x_n,y_n,z_n)$ is the position of anchor n. We linearize this equation by taking the partial derivative of $D_n$ with respect to 
each of the state variables. 

$$
\frac{\partial D_n}{\partial x_n} = \frac{x_{k|k-1} - x_n }{\sqrt{(x_n - x_{k|k-1})^2 + (y_n - y_{k|k-1})^2 + (z_n - z_{k|k-1})^2}}
$$

$$
\frac{\partial D_n}{\partial y_n} = \frac{y_{k|k-1} - y_n }{\sqrt{(x_n - x_{k|k-1})^2 + (y_n - y_{k|k-1})^2 + (z_n - z_{k|k-1})^2}}
$$

$$
\frac{\partial D_n}{\partial z_n} = \frac{z_{k|k-1} - z_n }{\sqrt{(x_n - x_{k|k-1})^2 + (y_n - y_{k|k-1})^2 + (z_n - z_{k|k-1})^2}}
$$

These partial deriviatves will need to be calcualted each loop, thus H will need to be reconstructed each loop as well. This observation model will
have a variable size [(15+n) x 9] where n is the number of anchors.

# Kalman Filter Steps:
1. Predicted State Estimate

    $\hat{x}_{k|k-1} = F \cdot x_{k-1} + v_f $

2. Predicted State Covariance 

    $P_{k|k-1} = F_{k}P_{k-1}F^{T}_{k} + Q_k$

3. Measurment Residiual
    
    $\tilde{y_k} = z_k - H_k\hat{x}_{k|k-1} + w_k$

4. Innovation Covariance
    
    $S_k = H_kP_{k|k-1}H^{T}_{k} + R_k$

5. Kalman Gain
    
    $K_k = P_{k|k-1}H^{T}_{k}S^{-1}_k$

6. Correct Estimate

    $x_k = \hat{x}_{k|k-1} + K_k\tilde{y}_k$

7. Update State Covariance
    
    $P_k = (I - K_kH_k)P_{k|k-1}$

Repeat these steps for all measurements
