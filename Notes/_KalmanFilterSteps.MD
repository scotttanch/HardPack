# Kalman Filter Operational Steps
Overview of the operation of a kalman filter

## Nomenclature
|       Variable       |                     Description                     |
|:--------------------:|:---------------------------------------------------:|
| $\hat{x}_{t_1\|t_2}$ | Estimation of state at time 2 given state at time 1 |
|         $F$          |                    Process Model                    |
|         $v$          |                    Process Noise                    |
|         $P$          |            Prediction Covariance Matrix             |
|         $Q$          |          State Estimate Covariance Matrix           |
|   $\tilde{y_{k}}$    |                      Inovation                      |
|         $H$          |                  Observation Model                  |
|        $z_k$         |                  Measuremet Vector                  |
|        $w_k$         |                  Measurement Noise                  |
|        $S_k$         |                Inovation Covariances                |
|        $R_k$         |              Sesnsor Noise Covariance               |
|        $K_k$         |                     Kalman Gain                     |

## Operations

1. State estimation:

   ## $\hat{x}_{k|k-1} = F_{k-1}x_{k-1} + v_{k-1}$

2. Predicted Covariance of State Estimate:

   ## $P_{k|k-1} = F_kP_{k-1|k-1}F^{T}_{k} + Q_k$

3. Innovation
    
   ## $\tilde{y}_k = z_k - H\hat{x}_{k|k-1} + w_k$

4. Inovation Covariance
   
   ## $S_k = H_kP_{k|k-1}H^{T}_{k} + R_k$

5. Kalman Gain

    ## $K_k =P_{k|k-1}H^{T}_{k}S^{-1}_{k}$

6. Updated State Estimate

    ## $\hat{x}_{k|k} = \hat{x}_{k|k-1} + K_{k}\tilde{y}_{k}$

7. Updated Covariance of the State Estimate
    
    ## $P_{k|k} = (I - K_{k}H_{k})P_{k|k-1}$

## Detailed Overiew of Kalman Filter Variables

### ${x}_{t_1}$
A vector containing the state space variables at time $t_1$. This is synonymus with
$\hat{x}_{t_1|t_1}$

### $\hat{x}_{t_2|t_1}$
State vector estimate at time $t_2$ based on the state at time $t_1$. Obtained by
through step 1.

### $F$
F is the plant process model, which is a state transition matrix. This matrix takes
the state at some time $t_1$ and estimates the state at time $t_2$.

### $v$
v is the process noise, which is added to the updated 
