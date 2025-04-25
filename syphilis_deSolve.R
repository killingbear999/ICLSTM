library(deSolve)

# Helper functions
get_C <- function(I, P, S, E) {
  I + P + S + E
}

get_N <- function(U, I, P, S, E, L, T, R) {
  U + I + P + S + E + L + T + R
}

get_pi <- function(c_target, N_target, c_remain, N_remain) {
  (c_target * N_target) / (c_target * N_target + c_remain * N_remain)
}

get_lambda <- function(t, t_0, c, beta, phi_beta, epsilon, C_target, N_target, pi_target, C_remain, N_remain, pi_remain) {
  c * beta * (1 + phi_beta * (t - t_0)) *
    (epsilon * C_target / N_target + (1 - epsilon) *
       (pi_target * C_target / N_target + pi_remain * C_remain / N_remain))
}

get_eta <- function(t, t_0, eta_H_init, phi_eta) {
  eta_H_init * (1 + phi_eta * (t - t_0))
}

# ODE system
syphilis_model <- function(t, y, parms) {
  with(as.list(c(y, parms)), {
    
    # Population sizes
    C_H <- get_C(I_N_H, P_N_H, S_N_H, E_N_H)
    C_L <- get_C(I_N_L, P_N_L, S_N_L, E_N_L)
    N_H <- get_N(U_N_H, I_N_H, P_N_H, S_N_H, E_N_H, L_N_H, T_N_H, R_N_H)
    N_L <- get_N(U_N_L, I_N_L, P_N_L, S_N_L, E_N_L, L_N_L, T_N_L, R_N_L)
    
    # Mixing
    pi_H <- get_pi(c_H, N_H, c_L, N_L)
    pi_L <- get_pi(c_L, N_L, c_H, N_H)
    lambda_H <- get_lambda(t, t_0, c_H, beta, phi_beta, epsilon, C_H, N_H, pi_H, C_L, N_L, pi_L)
    lambda_L <- get_lambda(t, t_0, c_L, beta, phi_beta, epsilon, C_L, N_L, pi_L, C_H, N_H, pi_H)
    
    # Screening
    eta_H <- get_eta(t, t_0, eta_H_init, phi_eta)
    eta_L <- omega * eta_H
    
    # ODEs - High risk
    dU_N_H <- q_H * alpha - (lambda_H + 1/gamma) * U_N_H + rho * R_N_H
    dI_N_H <- lambda_H * U_N_H - (sigma + 1/gamma) * I_N_H
    dP_N_H <- sigma * I_N_H - (mu + psi_S + 1/gamma) * P_N_H
    dS_N_H <- psi_S * P_N_H - (mu + psi_E + 1/gamma) * S_N_H
    dE_N_H <- psi_E * S_N_H - (eta_H + psi_L + 1/gamma) * E_N_H
    dL_N_H <- psi_L * E_N_H - (eta_H + psi_T + 1/gamma) * L_N_H
    dT_N_H <- psi_T * L_N_H - (mu + beta_nu * nu + 1/gamma) * T_N_H
    dR_N_H <- mu * (P_N_H + S_N_H + T_N_H) + eta_H * (E_N_H + L_N_H) - (rho + 1/gamma) * R_N_H
    
    # ODEs - Low risk
    dU_N_L <- q_L * alpha - (lambda_L + 1/gamma) * U_N_L + rho * R_N_L
    dI_N_L <- lambda_L * U_N_L - (sigma + 1/gamma) * I_N_L
    dP_N_L <- sigma * I_N_L - (mu + psi_S + 1/gamma) * P_N_L
    dS_N_L <- psi_S * P_N_L - (mu + psi_E + 1/gamma) * S_N_L
    dE_N_L <- psi_E * S_N_L - (eta_L + psi_L + 1/gamma) * E_N_L
    dL_N_L <- psi_L * E_N_L - (eta_L + psi_T + 1/gamma) * L_N_L
    dT_N_L <- psi_T * L_N_L - (mu + beta_nu * nu + 1/gamma) * T_N_L
    dR_N_L <- mu * (P_N_L + S_N_L + T_N_L) + eta_L * (E_N_L + L_N_L) - (rho + 1/gamma) * R_N_L
    
    # Return as list
    list(c(dU_N_H, dI_N_H, dP_N_H, dS_N_H, dE_N_H, dL_N_H, dT_N_H, dR_N_H,
           dU_N_L, dI_N_L, dP_N_L, dS_N_L, dE_N_L, dL_N_L, dT_N_L, dR_N_L))
  })
}

# time series of MSM syphilis cases
cases <- c(324, 365, 488, 346, 305, 197, 159, 187, 318, 308, 525, 496, 623, 573, 605)

# Initial population size of MSM in 2004
N_t0 <- 100000

# Annual MSM population entrants (at age 15)
alpha <- 2000

# Proportion of the MSM population in group j
q_H <- 0.207
q_L <- 0.793

# Annual rate of partner change in group j
c_H <- 14.866
c_L <- 1.989

# Years spent in the sexually-active population
gamma <- 50

# transmission rate
sigma <- 16 # 23 days
psi_S <- 13 # 4 weeks
psi_E <- 2 # 6 months
psi_L <- 0.5 # 2 years
psi_T <- 0.05 # 20 years
nu <- 0.025 # 40 years

# times
n_years <- length(cases) 
t <- seq(0, 100, by = 1)
t_0 = 0 
t <- t[-1]

# initial syphilis incidence guess
temp = 300

# Initial conditions (example)
U_N_H = (N_t0 - temp) * q_H;
I_N_H = 0.1 * temp * q_H;
P_N_H = 0.15 * temp * q_H;
S_N_H = 0.25 * temp * q_H;
E_N_H = 0.3 * temp * q_H;
L_N_H = 0.1 * temp * q_H;
T_N_H = 0.05 * temp * q_H;
R_N_H = 0.05 * temp * q_H;
U_N_L = (N_t0 - temp) * q_L;
I_N_L = 0.1 * temp * q_L;
P_N_L = 0.15 * temp * q_L;
S_N_L = 0.25 * temp * q_L;
E_N_L = 0.3 * temp * q_L;
L_N_L = 0.1 * temp * q_L;
T_N_L = 0.05 * temp * q_L;
R_N_L = 0.05 * temp * q_L;
y0 = c(U_N_H=U_N_H, I_N_H=I_N_H, P_N_H=P_N_H, S_N_H=S_N_H, E_N_H=E_N_H, L_N_H=L_N_H, T_N_H=T_N_H, R_N_H=R_N_H,
       U_N_L=U_N_L, I_N_L=I_N_L, P_N_L=P_N_L, S_N_L=S_N_L, E_N_L=E_N_L, L_N_L=L_N_L, T_N_L=T_N_L, R_N_L=R_N_L)

# Parameters
params <- list(
  q_H = q_H, c_H = c_H, c_L = c_L, q_L = q_L,
  t_0 = 0, alpha = alpha, gamma = gamma,
  sigma = sigma, psi_S = psi_S, psi_E = psi_E, psi_L = psi_L, psi_T = psi_T,
  nu = nu, beta = 0.1, phi_beta = 0.1, epsilon=0.1, rho=20.94, eta_H_init=0.1, 
  phi_eta=0.1, omega=0.451, mu=239.5, beta_nu=0.8
)

# Solve the system
out <- ode(y = y0, times = c(t_0, t), func = syphilis_model, parms = params)

# View output
out <- as.data.frame(out)
print(out)