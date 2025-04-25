library(tidyverse)
library(rstan)
library(ggplot2)
library(gridExtra)
rstan_options (auto_write = TRUE)
options (mc.cores = parallel::detectCores ())

# time series of MSM syphilis cases
cases <- c(324, 365, 488, 346, 305, 197, 159, 187, 318, 308, 525, 496, 623, 573, 605)

# Initial population size of MSM in 2004
N_t0 <- 97189

# Annual MSM population entrants (at age 15)
alpha <- 2524

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
t <- seq(0, n_years, by = 1)
t_0 = 0 
t <- t[-1]

# initial syphilis incidence guess
temp = 324

# initial conditions
# assume that for each group j
# 10% of incidences at incubation
# 20% of incidences at primary stage
# 25% of incidences at secondary stage
# 30% of incidences at early latent stage
# 10% of incidences at late latent stage
# 5% of incidences at teritary stage
# 5% of incidences at recovery stage
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

# data for Stan
data_syphilis <- list(n_years = n_years, y0 = y0, ts = t, t_0 = t_0, q_H = q_H, c_H = c_H, c_L = c_L, q_L = q_L, cases = cases, alpha = alpha, N_t0 = N_t0, gamma = gamma, sigma=sigma, psi_S=psi_S, psi_E=psi_E, psi_L=psi_L, psi_T=psi_T, nu=nu)

# number of MCMC steps
niter <- 10000

# parameter initialization for rstan for debugging
init_fun <- function() {
  list(beta = 0.1, phi_beta = 0.1, epsilon=0.1, rho=20.94, eta_H_init=0.1, phi_eta=0.1, omega=0.451, mu=239.5, kappa_D=0.1, beta_nu=0.8)
}

# run MCMC
model <- stan_model("syphilis_fixed_transmission_parameters.stan")
fit_syphilis_negbin <- sampling(model,
                                data = data_syphilis,
                                # algorithm = "Fixed_param",
                                iter = niter,
                                chains = 4, 
                                seed = 42,
                                # init=init_fun,
                                # diagnostic_file = "diagnostics.csv"
                                verbose=TRUE)

# print the mcmc results
pars=c('beta', 'phi_beta', 'epsilon')
print(fit_syphilis_negbin, pars = pars)

# trace plots to assess mixing of a chain
traceplot(fit_syphilis_negbin, pars = pars)

# marginal posterior densities
stan_dens(fit_syphilis_negbin, pars = pars, separate_chains = TRUE)
