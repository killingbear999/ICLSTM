functions {
  real get_C(real I_N, real P_N, real S_N, real E_N) {
    # compute infectious population size of MSM in group j
    return(I_N + P_N + S_N + E_N);
  }
  real get_N(real U_N, real I_N, real P_N, real S_N, real E_N, real L_N, real T_N, real R_N) {
    # compute population size of MSM in group j
    return(U_N + I_N + P_N + S_N + E_N + L_N + T_N + R_N);
  }
  real get_pi(real c_target, real N_target, real c_remain, real N_remain) {
    # compute proportion of all partnerships in the population that involve a member of group j
    return((c_target * N_target) / (c_target * N_target + c_remain * N_remain));
  }
  real get_lambda(real t, real t_0, real c, real beta, real phi_beta, real epsilon, real C_target, real N_target, real pi_target, real C_remain, real N_remain, real pi_remain) {
    # compute force of infection in group j
    return(c * beta * (1 + phi_beta * (t - t_0)) * (epsilon * C_target / N_target + (1 - epsilon) * (pi_target * C_target / N_target + pi_remain * C_remain / N_remain)));
  }
  real get_eta(real t, real t_0, real eta_H_init, real phi_eta) {
    # compute rate of screening in the absence of symptoms in group H
    return(eta_H_init * (1 + phi_eta * (t - t_0)));
  }
  real[] syphilis_model(real t, real[] y, real[] theta, real[] x_r, int[] x_i) {
      # initial conditions
      real U_N_H = y[1];
      real I_N_H = y[2];
      real P_N_H = y[3];
      real S_N_H = y[4];
      real E_N_H = y[5];
      real L_N_H = y[6];
      real T_N_H = y[7];
      real R_N_H = y[8]; # R_N_H
      real U_N_L = y[9];
      real I_N_L = y[10];
      real P_N_L = y[11];
      real S_N_L = y[12];
      real E_N_L = y[13];
      real L_N_L = y[14];
      real T_N_L = y[15];
      real R_N_L = y[16]; # R_N_L
      
      # fixed real parameters
      real q_H = x_r[1]; # Proportion of the MSM population in group H
      real c_H = x_r[2]; # Annual rate of partner change in group H
      real c_L = x_r[3]; # Annual rate of partner change in group L
      real t_0 = x_r[4]; # Initial time
      real q_L = x_r[5]; # Proportion of the MSM population in group L
      
      # fixed integer parameters
      int alpha = x_i[1]; # Annual MSM population entrants (at age 15)
      int N_t0 = x_i[2]; # Initial population size of MSM
      int gamma = x_i[3]; # Years spent in the sexually-active population
      
      # model parameters
      real beta = theta[1]; # Probability of transmission per partnership
      real phi_beta = theta[2]; # Annual increase in transmission risk behaviour
      real epsilon = theta[3]; # Level of assortativity in sexual mixing
      real rho = theta[4]; # Rate of recovery after treatment
      real eta_H_init = theta[5]; # Initial rate of asymptomatic screening in group H
      real phi_eta = theta[6]; # Annual increase in screening rate
      real omega = theta[7]; # Ratio of screening rate in group L vs group H
      real sigma = theta[8]; # Rate of leaving incubation period
      real mu = theta[9]; # Rate of seeking treatment due to symptoms 
      real psi_S = theta[10]; # Rate of leaving primary stage
      real psi_E = theta[11]; # Rate of leaving secondary stage
      real psi_L = theta[12]; # Rate of leaving early latent stage
      real psi_T = theta[13]; # Rate of leaving late latent stage
      real nu = theta[14]; # Mortality rate at tertiary stage
      real kappa_D = theta[15]; # Shape parameter of communicable disease surveillance data
      real beta_nu = theta[16]; # Probability of death at tertiary stage
      
      // print("t:", t);
      // // print("mu:", mu);
      // print("psi_S:", psi_S);
      // print("psi_E:", psi_E);
      // print("psi_L:", psi_L);
      // print("psi_T:", psi_T);
      // // print("rho:", rho);
      // print("U:", U_N_H);
      // print("I:", I_N_H);
      // print("P:", P_N_H);
      // print("S:", S_N_H);
      // print("E:", E_N_H);
      // print("L:", L_N_H);
      // print("T:", T_N_H);
      // print("R:", R_N_H);
      // 
      // print("U:", U_N_L);
      // print("I:", I_N_L);
      // print("P:", P_N_L);
      // print("S:", S_N_L);
      // print("E:", E_N_L);
      // print("L:", L_N_L);
      // print("T:", T_N_L);
      // print("R:", R_N_L);
      
      # time-dependent variables
      real C_H = get_C(I_N_H, P_N_H, S_N_H, E_N_H); # Number of infectious individuals in group H
      real C_L = get_C(I_N_L, P_N_L, S_N_L, E_N_L); # Number of infectious individuals in group L
      real N_H = get_N(U_N_H, I_N_H, P_N_H, S_N_H, E_N_H, L_N_H, T_N_H, R_N_H); # Total number of MSM population in group H
      real N_L = get_N(U_N_L, I_N_L, P_N_L, S_N_L, E_N_L, L_N_L, T_N_L, R_N_L); # Total number of MSM population in group L
      real pi_H = get_pi(c_H, N_H, c_L, N_L); # Proportion of all partnerships in the population that involve a member of group H
      real pi_L = get_pi(c_L, N_L, c_H, N_H); # Proportion of all partnerships in the population that involve a member of group L
      real lambda_H = get_lambda(t, t_0, c_H, beta, phi_beta, epsilon, C_H, N_H, pi_H, C_L, N_L, pi_L); # Force of infection in group H
      real eta_H = get_eta(t, t_0, eta_H_init, phi_eta); # Rate of screening in the absence of symptoms in group H
      real eta_L = omega * eta_H; # Rate of screening in the absence of symptoms in group L
      real lambda_L = get_lambda(t, t_0, c_L, beta, phi_beta, epsilon, C_L, N_L, pi_L, C_H, N_H, pi_H); # Force of infection in group L
      
      // print("lambda_H:", lambda_H);
      // print("lambda_L:", lambda_L);
      // print("eta_H:", eta_H);
      // print("eta_L:", eta_L);
      // print("C_H:", C_H);
      // print("C_L:", C_L);
      // print("total population:", N_H+N_L);
      
      # ODEs
      # debug: 1. when setting alpha, gamma, nu values to be 0, i.e., no inflow and outflow, the total population remains unchanged --> the odes have no problem.
      # high-risk group
      # non-doxy-pep (N)
      real dU_N_H = q_H * alpha - (lambda_H + 1/gamma) * fmax(1e-8, U_N_H) + rho *fmax(1e-8,  R_N_H);
      real dI_N_H = lambda_H * fmax(1e-8, U_N_H) - (sigma + 1/gamma) * fmax(1e-8, I_N_H);
      real dP_N_H = sigma * fmax(1e-8, I_N_H) - (beta_nu * mu + (1 - beta_nu) * psi_S + 1/gamma) * fmax(1e-8, P_N_H);
      real dS_N_H = (1 - beta_nu) * psi_S * fmax(1e-8, P_N_H) - (beta_nu * mu + (1 - beta_nu) * psi_E + 1/gamma) * fmax(1e-8, S_N_H);
      real dE_N_H = (1 - beta_nu) * psi_E * fmax(1e-8, S_N_H) - (beta_nu * eta_H + (1 - beta_nu) * psi_L + 1/gamma) * fmax(1e-8, E_N_H);
      real dL_N_H = (1 - beta_nu) * psi_L * fmax(1e-8, E_N_H) - (beta_nu * eta_H + (1 - beta_nu) * psi_T + 1/gamma) * fmax(1e-8, L_N_H);
      real dT_N_H = (1 - beta_nu) * psi_T * fmax(1e-8, L_N_H) - (beta_nu * mu + (1 - beta_nu) * nu + 1/gamma) * fmax(1e-8, T_N_H);
      real dR_N_H = beta_nu * mu * (fmax(1e-8, P_N_H) + fmax(1e-8, S_N_H) + fmax(1e-8, T_N_H)) + beta_nu * eta_H * (fmax(1e-8, E_N_H) + fmax(1e-8, L_N_H)) - (rho + 1/gamma) * fmax(1e-8, R_N_H);
      
      // print("inflow:", psi_L * E_N_H);
      // print("outflow:", (beta_nu * eta_H + (1 - beta_nu) * psi_T + 1/gamma) * L_N_H);
      // print("dL_N_H:", dL_N_H);

      # low-risk group
      # non-doxy-pep (N)
      real dU_N_L = q_L * alpha - (lambda_L + 1/gamma) * fmax(1e-8, U_N_L) + rho * fmax(1e-8, R_N_L);
      real dI_N_L = lambda_L * fmax(1e-8, U_N_L) - (sigma + 1/gamma) * fmax(1e-8, I_N_L);
      real dP_N_L = sigma * fmax(1e-8, I_N_L) - (beta_nu * mu + (1 - beta_nu) * psi_S + 1/gamma) * fmax(1e-8, P_N_L);
      real dS_N_L = (1 - beta_nu) * psi_S * fmax(1e-8, P_N_L) - (beta_nu * mu + (1 - beta_nu) * psi_E + 1/gamma) * fmax(1e-8, S_N_L);
      real dE_N_L = (1 - beta_nu) * psi_E * fmax(1e-8, S_N_L) - (beta_nu * eta_L + (1 - beta_nu) * psi_L + 1/gamma) * fmax(1e-8, E_N_L);
      real dL_N_L = (1 - beta_nu) * psi_L * fmax(1e-8, E_N_L) - (beta_nu * eta_L + (1 - beta_nu) * psi_T + 1/gamma) * fmax(1e-8, L_N_L);
      real dT_N_L = (1 - beta_nu) * psi_T * fmax(1e-8, L_N_L) - (beta_nu * mu + (1 - beta_nu) * nu + 1/gamma) * fmax(1e-8, T_N_L);
      real dR_N_L = beta_nu * mu * (fmax(1e-8, P_N_L) + fmax(1e-8, S_N_L) + fmax(1e-8, T_N_L)) + beta_nu * eta_L * (fmax(1e-8, E_N_L) + fmax(1e-8, L_N_L)) - (rho + 1/gamma) * fmax(1e-8, R_N_L);
      
      // if (L_N_H < 0) {
      //   reject("L_N_H must not be negative; found L_N_H=", L_N_H);
      // }
      // 
      return {dU_N_H, dI_N_H, dP_N_H, dS_N_H, dE_N_H, dL_N_H, dT_N_H, dR_N_H, dU_N_L, dI_N_L, dP_N_L, dS_N_L, dE_N_L, dL_N_L, dT_N_L, dR_N_L};
  }
}
data {
  int<lower=1> n_years; # Number of years modelling
  real<lower=0> y0[16]; # Initial conditions for all compartments
  real ts[n_years]; # Sequences of time steps
  real t_0;
  real q_H;
  real c_H;
  real c_L;
  real q_L;
  int cases[n_years]; # Annual syphilis cases
  int alpha;
  int N_t0;
  int gamma;
}
transformed data {
  real x_r[5];
  int x_i[3];
  
  # assign values to x_r
  x_r[1] = q_H;
  x_r[2] = c_H;
  x_r[3] = c_L;
  x_r[4] = t_0;
  x_r[5] = q_L;
  
  # assign values to x_i
  x_i[1] = alpha;
  x_i[2] = N_t0;
  x_i[3] = gamma;
}
parameters {
  real<lower=0, upper=1> beta;
  real<lower=0, upper=1> phi_beta;
  real<lower=0, upper=1> epsilon;
  real<lower=10, upper=50> rho;
  real<lower=0, upper=4> eta_H_init;
  real<lower=0, upper=1> phi_eta;
  real<lower=0.1, upper=0.9> omega;
  real<lower=5, upper=40> sigma;
  real<lower=50, upper=600> mu;
  real<lower=3, upper=25> psi_S;
  real<lower=0.5, upper=5> psi_E;
  real<lower=0.1, upper=1.5> psi_L;
  real<lower=0.01, upper=0.2> psi_T;
  real<lower=0.01, upper=0.1> nu;
  real<lower=0, upper=1> kappa_D;
  real<lower=0.8, upper=1> beta_nu;
}
transformed parameters{
  real y[n_years, 16];
  real incidence[n_years - 1];
  real theta[16];
  theta[1] = beta;
  theta[2] = phi_beta;
  theta[3] = epsilon;
  theta[4] = rho;
  theta[5] = eta_H_init;
  theta[6] = phi_eta;
  theta[7] = omega;
  theta[8] = sigma;
  theta[9] = mu;
  theta[10] = psi_S;
  theta[11] = psi_E;
  theta[12] = psi_L;
  theta[13] = psi_T;
  theta[14] = nu;
  theta[15] = kappa_D;
  theta[16] = beta_nu;

  y = integrate_ode_rk45(syphilis_model, y0, t_0, ts, theta, x_r, x_i);
  for (t in 1:(n_years - 1)) {
    // Trapezoidal rule: (f(a) + f(b)) / 2 * (b - a)
    incidence[t] = 0.5 * rho * (y[t, 8] + y[t + 1, 8] + y[t, 16] + y[t + 1, 16]);
  }
  // print("predicted_cases: ", incidence);
}
model {
  # priors
  beta ~ uniform(0,1);
  phi_beta ~ uniform(0,1);
  epsilon ~ uniform(0,1);
  rho ~ lognormal(log(20), 0.3); # 20.94 (11.1, 35.9)
  eta_H_init ~ uniform(0,4);
  phi_eta ~ uniform(0,1);
  omega ~ lognormal(-0.87,0.39); # 0.451 (0.22, 0.80)
  sigma ~ lognormal(log(15), 0.5); # 17 (6.6, 34.1)
  mu ~ lognormal(log(200), 0.6); # 239.5 (74.6, 535.9)
  psi_S ~ lognormal(log(10), 0.4); # 10.83 (5.18, 19.19)
  psi_E ~ lognormal(log(2), 0.4); # 2.17 (1.04, 3.86)
  psi_L ~ lognormal(log(0.5), 0.4); # 0.542 (0.26, 0.97)
  psi_T ~ lognormal(log(0.05), 0.7); # 0.064 (0.016, 0.158)
  nu ~ lognormal(log(0.04), 0.3); # 0.0419 (0.024, 0.066)
  kappa_D ~ uniform(0,1);
  beta_nu ~ uniform(0.8,0.9);
  
  # sampling distribution
  cases[1:(n_years-1)] ~ neg_binomial_2(incidence, kappa_D);
}
generated quantities {
  real pred_cases[n_years-1];
  pred_cases = neg_binomial_2_rng(incidence, kappa_D);
}
