% Example Lambda and Rho Distributions
Lambda = [0.3435, 3.164e-6, 2.3e-6, 1.372e-6, 3.844e-7, 0, 0, 0, 0, 0, 0, 0, 0.03874, 0.2021, 0.1395, 0.276];
Rho = [0.49086162, 0.50913838,0, 0, 0, 0, 0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0];

% Design rate
design_rate = 0.744;

% Calculate Lambda'(1) and Rho'(1)
Lambda_prime = dot(1:length(Lambda), fliplr(Lambda));
Rho_prime = dot(1:length(Rho), fliplr(Rho));

N = ceil((Rho_prime / (1 - design_rate))^2);
% Construct H and G
[H, G] = constructLDPC(N, Lambda, Rho);

% Display H and G
disp('Parity-Check Matrix (H):');
disp(full(H));

disp('Generator Matrix (G):');
disp(full(G));
