% Define Lambda Polynomial (Highest Degree First)
Lambda = [0.3435, 3.164e-6, 2.3e-6, 1.372e-6, 3.844e-7, 0, 0, 0, 0, 0, 0, 0, 0.03874, 0.2021, 0.1395, 0.276];
Lambda = Lambda / sum(Lambda);  % Normalize Lambda

Rho = [0.49086162, 0.50913838,0, 0, 0, 0, 0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0];

% Design rate
design_rate = 0.744;

% Calculate Lambda'(1) and Rho'(1)
Lambda_prime = dot(1:length(Lambda), fliplr(Lambda));
Rho_prime = dot(1:length(Rho), fliplr(Rho));

disp('Lambda Prime (1):');
disp(Lambda_prime);
disp('Rho Prime (1):');
disp(Rho_prime);

% Compute Number of Variable Nodes (N)
N = ceil((Rho_prime / (1 - design_rate))^2);
disp('Computed Number of Variable Nodes (N):');
disp(N);

% Check total edges
edges_lambda = N * sum((1:length(Lambda)) .* Lambda);
edges_rho = edges_lambda / sum((1:length(Rho)) .* Rho);

disp('Total edges (Lambda):');
disp(edges_lambda);
disp('Total edges (Rho):');
disp(edges_rho);


% Adjust N for Testing
N = max(N, 5000); % Try increasing N
disp('Adjusted N:');
disp(N);

% Generate Parity-Check Matrix H
try
    H = getIrregularH(N, Lambda, Rho);
    disp('Generated Parity-Check Matrix H:');
    disp(H);

    % Verify Properties of H
    dc_avg = mean(sum(H, 2)); % Average degree of check nodes
    dv_avg = mean(sum(H, 1)); % Average degree of variable nodes

    disp('Average Check Node Degree (d_c,avg):');
    disp(dc_avg);
    disp('Average Variable Node Degree (d_v,avg):');
    disp(dv_avg);

catch ME
    disp('Error generating H:');
    disp(ME.message);
end
