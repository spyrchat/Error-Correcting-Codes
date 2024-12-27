% Simulation parameters
n = 49; % Length of the codeword
lambda_dist_designed = [0.3442, 0, 0.276, 0, 0.1383, 0.21, 0, 0, 0, 0.03145, 0, 0, 0, 0, 1.715e-6, 0.3442]; % VN degree distribution
rho_dist_designed = [0.5, 0.5]; % CN degree distribution
erasure_thresholds = linspace(0.1, 1.0, 50); % Erasure thresholds
snr_values = [3, 5, 10]; % SNR values
output_dir = 'plots'; % Directory for saving plots
mkdir(output_dir);

% Function to generate irregular LDPC code
function H = generateIrregularLDPC(n, lambda_dist, rho_dist)
    vn_degrees = randsample(1:length(lambda_dist), n, true, lambda_dist);
    m = round(n * sum(lambda_dist .* (1:length(lambda_dist))) / sum(rho_dist .* (1:length(rho_dist))));
    cn_degrees = randsample(1:length(rho_dist), m, true, rho_dist);
    H = zeros(m, n);
    edges = [];
    for i = 1:n
        edges = [edges; repmat(i, vn_degrees(i), 1)];
    end
    edges = edges(randperm(length(edges)));
    edge_idx = 1;
    for j = 1:m
        for k = 1:cn_degrees(j)
            if edge_idx <= length(edges)
                H(j, edges(edge_idx)) = 1;
                edge_idx = edge_idx + 1;
            end
        end
    end
end

% Main simulation loop
for snr = snr_values
    fprintf('Running simulation for SNR = %d...\n', snr);

    % Generate LDPC matrix
    H = generateIrregularLDPC(n, lambda_dist_designed, rho_dist_designed);

    % Placeholder for results (replace with real LDPC encoding and decoding simulation)
    ser_results = exp(-erasure_thresholds * snr); % Mock SER results
    bit_rate_results = 1 - ser_results; % Mock bit rate results

    % Plotting results
    figure('Visible', 'off');
    subplot(1, 2, 1);
    semilogy(erasure_thresholds, ser_results, '-o');
    title(['Symbol Error Rate vs. Erasure Threshold (SNR = ', num2str(snr), ')']);
    xlabel('Erasure Threshold');
    ylabel('Symbol Error Rate');
    grid on;

    subplot(1, 2, 2);
    semilogy(erasure_thresholds, bit_rate_results, '-o');
    title(['Bit Rate vs. Erasure Threshold (SNR = ', num2str(snr), ')']);
    xlabel('Erasure Threshold');
    ylabel('Bit Rate');
    grid on;

    % Save plot
    plot_filename = fullfile(output_dir, ['results_snr_', num2str(snr), '.png']);
    saveas(gcf, plot_filename);
    fprintf('Saved plot at: %s\n', plot_filename);

    % Save results to CSV
    csv_filename = fullfile(output_dir, ['results_snr_', num2str(snr), '.csv']);
    results = [erasure_thresholds', ser_results', bit_rate_results'];
    writematrix(results, csv_filename, 'Delimiter', ',');
    fprintf('Saved CSV results at: %s\n', csv_filename);
end

fprintf('All simulations completed.\n');
