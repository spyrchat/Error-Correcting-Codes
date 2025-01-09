function H = constructIrregularLDPC(N, Lambda, Rho)
    % Constructs an irregular LDPC parity-check matrix (H)
    % based on degree distributions Lambda and Rho.

    % Normalize Lambda and Rho
    Lambda = Lambda / sum(Lambda);
    Rho = Rho / sum(Rho);

    % Compute Lambda'(1) and Rho'(1)
    Lambda_prime = dot(1:length(Lambda), fliplr(Lambda));
    Rho_prime = dot(1:length(Rho), fliplr(Rho));

    % Compute the total number of edges
    E = round(N * Lambda_prime);  % Total edges from variable nodes

    % Compute the number of check nodes (m)
    m = round(E / Rho_prime);

    % Debugging: Display total edges and check nodes
    disp('Total edges (E):');
    disp(E);
    disp('Number of check nodes (m):');
    disp(m);

    % Step 1: Generate degrees for variable nodes
    variable_degrees = generateDegrees(N, Lambda, E);

    % Step 2: Generate degrees for check nodes
    check_degrees = generateDegrees(m, Rho, E);

    % Ensure the total number of edges matches
    if sum(variable_degrees) ~= sum(check_degrees)
        error('Total edges for variable and check nodes do not match.');
    end

    % Step 3: Connect edges to form the parity-check matrix
    H = connectEdges(N, m, variable_degrees, check_degrees);
end

function degrees = generateDegrees(num_nodes, distribution, total_edges)
    % Generate degrees for nodes based on the degree distribution
    degrees = [];
    for i = 1:length(distribution)
        num_sockets = round(distribution(i) * total_edges);  % Allocate edges proportionally
        degrees = [degrees, repelem(i, num_sockets)];  % Assign degrees
    end

    % Shuffle and truncate to match the total number of nodes
    degrees = degrees(randperm(length(degrees)));  % Shuffle degrees
    degrees = degrees(1:num_nodes);  % Truncate to match the number of nodes
end

function H = connectEdges(N, m, variable_degrees, check_degrees)
    % Connect edges between variable and check nodes to construct H

    % Total edges
    total_edges = sum(variable_degrees);

    % Create sockets for variable and check nodes
    variable_sockets = repelem(1:N, variable_degrees);
    check_sockets = repelem(1:m, check_degrees);

    % Randomly shuffle and connect edges
    perm = randperm(total_edges);  % Random permutation of edges
    variable_sockets = variable_sockets(perm);
    check_sockets = check_sockets(perm);

    % Construct sparse H matrix
    H = sparse(check_sockets, variable_sockets, 1, m, N);
end
