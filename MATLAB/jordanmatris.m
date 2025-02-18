% Cameron Schofield - jordanmatris.m
% Compute Jordan normal form of matrix A, within tolerance
function J = jordanmatris(A, tol)
    if nargin < 2, tol = 0.0001; end
    [ev, mult] = heltalsev(A, tol);

    % Start with a zero matrix the same size as A
    J = zeros(size(A, 2));

    row = 1;

    % Iterate through unique eigenvalues
    for i = 1:length(ev)
        % T = A - lambda * I
        T = A - ev(i) * eye(size(A, 2));
        % Matrix of ps
        p = zeros(1, size(A, 2) + 1);
        
        % p(j) <= total number of blocks for given eigenvalue <=
        % multiplicity of that eigenvalue, so only some p(j) need to be
        % calculated
        for j = 1:(mult(i) + 1)
            % p(j) = dim Ker (A - lambda * I)^j
            p(j) = size(null(T^j, tol), 2);            
        end

        % Complete rest of p matrix
        for j = (mult(i) + 1):length(p)
            p(j) = p(j - 1);
        end

        % b(j) = n(j) + n(j+1) + n(j+2) + ...
        b = [p(1), diff(p)];

        % n(j) = number of Jordan blocks of size j for the eigenvalue
        n = diff(b) * -1;
    
        % Add all Jordan blocks to matrix
        for j = 1:length(n)
            k = 0;
            % Add n(j) Jordan blocks of size j
            while k < n(j)
                % Iterate through all j rows in the block
                for l = row:(row + j - 1)
                    % Set diagonal element to eigenvalue
                    J(l, l) = ev(i);
                    if l ~= row + j - 1
                        % Set element to left of diagonal to 1
                        % (if it is not the last row in the block)
                        J(l, l + 1) = 1;
                    end
                end
                % Keep track of how many rows have been used
                row = row + j;
                k = k + 1;
            end
        end
    end
end

% Compute eigenvalues of A with their algebraic multiplicity
function [ev, mult] = heltalsev(A, tol)
    if nargin < 2, tol = 0.0001; end

    % Compute eigenvalues of A
    e = eig(A);
    
    % Look through resulting eigenvalues
    all_integers = true;
    for i = 1:length(e)
        % Eigenvalue rounded to nearest integer
        round_e = round(e(i));
        % Check difference between computed eigenvalue and nearest integer
        if abs(round_e - e(i)) > tol
            % If difference exceeds tolerance, eigenvalue cannot be
            % considered to be an approximate integer
            all_integers = false;
        end
    end

    ev = [];
    mult = [];
    % If all eigenvalues are integers, list them and their multiplicities
    if all_integers
        for i = 1:length(e)
            % Find unique eigenvalues
            unique = true;
            for j = 1:length(ev)
                % If eigenvalue is known, just increment corresponding
                % multiplicity
                if round(e(i)) == ev(j)
                    mult(j) = mult(j) + 1;
                    unique = false;
                end
            end
            % If eigenvalue is unique (so far), add it to list and set its
            % multiplicity as 1
            if unique
                ev(end+1) = round(e(i));
                mult(end+1) = 1;
            end
        end
    end
end
