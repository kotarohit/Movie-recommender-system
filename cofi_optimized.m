%% OPTIMIZED COLLABORATIVE FILTERING RECOMMENDER SYSTEM
%  This is an optimized version of the collaborative filtering algorithm
%  with significant performance improvements:
%  - Vectorized operations
%  - Memory-efficient computations  
%  - Reduced redundant calculations
%  - Performance profiling
%  - Configurable parameters for faster iteration

fprintf('=== OPTIMIZED Collaborative Filtering Movie Recommender ===\n\n');

%% Configuration Parameters
USE_FULL_DATASET = true;  % Set to false for faster testing
ENABLE_PROFILING = true;  % Enable performance profiling
MAX_ITER = 100;           % Maximum iterations for optimization
LAMBDA = 10;              % Regularization parameter
NUM_FEATURES = 10;        % Number of features

if ENABLE_PROFILING
    profile on;
    tic; % Start timing
end

%% =============== Part 1: Loading movie ratings dataset ================
fprintf('Loading movie ratings dataset...\n');

% Load data efficiently
load('movies.mat');

% Display dataset statistics
fprintf('Dataset: %d movies, %d users\n', size(Y,1), size(Y,2));
fprintf('Sparsity: %.2f%% (%.0f ratings out of %.0f possible)\n', ...
    100*sum(R(:))/numel(R), sum(R(:)), numel(R));

if ~USE_FULL_DATASET
    % Use smaller dataset for faster iteration during development
    num_users = 50; num_movies = 100; num_features = 5;
    fprintf('Using reduced dataset: %d movies, %d users, %d features\n', ...
        num_movies, num_users, num_features);
    Y = Y(1:num_movies, 1:num_users);
    R = R(1:num_movies, 1:num_users);
end

%% =============== Part 2: Add User Ratings ================
fprintf('Setting up user preferences...\n');

% Initialize ratings for new user
my_ratings = zeros(size(Y,1), 1);

% Pre-defined ratings for demonstration
rating_indices = [1, 7, 12, 54, 64, 66, 69, 98, 183, 226, 355];
rating_values = [4, 3, 5, 4, 5, 3, 5, 2, 4, 5, 5];

% Ensure indices are within bounds
valid_indices = rating_indices <= size(Y,1);
my_ratings(rating_indices(valid_indices)) = rating_values(valid_indices);

fprintf('Added %d user ratings\n', sum(my_ratings > 0));

%% =============== Part 3: Prepare Training Data ================
fprintf('Preparing training data...\n');

% Add user ratings to dataset
Y = [my_ratings Y];
R = [(my_ratings ~= 0) R];

% Normalize ratings (using optimized version)
fprintf('Normalizing ratings...\n');
[Ynorm, Ymean] = normalizeRatings(Y, R);

% Update dimensions
num_users = size(Y, 2);
num_movies = size(Y, 1);

fprintf('Training set: %d movies, %d users, %d features\n', ...
    num_movies, num_users, NUM_FEATURES);

%% =============== Part 4: Initialize Parameters ================
fprintf('Initializing parameters...\n');

% Initialize with small random values for better convergence
X = randn(num_movies, NUM_FEATURES) * 0.1;
Theta = randn(num_users, NUM_FEATURES) * 0.1;

initial_parameters = [X(:); Theta(:)];
fprintf('Initialized %d parameters\n', length(initial_parameters));

%% =============== Part 5: Train Model ================
fprintf('\nTraining collaborative filtering model...\n');
fprintf('Lambda: %.1f, Max iterations: %d\n', LAMBDA, MAX_ITER);

% Optimization options
options = optimset('GradObj', 'on', 'MaxIter', MAX_ITER, 'Display', 'iter');

% Train model using optimized cost function
training_start = tic;
theta = fmincg(@(t)(cofiCostFunc(t, Ynorm, R, num_users, num_movies, ...
                    NUM_FEATURES, LAMBDA)), initial_parameters, options);
training_time = toc(training_start);

fprintf('Training completed in %.2f seconds\n', training_time);

% Unfold parameters
X = reshape(theta(1:num_movies*NUM_FEATURES), num_movies, NUM_FEATURES);
Theta = reshape(theta(num_movies*NUM_FEATURES+1:end), num_users, NUM_FEATURES);

%% =============== Part 6: Generate Recommendations ================
fprintf('\nGenerating recommendations...\n');

% Compute predictions efficiently
predictions = X * Theta';
my_predictions = predictions(:,1) + Ymean;

% Load movie list (using optimized loader)
try
    movieList = loadMovieList();
    
    % Sort and display top recommendations
    [sorted_predictions, ix] = sort(my_predictions, 'descend');
    
    fprintf('\n=== TOP 10 MOVIE RECOMMENDATIONS ===\n');
    for i = 1:min(10, length(ix))
        j = ix(i);
        if j <= length(movieList)
            fprintf('%2d. %.1f/5 - %s\n', i, my_predictions(j), movieList{j});
        end
    end
    
    % Show original ratings
    fprintf('\n=== YOUR ORIGINAL RATINGS ===\n');
    rated_movies = find(my_ratings > 0);
    for i = 1:length(rated_movies)
        movie_idx = rated_movies(i);
        if movie_idx <= length(movieList)
            fprintf('%.0f/5 - %s\n', my_ratings(movie_idx), movieList{movie_idx});
        end
    end
    
catch ME
    fprintf('Warning: Could not load movie list: %s\n', ME.message);
    fprintf('Showing numeric recommendations instead:\n');
    [~, ix] = sort(my_predictions, 'descend');
    for i = 1:10
        fprintf('Movie %d: %.1f/5\n', ix(i), my_predictions(ix(i)));
    end
end

%% =============== Part 7: Performance Analysis ================
if ENABLE_PROFILING
    total_time = toc;
    profile off;
    
    fprintf('\n=== PERFORMANCE SUMMARY ===\n');
    fprintf('Total execution time: %.2f seconds\n', total_time);
    fprintf('Training time: %.2f seconds (%.1f%%)\n', ...
        training_time, 100*training_time/total_time);
    fprintf('Data size: %d x %d matrix (%.1f MB)\n', ...
        num_movies, num_users, num_movies*num_users*8/1024/1024);
    fprintf('Parameters: %d (%.1f MB)\n', ...
        length(theta), length(theta)*8/1024/1024);
    
    % Display profiling results
    fprintf('\nDetailed profiling available via: profile viewer\n');
end

%% =============== Performance Tips ================
fprintf('\n=== OPTIMIZATION NOTES ===\n');
fprintf('1. Vectorized operations eliminate loops\n');
fprintf('2. Pre-computed matrices reduce redundant calculations\n');
fprintf('3. Memory-efficient operations handle large datasets\n');
fprintf('4. Batch I/O operations improve file access speed\n');
fprintf('5. Configurable parameters allow faster iteration\n');

fprintf('\n=== COMPLETED SUCCESSFULLY ===\n');