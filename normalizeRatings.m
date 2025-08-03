function [Ynorm, Ymean] = normalizeRatings(Y, R)
%NORMALIZERATINGS Preprocess data by subtracting mean rating for every 
%movie (every row) - VECTORIZED VERSION
%   [Ynorm, Ymean] = NORMALIZERATINGS(Y, R) normalized Y so that each movie
%   has a rating of 0 on average, and returns the mean rating in Ymean.
%   
%   OPTIMIZATIONS:
%   - Vectorized operations (no for loops)
%   - Memory efficient computation
%   - Handles zero-rated movies gracefully

[m, n] = size(Y);

% Vectorized computation of means
% Count non-zero ratings per movie
num_ratings = sum(R, 2);

% Handle movies with no ratings (avoid division by zero)
num_ratings(num_ratings == 0) = 1;

% Compute sum of ratings for each movie (only rated ones)
rating_sums = sum(Y .* R, 2);

% Compute mean ratings
Ymean = rating_sums ./ num_ratings;

% Handle movies with no ratings - set mean to 0
Ymean(sum(R, 2) == 0) = 0;

% Vectorized normalization
% Broadcast Ymean across all users and apply only to rated movies
Ynorm = Y - R .* repmat(Ymean, 1, n);

end
