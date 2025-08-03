function movieList = loadMovieList()
%GETMOVIELIST reads the fixed movie list in movie.txt and returns a
%cell array of the words - OPTIMIZED VERSION
%   movieList = GETMOVIELIST() reads the fixed movie list in movie.txt 
%   and returns a cell array of the words in movieList.
%   
%   OPTIMIZATIONS:
%   - Batch file reading for better I/O performance
%   - Pre-allocated cell array
%   - More efficient string processing

% Pre-allocate cell array for better memory performance
n = 1682;  % Total number of movies 
movieList = cell(n, 1);

% Read entire file at once (more efficient than line-by-line)
fid = fopen('movie_ids.txt');
if fid == -1
    error('Could not open movie_ids.txt');
end

try
    % Read all content at once
    content = textscan(fid, '%d %[^\n]', n);
    fclose(fid);
    
    % Extract movie names directly
    movieList = strtrim(content{2});
catch ME
    fclose(fid);
    rethrow(ME);
end

end
