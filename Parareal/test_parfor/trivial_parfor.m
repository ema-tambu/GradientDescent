% Parfor: first implementation attempt
clear all
close all
clc

% trivial use of parfor

w = [];
n = 30000000;
tic
for i = 1:n
    %w(i) = trivial_function();
    w(i) = i;
end
toc
% Elapsed time is higher

v = [];
for k = 1:3
    tic
    parfor i = 1:n
        %v(i) = trivial_function();
        v(i) = w(i);
        %v(i+1) = 5 + v(i); % not allowed
    end
    toc
end
% Elapsed time is lower