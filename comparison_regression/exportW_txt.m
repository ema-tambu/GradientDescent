clear all; close all; clc;
load Copy_2_of_initial_weights_GD.mat

m = W{3};

size(m)

v = []
for i=1:size(m,1)
    v = [v, m(i,:)'];
end
v = v';
size(v)
dlmwrite('W3r.txt', v, 'delimiter', '\t', 'precision', '%.16f');


%% 
clear all; close all; clc;
load Copy_2_of_initial_weights_GD.mat

m = b{3}
m = m';
size(m)
dlmwrite('b3r.txt', m, 'delimiter', '\t', 'precision', '%.16f');