function plot_solution(time_points, parareal_solution, time_reference, reference_solution)
dim = size(parareal_solution, 2);
for i=1:dim
    plot(time_points, parareal_solution(:,i), 'o-', 'MarkerSize', 10, 'DisplayName', 'Parareal Solution');
    hold on;
    plot(time_reference, reference_solution(:,i), '.--', 'DisplayName', 'Reference Solution');
end
hold off;
xlabel('Time');
ylabel('y(t)');
legend();
grid on;
end