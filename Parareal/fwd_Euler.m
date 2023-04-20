function [t,y] = fwd_Euler(t0,tN,y0,h,f)

%--------------------------------------------------------------------------
% Forward Euler algorithm
%--------------------------------------------------------------------------    
% y'(t) = f(t,y) , t in [t0,tN]
% y(0) = y0
% 
% h = discretization step

    N = (tN - t0) / h;
    
    t = zeros(N+1,1);
    t(1) = t0;
    
    y = zeros(N+1,1);
    y(1) = y0;
    
    for k = 1:N
        dy = f(t(k),y(k));
        t(k+1) = t(k) + h;
        y(k+1) = y(k) + h*dy;
    end
   
end