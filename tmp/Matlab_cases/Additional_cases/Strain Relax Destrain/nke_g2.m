function dy = nke_g2(t,y)

global t2 cmu c ce1 ce2 a b B0 B1 B2 B3 B4

dy = zeros(2,1);  

k=y(1); %kinetic energy
e=y(2); %dissipation rate
ke=k/e; %k/e
lam=c*ke; %memory time scale
S11=a*exp(-(t-t2)^2/b); %applied strain
S11e=S11-(B1/B0)*lam*(-2/b)*a*(t-t2)*exp(-(t-t2)^2/b)...
        +(B2/B0)*lam^2*((4*a/b^2)*(t-t2)^2-2*a/b)*exp(-(t-t2)^2/b); %effective strain
a11=-2*cmu*ke*S11e; %anisotropy 

%These are the governing equations
%dk/dt=-ka_{ij}S_{ij}-e
%de/dt=(-kC_{e1}a_{ij}S_{ij}-C_{e2}e)*e/k
dy(1)=-k*(2*a11*S11)-e; %dk/dt
dy(2)=(-k*ce1*(2*a11*S11)-ce2*e)/ke; %de/dt