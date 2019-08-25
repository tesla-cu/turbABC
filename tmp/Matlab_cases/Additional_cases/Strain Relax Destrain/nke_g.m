function dy = nke_g(t,y)

global t2 cmu c ce1 ce2 a b

dy = zeros(2,1);  

k=y(1); %kinetic energy
e=y(2); %dissipation rate
ke=k/e; %k/e
lam=c*ke; %memory time scale
S11=a*exp(-(t-t2)^2/b); %applied strain
S11e=a*sqrt(pi*b)/(2*lam)*exp((b-4*lam*t+4*lam*t2)/(4*lam^2))*...
    (erf((b+2*lam*t2)/(2*lam*sqrt(b)))-...
     erf((b+2*lam*(t2-t))/(2*lam*sqrt(b)))); %effective strain
a11=-2*cmu*ke*S11e; %anisotropy 

%These are the governing equations
%dk/dt=-ka_{ij}S_{ij}-e
%de/dt=(-kC_{e1}a_{ij}S_{ij}-C_{e2}e)*e/k
dy(1)=-k*(2*a11*S11)-e; %dk/dt
dy(2)=(-k*ce1*(2*a11*S11)-ce2*e)/ke; %de/dt