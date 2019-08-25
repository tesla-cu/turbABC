function dy = nke_4(t,y)

global k0e0 a1 a2 a3 a4 S0 Lt t1 t2 t3 t4 t5 t6 cmu c ce1 ce2

dy = zeros(2,1);  

k=y(1); %kinetic energy
e=y(2); %dissipation rate
ke=k/e; %k/e
lam=c*ke; %memory time scale
S11=0; %applied strain
S11e=a1*(lam*exp(-(t-t1)/lam)-(lam+t1-t2)*exp(-(t-t2)/lam))...
    +a2*(lam*exp(-(t-t3)/lam)-(lam-t2+t3)*exp(-(t-t2)/lam));     %effective strain
a11=-2*cmu*ke*S11e; %anisotropy 

%These are the governing equations
%dk/dt=-ka_{ij}S_{ij}-e
%de/dt=(-kC_{e1}a_{ij}S_{ij}-C_{e2}e)*e/k
dy(1)=-k*(2*a11*S11)-e; %dk/dt
dy(2)=(-k*ce1*(2*a11*S11)-ce2*e)/ke; %de/dt