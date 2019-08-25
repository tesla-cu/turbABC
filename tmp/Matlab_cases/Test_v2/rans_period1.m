function dx=rans_period1(t,x)

global C1 C2 ce1 ce2;

S0=3.3;
beta=0.125;

S11=0.;
S22=0.;
S33=0.;
S12=(S0/2)*sin(beta*S0*t); %applied shear
S13=0.;
S23=0.;

dx = zeros(8,1);  

k=x(1);                                         % turbulence kinetic energy
e=x(2);                                         % dissipation rate
a11=x(3);                                       % anisotropy tensor
a22=x(4);                                       % anisotropy tensor
a33=x(5);                                       % anisotropy tensor
a12=x(6);
a13=x(7);
a23=x(8);

P=-k*(a11*S11+a22*S22+a33*S33+2*a12*S12+2*a13*S13+2*a23*S23);

alf1=P/e-1+C1;
alf2=C2-4/3;

% Governing equations
dx(1)=P-e;                                      % dk/dt
dx(2)=(ce1*P-ce2*e)*e/k;                        % de/dt
dx(3)=-alf1*e*a11/k+alf2*S11;                   % da11/dt
dx(4)=-alf1*e*a22/k+alf2*S22;                   % da22/dt
dx(5)=-alf1*e*a33/k+alf2*S33;                   % da33/dt
dx(6)=-alf1*e*a12/k+alf2*S12;                   % da12/dt
dx(7)=-alf1*e*a13/k+alf2*S13;                   % da13/dt
dx(8)=-alf1*e*a23/k+alf2*S23;                   % da23/dt

