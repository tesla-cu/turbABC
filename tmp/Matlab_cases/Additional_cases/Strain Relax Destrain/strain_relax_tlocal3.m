%This m-file solves various turbulence flow models for homogeneous turbulence
%that is strain, relaxed, and then destrained.
%by P.E. Hamlington, November 14, 2007.

clc;
clear all;

load ske.txt %experimental straining from Chen et al.
load b11.txt %experimental anisotropy 

global t2 cmu c ce1 ce2 a b B0 B1 B2 B3 B4

k0e0=0.0092/0.0035; %initial value of k/e

%Model parameters
cmu=0.05; %eddy viscosity coefficient
ce1=1.44; %dissipation equation production coefficient
ce2=1.92; %dissipation equation disspation coefficient

c=0.26; %NKE memory time scale coefficient

%ODE Solution parameters
dt=0.001; %time step

%Plot location and dimension parameters------------------------------------
%Figure dimensions and location on screen in inches
fx=14;
fy=0;
fw=7;
fh=9;

%Plot dimensions
x=0.08;
y=0.05;
w=0.4;
h=0.265;

yoff=0.33;
xoff=0.5;

%Font sizes
pfont=12; %tick mark font size
legfont=12; %legend font size
xfont=13; %x label font size
yfont=13; %y label font size
yfontf=16;
tfont=12;

lwid=2; %line width
msize=7;

yx=7;
yxs=9;

tx=0;
tys=10.5;
tya=0.14;

txb=2.1;
tyb=9.3;
%--------------------------------------------------------------------------

%Set up straining
a=9.4;

%88888888888888888888888888888888888888888888888888888888888888888888888888
t0=0;
t2=0.41;
t4=3.5;
terf=[t0:dt:t4];
xlow=t0;
xhigh=t4;

b=0.008;
Sg=a*exp(-(terf-t2).^2./b);

%Solve SKE ODEs
[Tske,Yske] = ode45(@ske_g,[t0:dt:t4],[1,1]);
[Tnkeg,Ynkeg] = ode113(@nke_g,[t0:dt:t4],[1,1]);


%Find SKE anisotropy time series
ke=Yske(:,1)./Yske(:,2);
S11=a*exp(-(Tske-t2).^2./b); %applied strain
aske=-2*cmu*ke.*S11; %anisotropy 

ke=Ynkeg(:,1)./Ynkeg(:,2);
lam=c*ke; %memory time scale
S11=a*exp(-(Tnkeg-t2).^2./b); %applied strain
S11e=a*sqrt(pi*b)./(2*lam).*exp((b-4*lam.*Tnkeg+4*lam.*t2)./(4*lam.^2)).*...
    (erf((b+2*lam.*t2)./(2*lam.*sqrt(b)))-...
     erf((b+2*lam.*(t2-Tnkeg))./(2*lam.*sqrt(b))));
ankeg=-2*cmu*ke.*S11e; %anisotropy 
 
G=0.1;
B0=gammajn(1,0,G); 
B1=gammajn(1,1,G);
[Tnkeg1,Ynkeg1] = ode45(@nke_g1,[t0:dt:t4],[1,1]);
ke=Ynkeg1(:,1)./Ynkeg1(:,2);
lam=c*ke; %memory time scale
S11=a*exp(-(Tnkeg1-t2).^2./b); %applied strain
S11e=S11-(B1/B0)*lam.*(-2/b).*a.*(Tnkeg1-t2).*exp(-(Tnkeg1-t2).^2./b); %effective strain
ankeg1=-2*cmu*ke.*S11e; %anisotropy 

G=0.2;
B0=gammajn(1,0,G); 
B1=gammajn(1,1,G);
B2=gammajn(1,2,G); 
[Tnkeg2,Ynkeg2] = ode45(@nke_g2,[t0:dt:t4],[1,1]);
ke=Ynkeg2(:,1)./Ynkeg2(:,2);
lam=c*ke; %memory time scale
S11=a*exp(-(Tnkeg2-t2).^2./b); %applied strain
S11e=S11-(B1/B0)*lam.*(-2/b).*a.*(Tnkeg2-t2).*exp(-(Tnkeg2-t2).^2./b)...
        +(B2/B0)*lam.^2.*((4*a/b^2)*(Tnkeg2-t2).^2-2*a/b).*exp(-(Tnkeg2-t2).^2./b); %effective strain
ankeg2=-2*cmu*ke.*S11e; %anisotropy 

G=0.3;
B0=gammajn(1,0,G); 
B1=gammajn(1,1,G);
B2=gammajn(1,2,G); 
B3=gammajn(1,3,G); 
[Tnkeg3,Ynkeg3] = ode45(@nke_g3,[t0:dt:t4],[1,1]);
ke=Ynkeg3(:,1)./Ynkeg3(:,2);
lam=c*ke; %memory time scale
S11=a*exp(-(Tnkeg3-t2).^2./b); %applied strain
S11e=S11-(B1/B0)*lam.*(-2/b).*a.*(Tnkeg3-t2).*exp(-(Tnkeg3-t2).^2./b)...
        +(B2/B0)*lam.^2.*((4*a/b^2)*(Tnkeg3-t2).^2-2*a/b).*exp(-(Tnkeg3-t2).^2./b)...
        -(B3/B0)*lam.^3.*(12*a*(Tnkeg3-t2)./b^2-8*a*(Tnkeg3-t2).^3./b^3).*exp(-(Tnkeg3-t2).^2./b); %effective strain
ankeg3=-2*cmu*ke.*S11e; %anisotropy 

ifig=1;
figure(ifig)
set(gcf,'Units','inches','Position',[fx fy+(fh+1)*(1-ifig) fw fh],'Color','w')
clf;

ylow=-1;
yhigh=10;
exend=35;
subplot('Position',[x,y+2*yoff,w,h]);plot(ske(1:exend,1),ske(1:exend,2),'ok','MarkerSize',7)
hold on;
subplot('Position',[x,y+2*yoff,w,h]);plot(terf,Sg,'-k','LineWidth',lwid)
hold off;
set(gca,'XTick',[0,1,2,3])
text(tx,tys,'(a)','Interpreter','Latex','FontSize',tfont)
text(txb,tyb,'$\sigma_s \epsilon_0/k_0 = 0.09$','Interpreter','Latex','FontSize',tfont)
set(gca,'FontSize',pfont,'FontName','Times')
ylabel('$\frac{\overline{S}_{11}k_0}{\epsilon_0}$','Interpreter','latex','Rotation',0,...
    'FontSize',yfontf,'Position',[xlow-(xhigh-xlow)/yxs,(ylow+yhigh)/2])
xlabel('$t\epsilon_0/k_0$','Interpreter','Latex','FontSize',xfont)
axis([xlow,xhigh,ylow,yhigh])

%This is the ghetto legend generator---------------------------------------
%The legend is defined as an array with coordinates i x j
%Line styles and labels must be changed manually.
%All dimensions are in axis coordinates.

xs=1.3; %LEFT edge of legend
ys=3; %TOP edge of legend
len=(xhigh-xlow)/8; %line length
xlsp=(xhigh-xlow)/40; %horizontal space between line and label
ysp=(yhigh-ylow)/14; %vertical space between legend entries
xsp=(xhigh-xlow)/4; %horizontal space between legend entries

i=1; %x location
j=1; %y location
xmark=(xs+(i-1)*xsp+len-(xs+(i-1)*xsp))/2;
line([xs+(i-1)*xsp+xmark,xs+(i-1)*xsp+xmark],[(ys-(j-1)*ysp),(ys-(j-1)*ysp)],...
    'LineStyle','o','Color','k','MarkerSize',7)
text(xs+(i-1)*xsp+len+xlsp,(ys-(j-1)*ysp),'Experiment','Interpreter',...
    'latex','FontSize',legfont)

i=1; %x location
j=2; %y location
line([xs+(i-1)*xsp,xs+(i-1)*xsp+len],[(ys-(j-1)*ysp),(ys-(j-1)*ysp)],...
    'LineStyle','-','Color','k','LineWidth',lwid)
text(xs+(i-1)*xsp+len+xlsp,(ys-(j-1)*ysp),'Gaussian strain','Interpreter',...
    'latex','FontSize',legfont)
%--------------------------------------------------------------------------

sc=2/3;
xhigh=1;
ylow=-1;
yhigh=0.1;
exend=35;
subplot('Position',[x+xoff,y+2*yoff,w,h]);plot(Tske,aske,'-.r','LineWidth',lwid)
hold on;
subplot('Position',[x+xoff,y+2*yoff,w,h]);plot(b11(1:exend,1),2*sc*b11(1:exend,2),'ok','MarkerSize',7)
subplot('Position',[x+xoff,y+2*yoff,w,h]);plot(Tnkeg,ankeg,'-k','LineWidth',lwid)
subplot('Position',[x+xoff,y+2*yoff,w,h]);plot(Tnkeg1,ankeg1,'-b','LineWidth',1)
subplot('Position',[x+xoff,y+2*yoff,w,h]);plot(Tnkeg2,ankeg2,'--b','LineWidth',lwid)
subplot('Position',[x+xoff,y+2*yoff,w,h]);plot(Tnkeg3,ankeg3,'-.b','LineWidth',lwid)
hold off;
text(tx,0.15,'(b)','Interpreter','Latex','FontSize',tfont)
%line([0.7,0.7],[ylow,yhigh])
set(gca,'FontSize',pfont,'FontName','Times')
%set(gca,'YTick',[-1,-0.5,0,0.5,1])
ylabel('$a_{11}$','Interpreter','latex','Rotation',0,...
    'FontSize',yfont,'Position',[xlow-(xhigh-xlow)/yx,(ylow+yhigh)/2])
xlabel('$t\epsilon_0/k_0$','Interpreter','Latex','FontSize',xfont)
axis([xlow,xhigh,ylow,yhigh])

%This is the ghetto legend generator---------------------------------------
%The legend is defined as an array with coordinates i x j
%Line styles and labels must be changed manually.
%All dimensions are in axis coordinates.

xs=0.53; %LEFT edge of legend
ys=-0.53; %TOP edge of legend
len=(xhigh-xlow)/8; %line length
xlsp=(xhigh-xlow)/40; %horizontal space between line and label
ysp=(yhigh-ylow)/14; %vertical space between legend entries
xsp=(xhigh-xlow)/3.6; %horizontal space between legend entries

i=1; %x location
j=1; %y location
line([xs+(i-1)*xsp,xs+(i-1)*xsp+len],[(ys-(j-1)*ysp),(ys-(j-1)*ysp)],...
    'LineStyle','-.','Color','r','LineWidth',lwid)
text(xs+(i-1)*xsp+len+xlsp,(ys-(j-1)*ysp),'SKE','Interpreter',...
    'latex','FontSize',legfont)

i=1; %x location
j=2; %y location
line([xs+(i-1)*xsp,xs+(i-1)*xsp+len],[(ys-(j-1)*ysp),(ys-(j-1)*ysp)],...
    'LineStyle','-','Color','b','LineWidth',1)
text(xs+(i-1)*xsp+len+xlsp,(ys-(j-1)*ysp),'$(1,\,0.1)$','Interpreter',...
    'latex','FontSize',legfont)

i=1; %x location
j=3; %y location
line([xs+(i-1)*xsp,xs+(i-1)*xsp+len],[(ys-(j-1)*ysp),(ys-(j-1)*ysp)],...
    'LineStyle','--','Color','b','LineWidth',lwid)
text(xs+(i-1)*xsp+len+xlsp,(ys-(j-1)*ysp),'$(2,\,0.2)$','Interpreter',...
    'latex','FontSize',legfont)

i=1; %x location
j=4; %y location
line([xs+(i-1)*xsp,xs+(i-1)*xsp+len],[(ys-(j-1)*ysp),(ys-(j-1)*ysp)],...
    'LineStyle','-.','Color','b','LineWidth',lwid)
text(xs+(i-1)*xsp+len+xlsp,(ys-(j-1)*ysp),'$(3,\,0.3)$','Interpreter',...
    'latex','FontSize',legfont)

i=1; %x location
j=5; %y location
line([xs+(i-1)*xsp,xs+(i-1)*xsp+len],[(ys-(j-1)*ysp),(ys-(j-1)*ysp)],...
    'LineStyle','-','Color','k','LineWidth',lwid)
text(xs+(i-1)*xsp+len+xlsp,(ys-(j-1)*ysp),'NKE','Interpreter',...
    'latex','FontSize',legfont)

i=1; %x location
j=6; %y location
xmark=(xs+(i-1)*xsp+len-(xs+(i-1)*xsp))/2;
line([xs+(i-1)*xsp+xmark,xs+(i-1)*xsp+xmark],[(ys-(j-1)*ysp),(ys-(j-1)*ysp)],...
    'LineStyle','o','Color','k','MarkerSize',7)
text(xs+(i-1)*xsp+len+xlsp,(ys-(j-1)*ysp),'Experiment','Interpreter',...
    'latex','FontSize',legfont)
%--------------------------------------------------------------------------
%88888888888888888888888888888888888888888888888888888888888888888888888888


%88888888888888888888888888888888888888888888888888888888888888888888888888
t2=1.25;
b=0.3;
xhigh=3.5;
Sg=a*exp(-(terf-t2).^2./b);

%Solve SKE ODEs
[Tske,Yske] = ode113(@ske_g,[t0:dt:t4],[1,1]);
[Tnkeg,Ynkeg] = ode113(@nke_g,[t0:dt:t4],[1,1]);

%Find SKE anisotropy time series
ke=Yske(:,1)./Yske(:,2);
S11=a*exp(-(Tske-t2).^2./b); %applied strain
aske=-2*cmu*ke.*S11; %anisotropy 

ke=Ynkeg(:,1)./Ynkeg(:,2);
lam=c*ke; %memory time scale
S11=a*exp(-(Tnkeg-t2).^2./b); %applied strain
S11e=a*sqrt(pi*b)./(2*lam).*exp((b-4*lam.*Tnkeg+4*lam.*t2)./(4*lam.^2)).*...
    (erf((b+2*lam.*t2)./(2*lam.*sqrt(b)))-...
     erf((b+2*lam.*(t2-Tnkeg))./(2*lam.*sqrt(b))));
Ups1=abs(S11./S11e-1);
S11n=S11;
S11en=S11e;
ankeg=-2*cmu*ke.*S11e; %anisotropy 
 
G=0.7;
B0=gammajn(1,0,G); 
B1=gammajn(1,1,G);
[Tnkeg1,Ynkeg1] = ode113(@nke_g1,[0:dt:t4],[1,1]);
ke=Ynkeg1(:,1)./Ynkeg1(:,2);
lam=c*ke; %memory time scale
S11=a*exp(-(Tnkeg1-t2).^2./b); %applied strain
S11e=S11-(B1/B0)*lam.*(-2/b).*a.*(Tnkeg1-t2).*exp(-(Tnkeg1-t2).^2./b); %effective strain
ankeg1=-2*cmu*ke.*S11e; %anisotropy 

G=1.2;
B0=gammajn(1,0,G); 
B1=gammajn(1,1,G);
B2=gammajn(1,2,G); 
[Tnkeg2,Ynkeg2] = ode113(@nke_g2,[0:dt:t4],[1,1]);
ke=Ynkeg2(:,1)./Ynkeg2(:,2);
lam=c*ke; %memory time scale
S11=a*exp(-(Tnkeg2-t2).^2./b); %applied strain
S11e=S11-(B1/B0)*lam.*(-2/b).*a.*(Tnkeg2-t2).*exp(-(Tnkeg2-t2).^2./b)...
        +(B2/B0)*lam.^2.*((4*a/b^2)*(Tnkeg2-t2).^2-2*a/b).*exp(-(Tnkeg2-t2).^2./b); %effective strain
ankeg2=-2*cmu*ke.*S11e; %anisotropy 

G=1.7;
B0=gammajn(1,0,G); 
B1=gammajn(1,1,G);
B2=gammajn(1,2,G); 
B3=gammajn(1,3,G); 
[Tnkeg3,Ynkeg3] = ode113(@nke_g3,[0:dt:t4],[1,1]);
ke=Ynkeg3(:,1)./Ynkeg3(:,2);
lam=c*ke; %memory time scale
S11=a*exp(-(Tnkeg3-t2).^2./b); %applied strain
S11e=S11-(B1/B0)*lam.*(-2/b).*a.*(Tnkeg3-t2).*exp(-(Tnkeg3-t2).^2./b)...
        +(B2/B0)*lam.^2.*((4*a/b^2)*(Tnkeg3-t2).^2-2*a/b).*exp(-(Tnkeg3-t2).^2./b)...
        -(B3/B0)*lam.^3.*(12*a*(Tnkeg3-t2)./b^2-8*a*(Tnkeg3-t2).^3./b^3).*exp(-(Tnkeg3-t2).^2./b); %effective strain
ankeg3=-2*cmu*ke.*S11e; %anisotropy 

ylow=-1;
yhigh=10;
subplot('Position',[x,y+1*yoff,w,h]);plot(terf,Sg,'-k','LineWidth',lwid)
set(gca,'XTick',[0,1,2,3])
text(tx,tys,'(c)','Interpreter','Latex','FontSize',tfont)
text(txb,tyb,'$\sigma_s \epsilon_0/k_0 = 0.55$','Interpreter','Latex','FontSize',tfont)
set(gca,'FontSize',pfont,'FontName','Times')
ylabel('$\frac{\overline{S}_{11}k_0}{\epsilon_0}$','Interpreter','latex','Rotation',0,...
    'FontSize',yfontf,'Position',[xlow-(xhigh-xlow)/yxs,(ylow+yhigh)/2])
xlabel('$t\epsilon_0/k_0$','Interpreter','Latex','FontSize',xfont)
axis([xlow,xhigh,ylow,yhigh])

sc=2/3;

xhigh=2.5;
ylow=-0.8;
yhigh=0.1;
exend=35;

subplot('Position',[x+xoff,y+1*yoff,w,h]);plot(Tske,aske,'-.r','LineWidth',lwid)
hold on;
subplot('Position',[x+xoff,y+1*yoff,w,h]);plot(Tnkeg,ankeg,'-k','LineWidth',lwid)
subplot('Position',[x+xoff,y+1*yoff,w,h]);plot(Tnkeg1,ankeg1,'-b','LineWidth',1)
subplot('Position',[x+xoff,y+1*yoff,w,h]);plot(Tnkeg2,ankeg2,'--b','LineWidth',lwid)
subplot('Position',[x+xoff,y+1*yoff,w,h]);plot(Tnkeg3,ankeg3,'-.b','LineWidth',lwid)
hold off;
text(tx,tya,'(d)','Interpreter','Latex','FontSize',tfont)
%line([0.7,0.7],[ylow,yhigh])
set(gca,'FontSize',pfont,'FontName','Times')
set(gca,'YTick',[-0.8,-0.6,-0.4,-0.2,0])
ylabel('$a_{11}$','Interpreter','latex','Rotation',0,...
    'FontSize',yfont,'Position',[xlow-(xhigh-xlow)/yx,(ylow+yhigh)/2])
xlabel('$t\epsilon_0/k_0$','Interpreter','Latex','FontSize',xfont)
axis([xlow,xhigh,ylow,yhigh])

%This is the ghetto legend generator---------------------------------------
%The legend is defined as an array with coordinates i x j
%Line styles and labels must be changed manually.
%All dimensions are in axis coordinates.

xs=1.6; %LEFT edge of legend
ys=-0.47; %TOP edge of legend
len=(xhigh-xlow)/8; %line length
xlsp=(xhigh-xlow)/40; %horizontal space between line and label
ysp=(yhigh-ylow)/13; %vertical space between legend entries
xsp=(xhigh-xlow)/3.6; %horizontal space between legend entries

i=1; %x location
j=1; %y location
line([xs+(i-1)*xsp,xs+(i-1)*xsp+len],[(ys-(j-1)*ysp),(ys-(j-1)*ysp)],...
    'LineStyle','-.','Color','r','LineWidth',lwid)
text(xs+(i-1)*xsp+len+xlsp,(ys-(j-1)*ysp),'SKE','Interpreter',...
    'latex','FontSize',legfont)

i=1; %x location
j=2; %y location
line([xs+(i-1)*xsp,xs+(i-1)*xsp+len],[(ys-(j-1)*ysp),(ys-(j-1)*ysp)],...
    'LineStyle','-','Color','b','LineWidth',1)
text(xs+(i-1)*xsp+len+xlsp,(ys-(j-1)*ysp),'$(1,\,0.7)$','Interpreter',...
    'latex','FontSize',legfont)

i=1; %x location
j=3; %y location
line([xs+(i-1)*xsp,xs+(i-1)*xsp+len],[(ys-(j-1)*ysp),(ys-(j-1)*ysp)],...
    'LineStyle','--','Color','b','LineWidth',lwid)
text(xs+(i-1)*xsp+len+xlsp,(ys-(j-1)*ysp),'$(2,\,1.2)$','Interpreter',...
    'latex','FontSize',legfont)

i=1; %x location
j=4; %y location
line([xs+(i-1)*xsp,xs+(i-1)*xsp+len],[(ys-(j-1)*ysp),(ys-(j-1)*ysp)],...
    'LineStyle','-.','Color','b','LineWidth',lwid)
text(xs+(i-1)*xsp+len+xlsp,(ys-(j-1)*ysp),'$(3,\,1.7)$','Interpreter',...
    'latex','FontSize',legfont)

i=1; %x location
j=5; %y location
line([xs+(i-1)*xsp,xs+(i-1)*xsp+len],[(ys-(j-1)*ysp),(ys-(j-1)*ysp)],...
    'LineStyle','-','Color','k','LineWidth',lwid)
text(xs+(i-1)*xsp+len+xlsp,(ys-(j-1)*ysp),'NKE','Interpreter',...
    'latex','FontSize',legfont)
%--------------------------------------------------------------------------
%88888888888888888888888888888888888888888888888888888888888888888888888888


%88888888888888888888888888888888888888888888888888888888888888888888888888
b=0.6;
t2=1.8;
xhigh=3.5;
Sg=a*exp(-(terf-t2).^2./b);

%Solve SKE ODEs
[Tske,Yske] = ode113(@ske_g,[t0:dt:t4],[1,1]);
[Tnkeg,Ynkeg] = ode113(@nke_g,[t0:dt:t4],[1,1]);

%Find SKE anisotropy time series
ke=Yske(:,1)./Yske(:,2);
S11=a*exp(-(Tske-t2).^2./b); %applied strain
aske=-2*cmu*ke.*S11; %anisotropy 

ke=Ynkeg(:,1)./Ynkeg(:,2);
lam=c*ke; %memory time scale
S11=a*exp(-(Tnkeg-t2).^2./b); %applied strain
S11e=a*sqrt(pi*b)./(2*lam).*exp((b-4*lam.*Tnkeg+4*lam.*t2)./(4*lam.^2)).*...
    (erf((b+2*lam.*t2)./(2*lam.*sqrt(b)))-...
     erf((b+2*lam.*(t2-Tnkeg))./(2*lam.*sqrt(b))));
Ups1=abs(S11./S11e-1);
S11n=S11;
S11en=S11e;
ankeg=-2*cmu*ke.*S11e; %anisotropy 
 
G=1.1;
B0=gammajn(1,0,G); 
B1=gammajn(1,1,G);
[Tnkeg1,Ynkeg1] = ode113(@nke_g1,[0:dt:t4],[1,1]);
ke=Ynkeg1(:,1)./Ynkeg1(:,2);
lam=c*ke; %memory time scale
S11=a*exp(-(Tnkeg1-t2).^2./b); %applied strain
S11e=S11-(B1/B0)*lam.*(-2/b).*a.*(Tnkeg1-t2).*exp(-(Tnkeg1-t2).^2./b); %effective strain
ankeg1=-2*cmu*ke.*S11e; %anisotropy 

G=1.7;
B0=gammajn(1,0,G); 
B1=gammajn(1,1,G);
B2=gammajn(1,2,G); 
[Tnkeg2,Ynkeg2] = ode113(@nke_g2,[0:dt:t4],[1,1]);
ke=Ynkeg2(:,1)./Ynkeg2(:,2);
lam=c*ke; %memory time scale
S11=a*exp(-(Tnkeg2-t2).^2./b); %applied strain
S11e=S11-(B1/B0)*lam.*(-2/b).*a.*(Tnkeg2-t2).*exp(-(Tnkeg2-t2).^2./b)...
        +(B2/B0)*lam.^2.*((4*a/b^2)*(Tnkeg2-t2).^2-2*a/b).*exp(-(Tnkeg2-t2).^2./b); %effective strain
ankeg2=-2*cmu*ke.*S11e; %anisotropy 

G=2.3;
B0=gammajn(1,0,G); 
B1=gammajn(1,1,G);
B2=gammajn(1,2,G); 
B3=gammajn(1,3,G); 
[Tnkeg3,Ynkeg3] = ode113(@nke_g3,[0:dt:t4],[1,1]);
ke=Ynkeg3(:,1)./Ynkeg3(:,2);
lam=c*ke; %memory time scale
S11=a*exp(-(Tnkeg3-t2).^2./b); %applied strain
S11e=S11-(B1/B0)*lam.*(-2/b).*a.*(Tnkeg3-t2).*exp(-(Tnkeg3-t2).^2./b)...
        +(B2/B0)*lam.^2.*((4*a/b^2)*(Tnkeg3-t2).^2-2*a/b).*exp(-(Tnkeg3-t2).^2./b)...
        -(B3/B0)*lam.^3.*(12*a*(Tnkeg3-t2)./b^2-8*a*(Tnkeg3-t2).^3./b^3).*exp(-(Tnkeg3-t2).^2./b); %effective strain
ankeg3=-2*cmu*ke.*S11e; %anisotropy 

ylow=-1;
yhigh=10;
subplot('Position',[x,y+0*yoff,w,h]);plot(terf,Sg,'-k','LineWidth',lwid)
text(tx,tys,'(e)','Interpreter','Latex','FontSize',tfont)
text(txb,tyb,'$\sigma_s \epsilon_0/k_0 = 0.77$','Interpreter','Latex','FontSize',tfont)
set(gca,'XTick',[0,1,2,3])
set(gca,'FontSize',pfont,'FontName','Times')
ylabel('$\frac{\overline{S}_{11}k_0}{\epsilon_0}$','Interpreter','latex','Rotation',0,...
    'FontSize',yfontf,'Position',[xlow-(xhigh-xlow)/yxs,(ylow+yhigh)/2])
xlabel('$t\epsilon_0/k_0$','Interpreter','latex','FontSize',xfont)
axis([xlow,xhigh,ylow,yhigh])

sc=2/3;

ylow=-0.8;
yhigh=0.1;

subplot('Position',[x+xoff,y+0*yoff,w,h]);plot(Tske,aske,'-.r','LineWidth',lwid)
hold on;
subplot('Position',[x+xoff,y+0*yoff,w,h]);plot(Tnkeg,ankeg,'-k','LineWidth',lwid)
subplot('Position',[x+xoff,y+0*yoff,w,h]);plot(Tnkeg1,ankeg1,'-b','LineWidth',1)
subplot('Position',[x+xoff,y+0*yoff,w,h]);plot(Tnkeg2,ankeg2,'--b','LineWidth',lwid)
subplot('Position',[x+xoff,y+0*yoff,w,h]);plot(Tnkeg3,ankeg3,'-.b','LineWidth',lwid)
hold off;
text(tx,tya,'(f)','Interpreter','Latex','FontSize',tfont)
set(gca,'FontSize',pfont,'FontName','Times')
set(gca,'YTick',[-0.8,-0.6,-0.4,-0.2,0])
set(gca,'XTick',[0,1,2,3])
ylabel('$a_{11}$','Interpreter','latex','Rotation',0,...
    'FontSize',yfont,'Position',[xlow-(xhigh-xlow)/yx,(ylow+yhigh)/2])
xlabel('$t\epsilon_0/k_0$','Interpreter','Latex','FontSize',xfont)
axis([xlow,xhigh,ylow,yhigh])
annotation('rectangle',[0.75 0 0.01 0.004],'Color','w','FaceColor','w') 
annotation('rectangle',[0.55 0.01 0.022 0.027],'Color','w','FaceColor','w')

%This is the ghetto legend generator---------------------------------------
%The legend is defined as an array with coordinates i x j
%Line styles and labels must be changed manually.
%All dimensions are in axis coordinates.

xs=2.24; %LEFT edge of legend
ys=-0.47; %TOP edge of legend
len=(xhigh-xlow)/8; %line length
xlsp=(xhigh-xlow)/40; %horizontal space between line and label
ysp=(yhigh-ylow)/13; %vertical space between legend entries
xsp=(xhigh-xlow)/3.6; %horizontal space between legend entries

i=1; %x location
j=1; %y location
line([xs+(i-1)*xsp,xs+(i-1)*xsp+len],[(ys-(j-1)*ysp),(ys-(j-1)*ysp)],...
    'LineStyle','-.','Color','r','LineWidth',lwid)
text(xs+(i-1)*xsp+len+xlsp,(ys-(j-1)*ysp),'SKE','Interpreter',...
    'latex','FontSize',legfont)

i=1; %x location
j=2; %y location
line([xs+(i-1)*xsp,xs+(i-1)*xsp+len],[(ys-(j-1)*ysp),(ys-(j-1)*ysp)],...
    'LineStyle','-','Color','b','LineWidth',1)
text(xs+(i-1)*xsp+len+xlsp,(ys-(j-1)*ysp),'$(1,\,1.1)$','Interpreter',...
    'latex','FontSize',legfont)

i=1; %x location
j=3; %y location
line([xs+(i-1)*xsp,xs+(i-1)*xsp+len],[(ys-(j-1)*ysp),(ys-(j-1)*ysp)],...
    'LineStyle','--','Color','b','LineWidth',lwid)
text(xs+(i-1)*xsp+len+xlsp,(ys-(j-1)*ysp),'$(2,\,1.7)$','Interpreter',...
    'latex','FontSize',legfont)

i=1; %x location
j=4; %y location
line([xs+(i-1)*xsp,xs+(i-1)*xsp+len],[(ys-(j-1)*ysp),(ys-(j-1)*ysp)],...
    'LineStyle','-.','Color','b','LineWidth',lwid)
text(xs+(i-1)*xsp+len+xlsp,(ys-(j-1)*ysp),'$(3,\,2.3)$','Interpreter',...
    'latex','FontSize',legfont)

i=1; %x location
j=5; %y location
line([xs+(i-1)*xsp,xs+(i-1)*xsp+len],[(ys-(j-1)*ysp),(ys-(j-1)*ysp)],...
    'LineStyle','-','Color','k','LineWidth',lwid)
text(xs+(i-1)*xsp+len+xlsp,(ys-(j-1)*ysp),'NKE','Interpreter',...
    'latex','FontSize',legfont)
%--------------------------------------------------------------------------
%88888888888888888888888888888888888888888888888888888888888888888888888888


figure(3)
plot(Tnkeg,Ups1,'-k','LineWidth',2)
xlabel('t\epsilon_0/k_0')
ylabel('\Upsilon')
axis([0,3.5,0,2])