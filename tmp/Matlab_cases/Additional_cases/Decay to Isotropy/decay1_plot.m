%This m-file solves various turbulence flow models for homogeneous sheared
%turbulence. The turbulence is initially isotropic.
%by P.E. Hamlington, February 7, 2009.

clc;
clear all;
clf;

load decay1.txt
load decay1_lrr.txt

%Plot location and dimension parameters------------------------------------
%Figure dimensions and location on screen in inches
fx=8;
fy=3;
fw=5;
fh=4;

%Plot dimensions
x=0.14;
y=0.13;
w=0.8;
h=0.81;

%Font sizes
pfont=12; %tick mark font size
legfont=12; %legend font size
xfont=13; %x label font size
yfont=13; %y label font size
yfontf=17;
tfont=12;

lwid=2; %line width
msize=7;

yx=8;
%--------------------------------------------------------------------------
%Load in validation data

global  ce1 ce2 cmu c

%ODE Solution parameters
dt=0.001; %time step
tmax=45; %maximum time

cmu=0.09;
ce1=1.44;
ce2=1.92;

a110=0.3;
a220=-0.02;
a330=-0.28;

%--------------------------------------------------------------------------
%Solve turbulence models
[Tlrr,Ylrr] = ode113(@lrr,[0:dt:tmax],[a110,a220,a330,0,0,0,1,1]); %LRR RST model

a11_lrr=Ylrr(:,1);
a22_lrr=Ylrr(:,2);
a33_lrr=Ylrr(:,3);

tau_lrr=(1/(2*(ce2-1)))*log(1+(ce2-1)*Tlrr);
%--------------------------------------------------------------------------


tau=[0:0.01:tmax];
tau_nke=(1/(2*(ce2-1)))*log(1+(ce2-1)*tau);

c=0.26;
a11_nke=a110*exp(-tau./(c*(ce2-1)*tau+c));
a22_nke=a220*exp(-tau./(c*(ce2-1)*tau+c));
a33_nke=a330*exp(-tau./(c*(ce2-1)*tau+c));

c=2;
a11_nke_l=a110*exp(-tau./(c*(ce2-1)*tau+c));
a22_nke_l=a220*exp(-tau./(c*(ce2-1)*tau+c));
a33_nke_l=a330*exp(-tau./(c*(ce2-1)*tau+c));

ifig=1;
figure(ifig)
%Axis limits
xlow=0;  
xhigh=1;
ylow=-0.4;
yhigh=0.4;
set(gcf,'Units','inches','Position',[fx fy+(fh+1)*(1-ifig) fw fh],'Color','w')
subplot('Position',[x,y,w,h]);plot(tau_lrr,a11_lrr,'--m','LineWidth',lwid)
hold on;
%subplot('Position',[x,y,w,h]);plot(tau_nke,a11_nke,'-b','LineWidth',lwid)
%subplot('Position',[x,y,w,h]);plot(tau_nke,a22_nke,'-b','LineWidth',lwid)
%subplot('Position',[x,y,w,h]);plot(tau_nke,a33_nke,'-b','LineWidth',lwid)
subplot('Position',[x,y,w,h]);plot(tau_nke,a11_nke_l,'-b','LineWidth',2)
subplot('Position',[x,y,w,h]);plot(tau_nke,a22_nke_l,'-b','LineWidth',2)
subplot('Position',[x,y,w,h]);plot(tau_nke,a33_nke_l,'-b','LineWidth',2)
subplot('Position',[x,y,w,h]);plot(tau_lrr,a22_lrr,'--m','LineWidth',lwid)
subplot('Position',[x,y,w,h]);plot(tau_lrr,a33_lrr,'--m','LineWidth',lwid)
subplot('Position',[x,y,w,h]);plot(decay1(:,1),2*decay1(:,2),'ok','MarkerSize',msize)
%subplot('Position',[x,y,w,h]);plot(decay1_lrr(:,1),2*decay1_lrr(:,2),'dk','MarkerSize',msize)
hold off;
set(gca,'FontSize',pfont,'FontName','Times')
text(0.08,0.21,'$a_{11}$','Interpreter','Latex','FontSize',tfont) 
text(0.08,0.02,'$a_{22}$','Interpreter','Latex','FontSize',tfont) 
text(0.08,-0.28,'$a_{33}$','Interpreter','Latex','FontSize',tfont) 
%set(gca,'YTickLabel',{'0.5','1.0','1.5','2.0','2.5','3.0','3.5','4.0'})
ylabel('$a_{ij}$','Interpreter','latex','Rotation',0,'FontSize',xfont,...
    'Position',[xlow-(xhigh-xlow)/yx,(ylow+yhigh)/2])
xlabel('$\tau$','Interpreter','latex','Rotation',0,'FontSize',xfont)
axis([xlow,xhigh,ylow,yhigh])

%This is the ghetto legend generator---------------------------------------
%The legend is defined as an array with coordinates i x j
%Line styles and labels must be changed manually.
%All dimensions are in axis coordinates.

xs=0.6; %LEFT edge of legend
ys=-0.25; %TOP edge of legend
len=(xhigh-xlow)/8; %line length
xlsp=(xhigh-xlow)/40; %horizontal space between line and label
ysp=(yhigh-ylow)/14; %vertical space between legend entries
xsp=(xhigh-xlow)/3.6; %horizontal space between legend entries

i=1; %x location
j=1; %y location
line([xs+(i-1)*xsp,xs+(i-1)*xsp+len],[(ys-(j-1)*ysp),(ys-(j-1)*ysp)],...
    'LineStyle','-','Color','b','LineWidth',lwid)
text(xs+(i-1)*xsp+len+xlsp,(ys-(j-1)*ysp),'NKE','Interpreter',...
    'latex','FontSize',legfont)

i=1; %x location
j=2; %y location
line([xs+(i-1)*xsp,xs+(i-1)*xsp+len],[(ys-(j-1)*ysp),(ys-(j-1)*ysp)],...
    'LineStyle','--','Color','m','LineWidth',lwid)
text(xs+(i-1)*xsp+len+xlsp,(ys-(j-1)*ysp),'LRR','Interpreter',...
    'latex','FontSize',legfont)

i=1; %x location
j=3; %y location
xmark=(xs+(i-1)*xsp+len-(xs+(i-1)*xsp))/2;
line([xs+(i-1)*xsp+xmark,xs+(i-1)*xsp+xmark],[(ys-(j-1)*ysp),(ys-(j-1)*ysp)],...
    'LineStyle','o','Color','k','MarkerSize',msize)
text(xs+(i-1)*xsp+len+xlsp,(ys-(j-1)*ysp),'Experiment','Interpreter',...
    'latex','FontSize',legfont)
%--------------------------------------------------------------------------