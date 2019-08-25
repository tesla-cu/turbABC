clc;
clear all;
clf;

%--------------------------------------------------------------------------
%Load in validation data
axi_exp_k=load('./data/axi_exp_k.txt');
axi_exp_b=load('./data/axi_exp_b.txt');
axi_con_k=load('./data/axi_con_k.txt');
axi_con_b=load('./data/axi_con_b.txt');
shear_k=load('./data/shear_k.txt');
plane_k=load('./data/plane_k.txt');
plane_b=load('./data/plane_b.txt');
period1_k=load('./data/period1_k.txt');
period2_k=load('./data/period2_k.txt');
period3_k=load('./data/period3_k.txt');
period4_k=load('./data/period4_k.txt');
period5_k=load('./data/period5_k.txt');

global  C1 C2 ce1 ce2;

N=1e2;

C1min=1.3;      C1max=1.7;
C2min=0.7;      C2max=0.8;
ce1min=1.65;    ce1max=2;
ce2min=1.85;    ce2max=2.1;

ce1N=random('uniform',ce1min,ce1max,N,1);
ce2N=random('uniform',ce2min,ce2max,N,1);
C1N=random('uniform',C1min,C1max,N,1);
C2N=random('uniform',C2min,C2max,N,1);

errk_axi_exp=zeros(N,1);
erra_axi_exp=zeros(N,1);
errk_axi_con=zeros(N,1);
erra_axi_con=zeros(N,1);
errk_shear=zeros(N,1);
errk_plane=zeros(N,1);
erra_plane=zeros(N,1);
errk_period1=zeros(N,1);
errk_period2=zeros(N,1);
errk_period3=zeros(N,1);
errk_period4=zeros(N,1);
errk_period5=zeros(N,1);

%--------------------------------------------------------------------------
for i=1:N
    i
    ce1=ce1N(i);
    ce2=ce2N(i);
    C1=C1N(i);
    C2=C2N(i);
    
    %Axisymmetric expansion
    S11=-(1/2.45);
    [Tnke,Ynke]=ode45(@rans_axi_exp,0:0.01:5,[1,1,0,0,0,0,0,0]);
    errk_axi_exp(i)=max(abs(interp1(abs(S11)*Tnke,Ynke(:,1),axi_exp_k(:,1))-squeeze(axi_exp_k(:,2))));
    erra_axi_exp(i)=max(abs(interp1(abs(S11)*Tnke,Ynke(:,3),axi_exp_b(:,1))-squeeze(2*axi_exp_b(:,2))));
    
    %Axisymmetric contraction
    S11=(1/0.179);
    [Tnke,Ynke]=ode45(@rans_axi_con,0:0.01:2/abs(S11),[1,1,0,0,0,0,0,0]);
    errk_axi_con(i)=max(abs(interp1(abs(S11)*Tnke,Ynke(:,1),axi_con_k(:,1))-squeeze(axi_con_k(:,2))));
    erra_axi_con(i)=max(abs(interp1(abs(S11)*Tnke,Ynke(:,3),axi_con_b(:,1))-squeeze(2*axi_con_b(:,2))));
    
    %Pure shear
    S12=(1/0.296)/2;
    [Tnke,Ynke]=ode45(@rans_shear,0:0.01:15,[1,1,0,0,0,0,0,0]);
    errk_shear(i)=max(abs(interp1(2*S12*Tnke,Ynke(:,1),shear_k(:,1))-squeeze(shear_k(:,2))));
    
    %Plane strain
    S11=1/2;
    [Tnke,Ynke]=ode45(@rans_plane,0:0.01:5,[1,1,0,0,0,0,0,0]);
    errk_plane(i)=max(abs(interp1(abs(S11)*Tnke,Ynke(:,1),plane_k(:,1))-squeeze(plane_k(:,2))));
    erra_plane(i)=max(abs(interp1(abs(S11)*Tnke,Ynke(:,3),plane_b(:,1))-squeeze(2*plane_b(:,2))));
    
    %Periodic shear (five different frequencies)
    S0=3.3;
    [Tnke,Ynke]=ode45(@rans_period1,0:0.01:16,[1,1,0,0,0,0,0,0]);
    errk_period1(i)=max(abs(interp1(S0*Tnke,Ynke(:,1),period1_k(:,1))-squeeze(period1_k(:,2))));
    [Tnke,Ynke]=ode45(@rans_period2,0:0.01:16,[1,1,0,0,0,0,0,0]);
    errk_period2(i)=max(abs(interp1(S0*Tnke,Ynke(:,1),period2_k(:,1))-squeeze(period2_k(:,2))));
    [Tnke,Ynke]=ode45(@rans_period3,0:0.01:16,[1,1,0,0,0,0,0,0]);
    errk_period3(i)=max(abs(interp1(S0*Tnke,Ynke(:,1),period3_k(:,1))-squeeze(period3_k(:,2))));
    [Tnke,Ynke]=ode45(@rans_period4,0:0.01:16,[1,1,0,0,0,0,0,0]);
    errk_period4(i)=max(abs(interp1(S0*Tnke,Ynke(:,1),period4_k(:,1))-squeeze(period4_k(:,2))));
    [Tnke,Ynke]=ode45(@rans_period5,0:0.01:16,[1,1,0,0,0,0,0,0]);
    errk_period5(i)=max(abs(interp1(S0*Tnke,Ynke(:,1),period5_k(:,1))-squeeze(period5_k(:,2))));
end

%%
clear iSave;
cutk=0.06; cutkp=20; nbins=30;
iSave=find(errk_axi_con<cutk  & errk_axi_exp<cutk  &...
           errk_plane<cutk    & errk_shear<cutk    & ...
           errk_period1<cutkp & errk_period2<cutkp & ...
           errk_period3<cutkp & errk_period4<cutkp & ...
           errk_period5<cutkp);       

[ce1SvCNT,ce1SvEDG]=histcounts(ce1N(iSave),nbins);
[ce1CNT,  ce1EDG]  =histcounts(ce1N,nbins);
[ce2SvCNT,ce2SvEDG]=histcounts(ce2N(iSave),nbins);
[ce2CNT,  ce2EDG]  =histcounts(ce2N,nbins);

[C1SvCNT,C1SvEDG]=histcounts(C1N(iSave),nbins);
[C1CNT,  C1EDG]  =histcounts(C1N,nbins);
[C2SvCNT,C2SvEDG]=histcounts(C2N(iSave),nbins);
[C2CNT,  C2EDG]  =histcounts(C2N,nbins);

[C1_C2_SvCNT,   C1_C2_SvEDGx,   C1_C2_SvEDGy]=histcounts2(C1N(iSave),C2N(iSave),[30,30]);
[C1_ce1_SvCNT,  C1_ce1_SvEDGx,  C1_ce1_SvEDGy]=histcounts2(C1N(iSave),ce1N(iSave),[30,30]);
[C1_ce2_SvCNT,  C1_ce2_SvEDGx,  C1_ce2_SvEDGy]=histcounts2(C1N(iSave),ce2N(iSave),[30,30]);
[C2_ce1_SvCNT,  C2_ce1_SvEDGx,  C2_ce1_SvEDGy]=histcounts2(C2N(iSave),ce1N(iSave),[30,30]);
[C2_ce2_SvCNT,  C2_ce2_SvEDGx,  C2_ce2_SvEDGy]=histcounts2(C2N(iSave),ce2N(iSave),[30,30]);
[ce1_ce2_SvCNT, ce1_ce2_SvEDGx, ce1_ce2_SvEDGy]=histcounts2(ce1N(iSave),ce2N(iSave),[30,30]);

[count edges mid loc] = histcn([C1N(iSave) C2N(iSave) ce1N(iSave) ce2N(iSave)]);

%%
C1min=1.35;     C1max=1.65;
C2min=0.7;      C2max=0.76;
ce1min=1.7;     ce1max=1.95;
ce2min=1.85;    ce2max=2.02;

pfnt=11;
fnt=12;
lwid=2;
tl=[0.02,0.02];

figure(1)
clf;
set(gcf,'Units','inches','Position',[10 12 8 7],'Color','w')

pd=[0.08,0.07,0.18,0.19];
of=[0.24,0.242];

i=1; j=4;
subplot('Position',[pd(1)+(i-1)*of(1),pd(2)+(j-1)*of(2),pd(3),pd(4)]);
xtmp=0.5*(C1SvEDG(1:end-1)+C1SvEDG(2:end));  ytmp=C1SvCNT;
plot(xtmp,ytmp./trapz(xtmp,ytmp),'LineWidth',lwid)
hold on;
xtmp=0.5*(C1EDG(1:end-1)+C1EDG(2:end));  ytmp=C1CNT;
plot(xtmp,ytmp./trapz(xtmp,ytmp),'LineWidth',lwid)
hold off;
set(gca,'FontName','Times','FontSize',pfnt,'Ticklength',tl)
set(gca,'xlim',[C1min,C1max])
ylabel('$C_1$','Interpreter','Latex','FontSize',fnt)

i=2; j=3;
subplot('Position',[pd(1)+(i-1)*of(1),pd(2)+(j-1)*of(2),pd(3),pd(4)]);
xtmp=0.5*(C2SvEDG(1:end-1)+C2SvEDG(2:end));  ytmp=C2SvCNT;
plot(xtmp,ytmp./trapz(xtmp,ytmp),'LineWidth',lwid)
hold on;
xtmp=0.5*(C2EDG(1:end-1)+C2EDG(2:end));  ytmp=C2CNT;
plot(xtmp,ytmp./trapz(xtmp,ytmp),'LineWidth',lwid)
hold off;
set(gca,'FontName','Times','FontSize',pfnt,'Ticklength',tl)
set(gca,'xlim',[C2min,C2max])

i=3; j=2;
subplot('Position',[pd(1)+(i-1)*of(1),pd(2)+(j-1)*of(2),pd(3),pd(4)]);
xtmp=0.5*(ce1SvEDG(1:end-1)+ce1SvEDG(2:end));  ytmp=ce1SvCNT;
plot(xtmp,ytmp./trapz(xtmp,ytmp),'LineWidth',lwid)
hold on;
xtmp=0.5*(ce1EDG(1:end-1)+ce1EDG(2:end));  ytmp=ce1CNT;
plot(xtmp,ytmp./trapz(xtmp,ytmp),'LineWidth',lwid)
hold off;
set(gca,'FontName','Times','FontSize',pfnt,'Ticklength',tl)
set(gca,'xlim',[ce1min,ce1max])

i=4; j=1;
subplot('Position',[pd(1)+(i-1)*of(1),pd(2)+(j-1)*of(2),pd(3),pd(4)]);
xtmp=0.5*(ce2SvEDG(1:end-1)+ce2SvEDG(2:end));  ytmp=ce2SvCNT;
plot(xtmp,ytmp./trapz(xtmp,ytmp),'LineWidth',lwid)
hold on;
xtmp=0.5*(ce2EDG(1:end-1)+ce2EDG(2:end));  ytmp=ce2CNT;
plot(xtmp,ytmp./trapz(xtmp,ytmp),'LineWidth',lwid)
hold off;
set(gca,'FontName','Times','FontSize',pfnt,'Ticklength',tl)
set(gca,'xlim',[ce2min,ce2max])
xlabel('$C_{\varepsilon2}$','Interpreter','Latex','FontSize',fnt)

i=1; j=3;
subplot('Position',[pd(1)+(i-1)*of(1),pd(2)+(j-1)*of(2),pd(3),pd(4)]);
xtmp=0.5*(C1_C2_SvEDGx(1:end-1)+C1_C2_SvEDGx(2:end));
ytmp=0.5*(C1_C2_SvEDGy(1:end-1)+C1_C2_SvEDGy(2:end));  
ztmp=transpose(C1_C2_SvCNT);
contourf(xtmp,ytmp,ztmp,'EdgeColor','none')
set(gca,'FontName','Times','FontSize',pfnt,'Ticklength',tl)
axis([C1min,C1max,C2min,C2max])
ylabel('$C_2$','Interpreter','Latex','FontSize',fnt)
i=2; j=4;
subplot('Position',[pd(1)+(i-1)*of(1),pd(2)+(j-1)*of(2),pd(3),pd(4)]);
contourf(ytmp,xtmp,transpose(ztmp),'EdgeColor','none')
set(gca,'FontName','Times','FontSize',pfnt,'Ticklength',tl)
axis([C2min,C2max,C1min,C1max])

i=1; j=2;
subplot('Position',[pd(1)+(i-1)*of(1),pd(2)+(j-1)*of(2),pd(3),pd(4)]);
xtmp=0.5*(C1_ce1_SvEDGx(1:end-1)+C1_ce1_SvEDGx(2:end));
ytmp=0.5*(C1_ce1_SvEDGy(1:end-1)+C1_ce1_SvEDGy(2:end));  
ztmp=transpose(C1_ce1_SvCNT);
contourf(xtmp,ytmp,ztmp,'EdgeColor','none')
set(gca,'FontName','Times','FontSize',pfnt,'Ticklength',tl)
axis([C1min,C1max,ce1min,ce1max])
ylabel('$C_{\varepsilon1}$','Interpreter','Latex','FontSize',fnt)
i=3; j=4;
subplot('Position',[pd(1)+(i-1)*of(1),pd(2)+(j-1)*of(2),pd(3),pd(4)]);
contourf(ytmp,xtmp,transpose(ztmp),'EdgeColor','none')
set(gca,'FontName','Times','FontSize',pfnt,'Ticklength',tl)
axis([ce1min,ce1max,C1min,C1max])

i=1; j=1;
subplot('Position',[pd(1)+(i-1)*of(1),pd(2)+(j-1)*of(2),pd(3),pd(4)]);
xtmp=0.5*(C1_ce2_SvEDGx(1:end-1)+C1_ce2_SvEDGx(2:end));
ytmp=0.5*(C1_ce2_SvEDGy(1:end-1)+C1_ce2_SvEDGy(2:end));  
ztmp=transpose(C1_ce2_SvCNT);
contourf(xtmp,ytmp,ztmp,'EdgeColor','none')
set(gca,'FontName','Times','FontSize',pfnt,'Ticklength',tl)
axis([C1min,C1max,ce2min,ce2max])
ylabel('$C_{\varepsilon2}$','Interpreter','Latex','FontSize',fnt)
xlabel('$C_1$','Interpreter','Latex','FontSize',fnt)
i=4; j=4;
subplot('Position',[pd(1)+(i-1)*of(1),pd(2)+(j-1)*of(2),pd(3),pd(4)]);
contourf(ytmp,xtmp,transpose(ztmp),'EdgeColor','none')
set(gca,'FontName','Times','FontSize',pfnt,'Ticklength',tl)
axis([ce2min,ce2max,C1min,C1max])

i=3; j=3;
subplot('Position',[pd(1)+(i-1)*of(1),pd(2)+(j-1)*of(2),pd(3),pd(4)]);
ytmp=0.5*(C2_ce1_SvEDGx(1:end-1)+C2_ce1_SvEDGx(2:end));
xtmp=0.5*(C2_ce1_SvEDGy(1:end-1)+C2_ce1_SvEDGy(2:end));  
ztmp=(C2_ce1_SvCNT);
contourf(xtmp,ytmp,ztmp,'EdgeColor','none')
set(gca,'FontName','Times','FontSize',pfnt,'Ticklength',tl)
axis([ce1min,ce1max,C2min,C2max])
i=2; j=2;
subplot('Position',[pd(1)+(i-1)*of(1),pd(2)+(j-1)*of(2),pd(3),pd(4)]);
contourf(ytmp,xtmp,transpose(ztmp),'EdgeColor','none')
set(gca,'FontName','Times','FontSize',pfnt,'Ticklength',tl)
axis([C2min,C2max,ce1min,ce1max])

i=4; j=3;
subplot('Position',[pd(1)+(i-1)*of(1),pd(2)+(j-1)*of(2),pd(3),pd(4)]);
ytmp=0.5*(C2_ce2_SvEDGx(1:end-1)+C2_ce2_SvEDGx(2:end));
xtmp=0.5*(C2_ce2_SvEDGy(1:end-1)+C2_ce2_SvEDGy(2:end));  
ztmp=(C2_ce2_SvCNT);
contourf(xtmp,ytmp,ztmp,'EdgeColor','none')
set(gca,'FontName','Times','FontSize',pfnt,'Ticklength',tl)
axis([ce2min,ce2max,C2min,C2max])
i=2; j=1;
subplot('Position',[pd(1)+(i-1)*of(1),pd(2)+(j-1)*of(2),pd(3),pd(4)]);
contourf(ytmp,xtmp,transpose(ztmp),'EdgeColor','none')
set(gca,'FontName','Times','FontSize',pfnt,'Ticklength',tl)
axis([C2min,C2max,ce2min,ce2max])
xlabel('$C_2$','Interpreter','Latex','FontSize',fnt)

i=3; j=1;
subplot('Position',[pd(1)+(i-1)*of(1),pd(2)+(j-1)*of(2),pd(3),pd(4)]);
xtmp=0.5*(ce1_ce2_SvEDGx(1:end-1)+ce1_ce2_SvEDGx(2:end));
ytmp=0.5*(ce1_ce2_SvEDGy(1:end-1)+ce1_ce2_SvEDGy(2:end));  
ztmp=transpose(ce1_ce2_SvCNT);
contourf(xtmp,ytmp,ztmp,'EdgeColor','none')
set(gca,'FontName','Times','FontSize',pfnt,'Ticklength',tl)
axis([ce1min,ce1max,ce2min,ce2max])
xlabel('$C_{\varepsilon1}$','Interpreter','Latex','FontSize',fnt)
i=4; j=2;
subplot('Position',[pd(1)+(i-1)*of(1),pd(2)+(j-1)*of(2),pd(3),pd(4)]);
contourf(ytmp,xtmp,transpose(ztmp),'EdgeColor','none')
set(gca,'FontName','Times','FontSize',pfnt,'Ticklength',tl)
axis([ce2min,ce2max,ce1min,ce1max])

set(gcf,'PaperPositionMode','auto','InvertHardCopy','off')
print(gcf,'-dpng','-r600',['./figure1_cut_' num2str(cutk) '.png'])

[C,I]=max(count(:));
[i1,i2,i3,i4]=ind2sub(size(count),I);
C1m=mid{1}(i1)
C2m=mid{2}(i2)
ce1m=mid{3}(i3)
ce2m=mid{4}(i4)
frac=100*length(iSave)/N

C1t=1.5;
C2t=0.8;
ce1t=1.44;
ce2t=1.92;

%Axisymmetric expansion
C1=C1m; C2=C2m; ce1=ce1m; ce2=ce2m;
[Tnke_axi_exp,Ynke_axi_exp]=ode23(@rans_axi_exp,0:0.001:5,[1,1,0,0,0,0,0,0]);
C1=C1t; C2=C2t; ce1=ce1t; ce2=ce2t;
[Tnke_axi_exp_old,Ynke_axi_exp_old]=ode23(@rans_axi_exp,0:0.001:5,[1,1,0,0,0,0,0,0]);

%Axisymmetric contraction
C1=C1m; C2=C2m; ce1=ce1m; ce2=ce2m;
[Tnke_axi_con,Ynke_axi_con]=ode23(@rans_axi_con,0:0.01:2/abs(S11),[1,1,0,0,0,0,0,0]);
C1=C1t; C2=C2t; ce1=ce1t; ce2=ce2t;
[Tnke_axi_con_old,Ynke_axi_con_old]=ode23(@rans_axi_con,0:0.01:2/abs(S11),[1,1,0,0,0,0,0,0]);

%Pure shear
C1=C1m; C2=C2m; ce1=ce1m; ce2=ce2m;
[Tnke_shear,Ynke_shear]=ode23(@rans_shear,0:0.01:15,[1,1,0,0,0,0,0,0]);
C1=C1t; C2=C2t; ce1=ce1t; ce2=ce2t;
[Tnke_shear_old,Ynke_shear_old]=ode23(@rans_shear,0:0.01:15,[1,1,0,0,0,0,0,0]);

%Plane strain
C1=C1m; C2=C2m; ce1=ce1m; ce2=ce2m;
[Tnke_plane,Ynke_plane]=ode23(@rans_plane,0:0.01:5,[1,1,0,0,0,0,0,0]);
C1=C1t; C2=C2t; ce1=ce1t; ce2=ce2t;
[Tnke_plane_old,Ynke_plane_old]=ode23(@rans_plane,0:0.01:5,[1,1,0,0,0,0,0,0]);

%Periodic shear
C1=C1m; C2=C2m; ce1=ce1m; ce2=ce2m;
[Tnke_period1,Ynke_period1]=ode23(@rans_period1,0:0.01:15,[1,1,0,0,0,0,0,0]);
[Tnke_period2,Ynke_period2]=ode23(@rans_period2,0:0.01:15,[1,1,0,0,0,0,0,0]);
[Tnke_period3,Ynke_period3]=ode23(@rans_period3,0:0.01:15,[1,1,0,0,0,0,0,0]);
[Tnke_period4,Ynke_period4]=ode23(@rans_period4,0:0.01:15,[1,1,0,0,0,0,0,0]);
[Tnke_period5,Ynke_period5]=ode23(@rans_period5,0:0.01:15,[1,1,0,0,0,0,0,0]);
C1=C1t; C2=C2t; ce1=ce1t; ce2=ce2t;
[Tnke_period1_old,Ynke_period1_old]=ode23(@rans_period1,0:0.01:15,[1,1,0,0,0,0,0,0]);
[Tnke_period2_old,Ynke_period2_old]=ode23(@rans_period2,0:0.01:15,[1,1,0,0,0,0,0,0]);
[Tnke_period3_old,Ynke_period3_old]=ode23(@rans_period3,0:0.01:15,[1,1,0,0,0,0,0,0]);
[Tnke_period4_old,Ynke_period4_old]=ode23(@rans_period4,0:0.01:15,[1,1,0,0,0,0,0,0]);
[Tnke_period5_old,Ynke_period5_old]=ode23(@rans_period5,0:0.01:15,[1,1,0,0,0,0,0,0]);

pfnt=12;
fnt=14;
tl=[0.02,0.02];
msize=8;
lwid=2;

figure(2)
clf;
set(gcf,'Units','inches','Position',[12 12 5.5 5],'Color','w')
subplot('Position',[0.11,0.1,0.86,0.88]);
h(1)=plot((1/2.45)*Tnke_axi_exp,Ynke_axi_exp(:,1),'-b','LineWidth',lwid);
hold on;
plot(axi_exp_k(:,1),axi_exp_k(:,2),'ok','MarkerSize',msize,'LineWidth',1.1)
h(2)=plot((1/0.179)*Tnke_axi_con,Ynke_axi_con(:,1),'-r','LineWidth',lwid);
plot(axi_con_k(:,1),axi_con_k(:,2),'dk','MarkerSize',msize,'LineWidth',1.1)
h(3)=plot((1/0.296)*Tnke_shear,Ynke_shear(:,1),'-g','LineWidth',lwid);
plot(shear_k(:,1),shear_k(:,2),'sk','MarkerSize',msize,'LineWidth',1.1)
h(4)=plot((1/2)*Tnke_plane,Ynke_plane(:,1),'-k','LineWidth',lwid);
plot(plane_k(:,1),plane_k(:,2),'^k','MarkerSize',msize,'LineWidth',1.1)
hold off;
set(gca,'FontName','Times','FontSize',pfnt,'Ticklength',tl)
axis([0,2,0,2.3])
ylabel('$k/k_0$','Interpreter','Latex','FontSize',fnt)
xlabel('$S\cdot t$','Interpreter','Latex','FontSize',fnt)
legend(h,{'Axisymmetric expansion','Axisymmetric contraction',...
    'Pure shear','Plane strain'},'Location','NorthWest','Box','off',...
    'FontSize',fnt,'Interpreter','Latex')
set(gcf,'PaperPositionMode','auto','InvertHardCopy','off')
print(gcf,'-dpng','-r600',['./figure2_cut_' num2str(cutk) ...
                                    '_perc_' num2str(frac) ...
                                    '_C1_' num2str(C1) ...
                                    '_C2_' num2str(C2) ...
                                    '_Ce1_' num2str(ce1) ...
                                    '_Ce2_' num2str(ce2) '.png'])
   
%%
figure(3)
clf;
set(gcf,'Units','inches','Position',[12 12 5.5 5],'Color','w')
subplot('Position',[0.11,0.1,0.86,0.88]);
h(1)=semilogy(S0*Tnke_period1,Ynke_period1(:,1),'-b','LineWidth',lwid);
hold on;
h(2)=plot(S0*Tnke_period2,Ynke_period2(:,1),'-r','LineWidth',lwid);
h(3)=plot(S0*Tnke_period3,Ynke_period3(:,1),'-g','LineWidth',lwid);
h(4)=plot(S0*Tnke_period4,Ynke_period4(:,1),'-k','LineWidth',lwid);
h(5)=plot(S0*Tnke_period5,Ynke_period5(:,1),'-m','LineWidth',lwid);
plot(period1_k(:,1),period1_k(:,2),'ob','MarkerSize',msize,'LineWidth',1.1)
plot(period2_k(:,1),period2_k(:,2),'dr','MarkerSize',msize,'LineWidth',1.1)
plot(period3_k(:,1),period3_k(:,2),'sg','MarkerSize',msize,'LineWidth',1.1)
plot(period4_k(:,1),period4_k(:,2),'^k','MarkerSize',msize,'LineWidth',1.1)
plot(period5_k(:,1),period5_k(:,2),'*m','MarkerSize',msize,'LineWidth',1.1)
hold off;
set(gca,'FontName','Times','FontSize',pfnt,'Ticklength',tl)
axis([0,50,0,15])
ylabel('$k/k_0$','Interpreter','Latex','FontSize',fnt)
xlabel('$S\cdot t$','Interpreter','Latex','FontSize',fnt)
legend(h,{'$\beta=0.125$','$\beta=0.25$',...
    '$\beta=0.50$','$\beta=0.75$','$\beta=1.0$'},'Location','NorthWest','Box','off',...
    'FontSize',fnt,'Interpreter','Latex')
set(gcf,'PaperPositionMode','auto','InvertHardCopy','off')
print(gcf,'-dpng','-r600',['./figure3_cut_' num2str(cutk) ...
                                    '_perc_' num2str(frac) ...
                                    '_C1_' num2str(C1) ...
                                    '_C2_' num2str(C2) ...
                                    '_Ce1_' num2str(ce1) ...
                                    '_Ce2_' num2str(ce2) '.png'])

