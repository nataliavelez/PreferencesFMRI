% plotEtas

load('Etas.mat');

figure(1)
set(gcf,'Position',[100 100 450 300]);

[n,x]=hist(Etas);
h=bar(x,n);
set(gca,'FontSize',14)
axis([0 1 0 3.2]);

xlabel('Eta')


fig_dest = 'Eta_hist';
set(gcf,'paperpositionmode','auto');
print('-dpng',fig_dest);
