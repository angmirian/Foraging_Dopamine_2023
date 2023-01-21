%% Load data
clear all; clc; close all;
cd /Users/angela/Documents/DPHIL/Data/Imaging/Foraging_PET_data/Jan2023

%Load Behavioral Data
[behavdata,behavtext]=xlsread('demographic_data.xlsx','Apple Picking');
names=behavtext(2:end,2);
subjs=regexprep(names,' ','_');
behavdata(56,:)=[]; %exclude subject with odd performance suggesting not following task directions (lack of leave decisions for one condition of task)
load params.mat;
numsubjs=length(age);
short_shallow=behavdata(:,11);
short_steep=behavdata(:,12); 
long_shallow=behavdata(:,13); 
long_steep=behavdata(:,14);

% Load PET data
[PETdata,PETtext]=xlsread('PET_ROI_data.xlsx');
subjnums=PETdata(:,1); % get subject numbers

%% Plot behavioral results (Figure 2)
sig_mark=1; % set to 0 if you don't want to significance markers on the plots
optimal_exits=[5.88088,6.55581,7.73821,8.04002]; % optimal exit values I calculated from 100,000 simulations of the MVT
% Get data into correct format for plot
data_to_plot=[long_steep';long_shallow';short_steep';short_shallow'];

% Plot the optimal exits
grey=[0.8,0.8,0.8]; 
figure('Position',[50 50 1800 800]);
set(gcf,'color','w');
bar(optimal_exits,'FaceColor',grey,'EdgeColor','k','LineWidth',2); hold on; 
colors={[1 0 0];[1 0 1];[0 0 1];[0 0.75 0]};
% Finish editing from here; code copied from foraging_plots.m located in
% Data/Behavioral/Foraging
for count=1:4
    plot(count,data_to_plot(count,:),'o','MarkerEdgeColor',colors{count},'MarkerSize',16,'LineWidth',4);hold on;
    plot(count,nanmean(data_to_plot(count,:)),'diamond','MarkerEdgeColor','k','MarkerFaceColor',colors{count},'MarkerSize',36,'LineWidth',6); hold on;
end
mean_deviation_optimal_exit_thresholds=abs(nanmean(data_to_plot,1)-mean(optimal_exits));
lst_deviation_optimal=abs(data_to_plot(1,:)-optimal_exits(1));
ssh_deviation_optimal=abs(data_to_plot(4,:)-optimal_exits(4));
axis([0.5 4.5 0 12]);
xlabel(' Reward Environment ','fontname','arial','fontsize',45,'fontweight','bold'); 
ylabel(' Threshold for Leaving Patch ','fontname','arial','fontsize',45,'fontweight','bold');
set(gca,'Xtick',[1:4]);
set(gca,'fontsize',50,'fontweight','demi');
set(gca,'XTickLabel',{'long steep','long shallow','short steep','short shallow'},'fontsize',30,'fontweight','demi'); 

%ANOVA comparing leaving threshold across reward environments
anova_data=[long_steep,long_shallow;short_steep,short_shallow]; %rearrange data - rows are travel time (long and short), columns are decay rate (steep and shallow)
[p,tbl,stats] = anova2(anova_data,56); %two way anova with factors tarvel time and decay rate
disp(tbl); %disp table of ANOVA2 results

% Post-hoc t-tests to calculate p-values of leaving threshold mean between reward patches
if sig_mark==1 
    for i=1:4
        for j=1:4
            [h(i,j),p(i,j)]=ttest(data_to_plot(i,:),data_to_plot(j,:));
        end
    end
    % Add appropriate significance markers to plot
    H=sigstar({[1,2],[1,3],[1,4],[2,3],[2,4]},[p(1,2),p(1,3),p(1,4),p(2,3),p(2,4)]); % Only keep pairs that are significant for the plot
end

%display results from post-hoc t-tests
disp(sprintf("long-steep vs. long-shallow: %d",p(1,2)));
disp(sprintf("long-steep vs. short-steep: %d",p(1,3)));
disp(sprintf("long-steep vs. short-shallow: %d",p(1,4)));
disp(sprintf("long-shallow vs. short-steep: %d",p(2,3)));
disp(sprintf("long-shallow vs. short-shallow: %d",p(2,4)));
disp(sprintf("short-steep vs. short-shallow: %d",p(3,4)));

%% Get desired PET data for each individual tracer
% Load bilateral FDOPA data (putamen, caudate n., ventral striatum, MB)
fdopa_string={'FDOPA_Ki_045_B-Puta','FDOPA_Ki_045_B-Caud','FDOPA_Ki_045_B-VStr','FDOPA_Ki_045_B-MidB'};
for i=1:length(fdopa_string)
    [a(i),b(i)]=find(strcmp(PETtext,fdopa_string{i}));
end
FDOPA_data=PETdata(:,b);

% Load bilateral FALLY data (putamen, caudate n., ventral striatum, MB)
fally_string={'FALLY_BPnd_045_B-Puta','FALLY_BPnd_045_B-Caud','FALLY_BPnd_045_B-VStr','FALLY_BPnd_045_B-MidB'};
for i=1:length(fally_string)
    [c(i),d(i)]=find(strcmp(PETtext,fally_string{i}));
end
FALLY_data=PETdata(:,d);

% Load bilateral NNC data (putamen, caudate n., ventral striatum) - No MB
% b/c poor signal here (not really any D1 receptors there)
nnc_string={'NNC_BPnd_042_B-Puta','NNC_BPnd_042_B-Caud','NNC_BPnd_042_B-VStr'};
for i=1:length(nnc_string)
    [e(i),f(i)]=find(strcmp(PETtext,nnc_string{i}));
end
NNC_data=PETdata(:,f);

% Find FDOPA outlier and replace data with NaN values (subj 55; mean=0.091, std=9.09e-4, mean-3*std=0.0064)
[i,j]=find(FDOPA_data(:,1)<0.0064); FDOPA_data(i,:)=NaN;

%% Regress out effect of age & gender on PET data
% b = coefficient estimates for multilinear regression; treats NaNs in X or
% y as missing values, and ignores them
% bint = 95% confidence interval for the coefficient estimates
% r = residuals
% rint = an n-by-2 matrix rint of intervals that can be used to diagnose outliers. If the interval rint(i,:) for observation i does not contain zero, the corresponding residual is larger than expected in 95% of new observations, suggesting an outlier.
% stats = 1-by-4 vector stats that contains, in order, the R2 statistic, the F statistic and its p value, and an estimate of the error variance.
X=[ones(size(age)) age gender]; 
y=[FDOPA_data FDOPA_ACC FALLY_data FALLY_ACC NNC_data NNC_ACC]; 
for i = 1:size(y,2)
    [b,bint,r,rint,stats] = regress(y(:,i),X);
    meancentered_controlAgeGender_y(:,i)=r;
end

%% Normalise PET data, run PCA, and plot results
norm_PETdata=normaliseNaN(meancentered_controlAgeGender_y);
[coeff,score,latent,tsquared,explained,mu] = pca(norm_PETdata);

% make screen plot of eigenvalues
latent_cutoff=1; % 0.5 or 1
order=[5,1,2,3,4,10,6,7,8,9,14,11,12,13];
figure('Position',[50 50 1800 800]);
subplot(1,2,1);
plot(1:length(latent),latent,'bo-','LineWidth',4,'MarkerSize',25,'MarkerFaceColor','b'); set(gca,'FontSize',24);
ylabel('Eigenvalue');
xlabel('Component');
set(gca,'XTick',1:length(latent));
title('Scree Plot');
hold on;
plot(1:length(latent),latent_cutoff*ones(size(latent)),'k--','LineWidth',5);
xlim([0.5 length(latent)+0.5]);
subplot(1,2,2);
plot(1:length(explained),explained,'r*-','LineWidth',4,'MarkerSize',25,'MarkerFaceColor','r'); set(gca,'FontSize',24);
ylabel('Variance Explained');
xlabel('Component');
set(gca,'XTick',1:length(latent));
title('Percent of Variance Explained');

% Plot PCA coefficients (Figure 4c)
%calculate the color map
cmap=jet(256);
limits=max(max(coeff));
cmin=-limits;
cmax=limits;
m=length(cmap);

% get values from coefficients for figures
final_coefficients = coeff(order,find(latent>latent_cutoff));
c1_fdopa=final_coefficients(1:5,1);
c1_fally=final_coefficients(6:10,1);
c1_nnc=final_coefficients(11:end,1);

c2_fdopa=final_coefficients(1:5,2);
c2_fally=final_coefficients(6:10,2);
c2_nnc=final_coefficients(11:end,2);

c3_fdopa=final_coefficients(1:5,3);
c3_fally=final_coefficients(6:10,3);
c3_nnc=final_coefficients(11:end,3);

c4_fdopa=final_coefficients(1:5,4);
c4_fally=final_coefficients(6:10,4);
c4_nnc=final_coefficients(11:end,4);

index=fix((final_coefficients(14,4)-cmin)/(cmax-cmin)*m)+1; 
RGB = cmap(index,:);

% white out coefficients less than abs(0.2) for visualization
min_value=min(min(coeff));
tmp_coeff=coeff;
tmp_coeff(abs(coeff)<0.2)=-max(max(coeff))-0.1;

% Create plot for Figure 4c
figure('Position',[50 50 800 800]);
imagesc(tmp_coeff(order,find(latent>latent_cutoff)));
cmap=jet(256);
cmap(1,:)=1;
colormap(cmap);
caxis(gca,[-max(max(coeff)),max(max(coeff))]);
set(gca,'FontSize',24)
xlabel('Component')
set(gca,'YTick',1:20);
set(gca,'XTick',1:length(find(latent>latent_cutoff)));
labels={'FDOPA-ACC','FDOPA-Putamen','FDOPA-Caudate','FDOPA-Vstr','FDOPA-mb','FALLY-ACC','FALLY-Putamen','FALLY-Caudate','FALLY-Vstr','FALLY-mb','NNC-ACC','NNC-Putamen','NNC-Caudate','NNC-Vstr',};
set(gca,'YTickLabel',labels);
colorbar;
title('PET PCA - control age and gender');
hold on;
h1 = rectangle('position',[0.5 0.5 length(find(latent>latent_cutoff)) 4.9]);
set(h1,'EdgeColor',[1 0 0],'linewidth',7);
h2 = rectangle('position',[0.5 5.5 length(find(latent>latent_cutoff)) 4.9]);
set(h2,'EdgeColor',[0 0 1],'linewidth',7);
h3 = rectangle('position',[0.5 10.5 length(find(latent>latent_cutoff)) 4]);
set(h3,'EdgeColor',[0 1 0],'linewidth',7);

%% Run multiple regression with behavioral measures and PCA component scores (mean_thresh, thresh_change, mean_RT, RT_change)
thresh_change=ssh_thresh-lst_thresh;
RT_change=RT_ssh-RT_lst;
%Calculate the average threshold change due to travel time (short travel time - long travel time)
travel_time_change=mean([ssh_thresh,sst_thresh],2)-mean([lsh_thresh,lst_thresh],2);
%Calculate the average threshold change due to decay rate (shallow decay rate - steep decay rate)
decay_rate_change=mean([ssh_thresh,lsh_thresh],2)-mean([sst_thresh,lst_thresh],2);
foraging_data=[thresh_change RT_change travel_time_change decay_rate_change];

X=[score(:,1:4)]; %these are the PCA component weights

% Do multiple regression for each of the behaviors 
% Total change in leaving threshold
y=foraging_data(:,1);
mdl_1 = fitlm(X,y); % run multiple regression; intercept is included

% Total change in reaction time
y=foraging_data(:,2);
mdl_2 = fitlm(X,y); % run multiple regression

% Change in leaving threshold due to travel time 
y=foraging_data(:,3);
mdl_3 = fitlm(X,y); % matlab version of the multiple regression

% Change in leaving threshold due to decay rate
y=foraging_data(:,4);
mdl_4 = fitlm(X,y); % matlab version of the multiple regression

% Check if correlations between travel time and decay rate are significantly different
[r,p]=corr(travel_time_change, decay_rate_change,'rows','pairwise');
fprintf('travel-time and decay-rate change r = %.4f, p = %.4f\n',r,p);

%% Create plots for multiple regression results - total change in exit threshold and travel time vs. decay rate change
comps=[1,4]; % Just get data for PCA components 1 and 4

%Plot total change in exit threshold
figure('Position',[50 50 1800 800]);
X=ssh_thresh-lst_thresh;
for x=1:2
    subplot(1,2,x);
    i=comps(x);
    disp(i)
    y=score(:,i); % subject's component scores
    if x == 1
        plot(y,X(:,1),'o','Color','red','LineWidth',3,'MarkerSize',25,'MarkerFaceColor','red'); hold on;
        l=lsline;
    elseif x==2
         plot(y,X(:,1),'o','Color','red','LineWidth',3,'MarkerSize',25,'MarkerFaceColor','red'); hold on;
         l=lsline;
         xlim([-2.5 2.5]);
    end
    set(gca,'FontSize',24); 
    set(l,'LineWidth',10);
    xlabel('Component Score','FontSize',30);
    ylabel('Foraging Exit Change','FontSize',30)
    legend({'Total Change in Exit Threshold'},'FontSize',30);
end

% Plot travel time and decay rate
X=[travel_time_change,decay_rate_change];
figure('Position',[50 50 1800 800]);
for x=1:2
    subplot(1,2,x);
    i=comps(x);
    disp(i)
    y=score(:,i); % subject's component scores for
    plot(y,X(:,1),'o','Color',[0.3 0 0.6],'LineWidth',3,'MarkerSize',25,'MarkerFaceColor',[0.3 0 0.6]); hold on;
    plot(y,X(:,2),'o','Color',[1 0.5 0],'LineWidth',3,'MarkerSize',25,'MarkerFaceColor',[1 0.5 0]); 
    set(gca,'FontSize',24); 
    l=lsline;
    set(l,'LineWidth',10);
    xlabel('Component Score','FontSize',30);
    ylabel('Foraging Exit Change','FontSize',30)
    legend({'Travel Time','Decay Rate'},'FontSize',30);
    if x==2
        xlim([-3 3]);
    end
end

%% Plot Reaction Time (Figure 6b)
%Load the reaction time data to plot
lsh_RT=load('all_subjs_long_shallow_RT.txt');
lst_RT=load('all_subjs_long_steep_RT.txt');
ssh_RT=load('all_subjs_short_shallow_RT.txt');
sst_RT=load('all_subjs_short_steep_RT.txt');
%Remove data for subject with odd behavior (no leave decisions in one
%block)
lst_RT(56)=[];
lsh_RT(56)=[];
sst_RT(56)=[];
ssh_RT(56)=[];
new_all=[lst_RT lsh_RT sst_RT ssh_RT]'; % rearrange the data for the plot

figure('Position',[50 50 1800 800]);
sig_mark=1;
numsubjs=length(lsh_RT);
colors={[1 0 0];[1 0 1];[0 0 1];[0 1 0]};
for count=1:4
    h = errorbar(count,mean(new_all(count,:)),std(new_all(count,:))/sqrt(numsubjs),'color',colors{count},'LineWidth',5);
    h.CapSize = 12;
    hold on;
    plot(count,mean(new_all(count,:)),'diamond','MarkerEdgeColor','k','MarkerFaceColor',colors{count},'MarkerSize',30,'LineWidth',5); hold on;
end
axis([0.5 4.5 0.3 0.37]);
if sig_mark==1
    %Calculate p-values
    for i=1:4
        for j=1:4           
            [h_rt(i,j),p_rt(i,j)]=ttest(new_all(i,:),new_all(j,:));
        end
    end
    %delete pairs that aren't significant
    %H=sigstar({[1,2],[1,3],[1,4],[2,3],[2,4],[3,4]},[p_rt(1,2),p_rt(1,3),p_rt(1,4),p_rt(2,3),p_rt(2,4),p_rt(3,4)]);
    H=sigstar({[1,4],[2,4],[3,4]},[p_rt(1,4),p_rt(2,4),p_rt(3,4)]);
end
set(gca,'Xtick',[1:4])
set(gca,'XTickLabel',{'lst','lsh','sst','ssh'},'fontsize',30,'fontweight','demi');

disp('MEAN RTs');
disp(['long steep: ',num2str(mean(new_all(1,:)))]);
disp(['long shallow: ',num2str(mean(new_all(2,:)))]);
disp(['short steep: ',num2str(mean(new_all(3,:)))]);
disp(['short shallow: ',num2str(mean(new_all(4,:)))]);
disp('RT stats:');
%display results from post-hoc t-tests
disp(sprintf("long-steep vs. long-shallow: %d",p_rt(1,2)));
disp(sprintf("long-steep vs. short-steep: %d",p_rt(1,3)));
disp(sprintf("long-steep vs. short-shallow: %d",p_rt(1,4)));
disp(sprintf("long-shallow vs. short-steep: %d",p_rt(2,3)));
disp(sprintf("long-shallow vs. short-shallow: %d",p_rt(2,4)));
disp(sprintf("short-steep vs. short-shallow: %d",p_rt(3,4)));

%% Plot Average Reward Rate (Figure 6a)
figure('Position',[50 50 1800 800]);
load total_rewards.mat
vector1=[long_steep_total_rewards' long_shallow_total_rewards' short_steep_total_rewards' short_shallow_total_rewards']';
colors={[1 0 0];[1 0 1];[0 0 1];[0 1 0]};
for count=1:4 %390 seconds for each condition
    errorbar(count,mean(vector1(count,:)./390),std(vector1(count,:)./390)/sqrt(numsubjs),'color',colors{count},'LineWidth',5); %standard error
    hold on;
    plot(count,mean(vector1(count,:)./390),'diamond','MarkerEdgeColor','k','MarkerFaceColor',colors{count},'MarkerSize',30,'LineWidth',5);
end
xlim([0.5 4.5]);
ylim([1 2.3]);
set(gca,'Ytick',(1.2:0.2:2.2))
sig_mark=1;
if sig_mark==1
    %Calculate p-values
    for i=1:4
        for j=1:4
            [h1(i,j),p1(i,j)]=ttest(vector1(i,:),vector1(j,:));
        end
    end
    % Hand delete pairs that aren't significant
    %H=sigstar({[1,2],[1,3],[1,4],[2,3],[2,4],[3,4]},[p1(1,2),p1(1,3),p1(1,4),p1(2,3),p1(2,4),p1(3,4)]);
    H=sigstar({[1,2],[1,3],[1,4],[2,4],[3,4]},[p1(1,2),p1(1,3),p1(1,4),p1(2,4),p1(3,4)]);
end
ti=title('Average Reward Rates','Fontsize',28); 
xlabel(' Orchard type ','fontname','arial','fontsize',26,'fontweight','demi'); 
ylabel(' Average Reward Rate (apples/sec) ','fontname','arial','fontsize',26,'fontweight','demi');
set(gca,'Xtick',[1:4])
set(gca,'XTickLabel',{'lst','lsh','sst','ssh'},'fontsize',30,'fontweight','demi');

disp(['long steep: ',num2str(mean(vector1(1,:)))]);
disp(['long shallow: ',num2str(mean(vector1(2,:)))]);
disp(['short steep: ',num2str(mean(vector1(3,:)))]);
disp(['short shallow: ',num2str(mean(vector1(4,:)))]);
