%function calc_optimal_exit(long_travel_time,short_travel_time,stay_time)
% calc_optimal_exit.m
% Author: Angela Ianni
% This runs MVT simulations for the foraging task to calculate the optimal
% exit values

clc; close all;

%% Set Common Parameters:
% Decision parameters: 1 = stay, 2 = leave
max_block_time = 390; % Set block time (390 seconds per block, 4 blocks total)
minimum_R = 0.5; % Set minimum reward
stay_time = 3; % (3) set time it takes to harvest - apple time (0.5 sec) + delay to reward within a patch (2.5 sec)
long_travel_time = 12;
short_travel_time = 6;

h = stay_time; 
% Set parameters for beta distribution to draw decay rate (k) from
alpha_shallow = 31.55811;
alpha_steep = 14.908728;
beta_shallow = 1.896899;
beta_steep = 2.033008;

plot_individual_parameters = 0;
plot_final_simulation_results = 0;
simulations = 100000;

%% Long-steep Environment
% Long steep (lst): 
%    Mean depletion rate: 0.9433
%    Travel time: 12 seconds
% Mean reward at exit: ~5.88
disp('Working on long-steep environment');
travel_time = long_travel_time; 

for s=1:simulations
    % Initialize parameters for simulation
    %disp(s);
    time = 0;
    iteration = 0;
    R_total = 0;
    k = 1; % initialize to no predicted decay
    decision = 2; % leave decision to start off so reward (R) is initialized from normal distribution
    while time < max_block_time
        iteration = iteration + 1;
        % Set reward for this iteration
        if decision == 2
            R(iteration) = normrnd(10,1); % Random number drawn from normal distribution with mean 10, SD 1, max 15
            if R(iteration) > 15 % Max initial reward is 15
                R(iteration) = 15;
            elseif R(iteration) < minimum_R
                R(iteration) = 0.5;
            end
            time = time + travel_time;
        else
            k(iteration)=betarnd(alpha_steep,beta_steep); % calculate decay rate
            R(iteration) = k(iteration)*R(iteration-1);
            if R(iteration) < minimum_R
                R(iteration) = 0.5;
            end
            time = time + h;
        end
        R_total = R_total + R(iteration);
        
        % Calculate average reward rate (p)
        p = R_total/time;
        
        % Determine optimal decision
        k_av = mean(k(k~=0)); % calculate the average decay rate observed
        R_pred = k_av*R(iteration); % calculate the predicted reward if stay at current tree
        %    Decision rule: leave if k*R<p*h   
        if R_pred < (p*h) % if predicted reward is less than average reward rate
            decision = 2;
        else
            decision = 1;
        end
        dec(iteration) = decision;
        R_total_tracker(iteration) = R_total;
        k_av_tracker(iteration) = k_av;
        p_tracker(iteration) = p;
        R_pred_tracker(iteration) = R_pred;

    end
    leaves = find(dec-1);
    
    % Average the R at leave decision and the one from the prior trial
    R_leaves_tracker(s)=mean((R(leaves)+R(leaves-1))/2);
end

disp(sprintf('Mean reward at exit (%d simulations): %0.5f\n',simulations,mean(R_leaves_tracker)));

if plot_final_simulation_results == 1
    f5=figure;
    figure(f5);
    subplot(2,2,1);
    plot(1:simulations,R_leaves_tracker,'bo','LineWidth',3,'MarkerSize',8);
    set(gca,'FontSize',24);
    axis([0 simulations min(R_leaves_tracker)-0.01 max(R_leaves_tracker)+0.01]);
    hold on;
    plot(1:simulations,mean(R_leaves_tracker).*ones(simulations,1),'m--','LineWidth',3);
    title(sprintf('Long-Steep Environment: %0.5f',mean(R_leaves_tracker)));
    xlabel('Simulations');
    ylabel('Reward at exit');
end

R_leaves_tracker_allfour(1,1)=mean(R_leaves_tracker);
R_leaves_tracker_allfour(1,2)=std(R_leaves_tracker);

%% Long-shallow Environment
% Long shallow (lsh):
%    Mean depletion rate: 0.88
%k=betarnd(alpha_shallow,beta_shallow);
%    Travel time: 12 seconds
% Mean reward at exit: ~6.56
disp('Working on long-shallow environment');
clear R R_leaves_tracker R_pred R_pred_tracker R_total R_total_tracker dec decision iteration k k_av k_av_tracker l leaves p p_tracker s time travel_time;
travel_time = long_travel_time; %(12)

for s=1:simulations
    % Initialize parameters for simulation
    time = 0;
    iteration = 0;
    R_total = 0;
    k = 1; % initialize to no predicted decay
    decision = 2; % leave decision to start off so reward (R) is initialized from normal distribution
    while time < max_block_time
        iteration = iteration + 1;
        % Set reward for this iteration
        if decision == 2
            R(iteration) = normrnd(10,1); % Random number drawn from normal distribution with mean 10, SD 1, max 15
            if R(iteration) > 15 % Max initial reward is 15
                R(iteration) = 15;
            elseif R(iteration) < minimum_R
                R(iteration) = 0.5;
            end
            time = time + travel_time; 
        else
            k(iteration)=betarnd(alpha_shallow,beta_shallow); % calculate decay rate
            R(iteration) = k(iteration)*R(iteration-1);
            if R(iteration) < minimum_R
                R(iteration) = 0.5;
            end
            time = time + h;
        end

        R_total = R_total + R(iteration);

        % Calculate average reward rate (p)
        p = R_total/time;
        % Determine optimal decision
        k_av = mean(k(k~=0)); % calculate the average decay rate observed
        R_pred = k_av*R(iteration); % calculate the predicted reward if stay at current tree
        %    Decision rule: leave if k*R<p*h   
        if R_pred < (p*h) % if predicted reward is less than average reward rate
            decision = 2;
        else
            decision = 1;
        end

        dec(iteration) = decision;
        R_total_tracker(iteration) = R_total;
        k_av_tracker(iteration) = k_av;
        p_tracker(iteration) = p;
        R_pred_tracker(iteration) = R_pred;

    end

    leaves = find(dec-1);
    
    % Average the R at leave decision and the one from the prior trial
    R_leaves_tracker(s)=mean((R(leaves)+R(leaves-1))/2);
end

disp(sprintf('Mean reward at exit (%d simulations): %0.5f\n',simulations,mean(R_leaves_tracker)));


%% Short-steep Environment
% Short steep (sst):
%    Mean depletion rate: 0.9433
%k=betarnd(alpha_steep,beta_steep);
%    Travel time: 6 seconds
% Mean reward at exit: ~7.74
disp('Working on short-steep environment');
clear R R_leaves_tracker R_pred R_pred_tracker R_total R_total_tracker dec decision iteration k k_av k_av_tracker l leaves p p_tracker s time travel_time;
travel_time = short_travel_time; 

for s=1:simulations
    % Initialize parameters for simulation
    time = 0;
    iteration = 0;
    R_total = 0;
    k = 1; % initialize to no predicted decay
    decision = 2; % leave decision to start off so reward (R) is initialized from normal distribution
    while time < max_block_time
        iteration = iteration + 1;
        % Set reward for this iteration
        if decision == 2
            %disp('Last decision was to leave')
            R(iteration) = normrnd(10,1); % Random number drawn from normal distribution with mean 10, SD 1, max 15
            if R(iteration) > 15 % Max initial reward is 15
                R(iteration) = 15;
            elseif R(iteration) < minimum_R
                R(iteration) = 0.5;
            end
            time = time + travel_time;
        else
            k(iteration)=betarnd(alpha_steep,beta_steep); % calculate decay rate
            R(iteration) = k(iteration)*R(iteration-1);
            if R(iteration) < minimum_R
                R(iteration) = 0.5;
            end
            time = time + h;
        end

        R_total = R_total + R(iteration);

        % Calculate average reward rate (p)
        p = R_total/time;
        % Determine optimal decision
        k_av = mean(k(k~=0)); % calculate the average decay rate observed
        R_pred = k_av*R(iteration); % calculate the predicted reward if stay at current tree
        %    Decision rule: leave if k*R<p*h   
        if R_pred < (p*h) % if predicted reward is less than average reward rate
            decision = 2;
        else
            decision = 1;
        end

        dec(iteration) = decision;
        R_total_tracker(iteration) = R_total;
        k_av_tracker(iteration) = k_av;
        p_tracker(iteration) = p;
        R_pred_tracker(iteration) = R_pred;

    end

    leaves = find(dec-1);
    
    % Average the R at leave decision and the one from the prior trial
    R_leaves_tracker(s)=mean((R(leaves)+R(leaves-1))/2);
end

disp(sprintf('Mean reward at exit (%d simulations): %0.5f\n',simulations,mean(R_leaves_tracker)));


%% Short-shallow Environment
% Short shallow (ssh): 
%    Mean depletion rate: 0.88
%k=betarnd(alpha_shallow,beta_shallow);
%    Travel time: 6 seconds
% Mean reward at exit: ~8.05 

disp('Working on short-shallow environment');
clear R R_leaves_tracker R_pred R_pred_tracker R_total R_total_tracker dec decision iteration k k_av k_av_tracker l leaves p p_tracker s time travel_time;
travel_time = short_travel_time; % (6)

for s=1:simulations
    % Initialize parameters for simulation
    time = 0;
    iteration = 0;
    R_total = 0;
    k = 1; % initialize to no predicted decay
    decision = 2; % leave decision to start off so reward (R) is initialized from normal distribution
    while time < max_block_time
        iteration = iteration + 1;
        % Set reward for this iteration
        if decision == 2
            R(iteration) = normrnd(10,1); % Random number drawn from normal distribution with mean 10, SD 1, max 15
            if R(iteration) > 15 % Max initial reward is 15
                R(iteration) = 15;
            elseif R(iteration) < minimum_R
                R(iteration) = 0.5;
            end
            time = time + travel_time; 
        else
            k(iteration)=betarnd(alpha_shallow,beta_shallow); % calculate decay rate
            R(iteration) = k(iteration)*R(iteration-1);
            if R(iteration) < minimum_R
                R(iteration) = 0.5;
            end
            time = time + h;
        end
        R_total = R_total + R(iteration);

        % Calculate average reward rate (p)
        p = R_total/time;

        % Determine optimal decision
        k_av = mean(k(k~=0)); % calculate the average decay rate observed
        R_pred = k_av*R(iteration); % calculate the predicted reward if stay at current tree
        %    Decision rule: leave if k*R<p*h   
        if R_pred < (p*h) % if predicted reward is less than average reward rate
            decision = 2;
        else
            decision = 1;
        end

        dec(iteration) = decision;
        R_total_tracker(iteration) = R_total;
        k_av_tracker(iteration) = k_av;
        p_tracker(iteration) = p;
        R_pred_tracker(iteration) = R_pred;

    end

    leaves = find(dec-1);
    
    % Average the R at leave decision and the one from the prior trial
    R_leaves_tracker(s)=mean((R(leaves)+R(leaves-1))/2);
end

disp(sprintf('Mean reward at exit (%d simulations): %0.5f\n',simulations,mean(R_leaves_tracker)));
