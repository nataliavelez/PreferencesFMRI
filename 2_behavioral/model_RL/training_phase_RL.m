%% self-phase
% YC Leong 10/19
% This script trains a reinforcement learning model on training phase data, and outputs a 3-column .csv file
% IMPORTANT: NOTE THAT THE ONSETS ASSUME DELETION OF FIRST 3TRs
% 
% First version: 3 feature No Decay No CounterFactual Updating
%
% Possible variants: 
%    3 vs. 6 feature RL
%    Non-decay vs. decay RL
%    Counter_factual updating
% Example Usage: self_phase_events('sll_opusfmri_01');  



function training_phase_RL(subj_id)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                   Set Parameters                                     %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
addpath(genpath('utils/'))

dirs.session = '../../../inlab/data_session';

% subj_id = 'sll_opusfmri_01';

TR = 2;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                     Read in JSON Files                               %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% load movie data
load('movie_info.mat');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                             Fit RL Model to all data                               %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear Fit

% Initialist parameters
Fit.Subjects = 1;
Fit.NIter = 3;
Fit.Start = ones(1,length(Fit.Subjects));
Fit.End = ones(1,length(Fit.Subjects))*112;
Fit.Nparms = 4; % Number of Fre Parms; Beta + 3 weights
Fit.LB = [0.000001 0 -50 -50];
Fit.UB = [50 1 50 50];
Fit.Priors.Use(1) = 1;   % use (gamma) priors on the Beta (softmax) parameter?
Fit.Priors.Parms(1,1) = 2;
Fit.Priors.Parms(1,2) = 3;
Fit.Priors.Use(2) = 0;   % use (gamma) priors on the Beta (softmax) parameter?
Fit.Priors.Use(3) = 0;   % use (gamma) priors on the Beta (softmax) parameter?
Fit.Priors.Use(4) = 0;   % use (gamma) priors on the Beta (softmax) parameter?

% Fit model
for s = 1
    
    fprintf('Subject %s... \n',subj_id)

   
    %% Get in lab data
    behav_path = dir(fullfile(dirs.session,sprintf('%s*.train*.mat',subj_id)));
    
    options = [];
    subject_choice = [];
    outcome = [];
    
    for r = 1:length(behav_path)
        thisData = load(fullfile(dirs.session,behav_path(r).name));
        
        options = [options; thisData.Data.options];
        subject_choice = [subject_choice, thisData.Data.subject_choice];
        outcome = [outcome, thisData.Data.correct];
    end
    
    option1 = NaN(length(options),3);
    option2 = NaN(length(options),3);

    for t = 1:length(options)
        option1(t,:) = movies(options(t,1)).features;
        option2(t,:) = movies(options(t,2)).features;
    end
    
    stim1 = option1;
    stim2 = option2;
    choice = subject_choice';
    outcome = outcome';

    %%
    Fit.NTrials(s) = length(choice);
    
    for iter = 1:Fit.NIter
        
        Fit.init(s,iter,:) = rand(1,length(Fit.LB)).*(Fit.UB-Fit.LB)+Fit.LB;
        if any(Fit.init(s,iter,:)==inf)
            Fit.init(s,iter,find(Fit.init(s,iter,:)==inf)) = rand*5;
        end
        
        [res,lik,flag,out,lambda,grad,hess] = ...
            fmincon(@(x) RL_3feature(choice,stim1,stim2,outcome,Fit.Priors,x),...
            Fit.init(s,iter,:),[],[],[],[],Fit.LB,Fit.UB,[],optimset('maxfunevals',5000,'maxiter',2000,...
            'GradObj','off','DerivativeCheck','off','LargeScale','off','Algorithm','active-set','Hessian','off','display','off'));
        
        Fit.Result.Beta(s,:,iter) = res(1);
        Fit.Result.Eta(s,:,iter) = res(2);
        Fit.Result.Hessian(s,:,:,iter) = full(hess); % save the Hessian to do Laplace approx later; it was sparse initially, so use "full" to expand
        Fit.Result.Lik(s,iter) = lik;
        
        %Calculate BIC here...
        Fit.Result.BIC(s,iter) = lik + (Fit.Nparms/2*log(Fit.NTrials(s)));
        Fit.Result.AverageBIC(s,iter) = -Fit.Result.BIC(s,iter)/Fit.NTrials(s);
        Fit.Result.CorrectedLikPerTrial(s,iter) = exp(Fit.Result.AverageBIC(s,iter));
        [[1:s]' Fit.Result.CorrectedLikPerTrial];  % to view progress so far
        
    end
end

% Saving Data
[a,b] = min(Fit.Result.Lik,[],2);
d = length(hess); % how many parameters are we fitting

for s = 1
    Fit.Result.BestFit(s,:) = [Fit.Subjects(s),...
        Fit.Result.Beta(s,b(s)),...
        Fit.Result.Eta(s,b(s)),...
        Fit.Result.Lik(s,b(s)),...
        Fit.Result.BIC(s,b(s)),...
        Fit.Result.AverageBIC(s,b(s)),...
        Fit.Result.CorrectedLikPerTrial(s,b(s))];
    % compute Laplace approximation at the ML point, using the Hessian
    
    Fit.Result.Laplace(s) = -a(s) + 0.5*d*log(2*pi) - 0.5*log(det(squeeze(Fit.Result.Hessian(s,:,:,b(s)))));
end

Fit.Result.BestFit


%% Get value and timing information
for s = 1

    behav_path = dir(fullfile(dirs.session,sprintf('%s*.train*.mat',subj_id)));
    
    % extract latents for all trials   
    for r = 1:length(behav_path)
        thisData = load(fullfile(dirs.session,behav_path(r).name));

        options = [];
        subject_choice = [];
        outcome = [];
        
        for r = 1:length(behav_path)
            thisData = load(fullfile(dirs.session,behav_path(r).name));
            
            options = [options; thisData.Data.options];
            subject_choice = [subject_choice, thisData.Data.subject_choice];
            outcome = [outcome, thisData.Data.correct];
        end
        
        option1 = NaN(length(options),3);
        option2 = NaN(length(options),3);
        
        for t = 1:length(options)
            option1(t,:) = movies(options(t,1)).features;
            option2(t,:) = movies(options(t,2)).features;
        end
        
        stim1 = option1;
        stim2 = option2;
        choice = subject_choice';
        outcome = outcome';
    end
    
    % find value
    opt_parms(1) = Fit.Result.BestFit(s,2);
    opt_parms(2) = Fit.Result.BestFit(s,3);
    
    [lik,latents] = RL_3feature(choice,stim1,stim2,outcome,Fit.Priors,opt_parms);
       
    % Save latents
    Fit.latents = latents;
    save(sprintf('modelfits/%s_train_RL3feature.mat',subj_id),'Fit');

    
    start_trial = 1;
    
    % Break it down into runs again    
     for r = 1:length(behav_path)
         thisData = load(fullfile(dirs.session,behav_path(r).name));
         
         clear onsets duration trial_type rt choice outcome PE value1 value2 value_c W
         
         onsets = thisData.Data.true_ons' - 6;
         duration = zeros(length(onsets), 1);
         trial_type = 'training_phase';
         rt = thisData.Data.rt';
         choice = thisData.Data.subject_choice';
         outcome = thisData.Data.correct';
         
         end_trial = start_trial + length(onsets)-1;
         
         PE = latents.PE(start_trial:end_trial,:);
         value1 = latents.V(start_trial:end_trial,1);
         value2 = latents.V(start_trial:end_trial,2);
         value_c = latents.Vchosen(start_trial:end_trial);
         W = latents.W(start_trial:end_trial,:);
         
         start_trial = start_trial + length(onsets);
         
         % create TSV file
         outfile = sprintf('training_phase_RL/%s.tsv',behav_path(r).name(1:end-4));
         o = fopen(outfile,'w+');
         
         fprintf(o, 'onset\tduration\ttrial_type\trt\tchoice\toutcome\tPE\tValue1\tValue2\tValue_c\tW1\tW2\tW3\n');
         
         for t = 1:length(onsets)
             fprintf(o,'%.3f\t%.3f\t%s\t%.3f\t%i\t%0.3f\t%0.3f\t%0.3f\t%0.3f\t%0.3f\t%0.3f\t%0.3f\t%0.3f\n', ...
                 onsets(t,:), 0, 'self',rt(t,:),choice(t,:),outcome(t,:),PE(t,:),value1(t,:),value2(t,:),value_c(t,:),W(t,:));
         end
         
         fclose(o);
        
     end
     
     
     
     

end