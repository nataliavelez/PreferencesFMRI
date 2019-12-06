%% otherWeights
% YC Leong 12/19
% This script fits an ideal observer model to the training dataset, and
% saves the fits in modelfits

function otherWeights(subj_id)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                   Set Parameters                                     %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
addpath(genpath('utils/'))

basepath = '../../../inlab';
% basepath = '/scratch/groups/hyo/OPUS/';

dirs.session = fullfile(basepath,'session_data');

% subj_id = 'sll_opusfmri_01';

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
Fit.LB = [0.000001 -50 -50 -50];
Fit.UB = [50 50 50 50];
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
            fmincon(@(x) MLL_train(choice,stim1,stim2,Fit.Priors,x),...
            Fit.init(s,iter,:),[],[],[],[],Fit.LB,Fit.UB,[],optimset('maxfunevals',5000,'maxiter',2000,...
            'GradObj','off','DerivativeCheck','off','LargeScale','off','Algorithm','active-set','Hessian','off','display','off'));
        
        Fit.Result.Beta(s,:,iter) = res(1);
        Fit.Result.Valence(s,:,iter) = res(2);
        Fit.Result.Setting(s,:,iter) = res(3);
        Fit.Result.Genre(s,:,iter) = res(4);
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
        Fit.Result.Valence(s,b(s)),...
        Fit.Result.Setting(s,b(s)),...
        Fit.Result.Genre(s,b(s)),...
        Fit.Result.Lik(s,b(s)),...
        Fit.Result.BIC(s,b(s)),...
        Fit.Result.AverageBIC(s,b(s)),...
        Fit.Result.CorrectedLikPerTrial(s,b(s))];
    % compute Laplace approximation at the ML point, using the Hessian
    
    Fit.Result.Laplace(s) = -a(s) + 0.5*d*log(2*pi) - 0.5*log(det(squeeze(Fit.Result.Hessian(s,:,:,b(s)))));
end

Fit.Result.BestFit

save(sprintf('modelfits/%s_OtherWeights.mat',subj_id),'Fit');

end
     
     
     
