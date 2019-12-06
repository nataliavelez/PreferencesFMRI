%% Combined model 
% YC Leong 12/19
% This script take self and other weights, and estimates the extent to
% which a participant is allo or ego-ecentric in his/her preferences

function Eta = CMLL_model(subj_id)

%subj_id = 'sll_opusfmri_01';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                   Set Parameters                                     %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
addpath(genpath('utils/'))

basepath = '../../../inlab';
% basepath = '/scratch/groups/hyo/OPUS/';
dirs.session = fullfile(basepath,'session_data');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                     Read in JSON Files                               %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% load movie data
load('movie_info.mat');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                             Fit RL Model to all data                               %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Fit model
for s = 1
    
    % Load self vs. other weights
    load(sprintf('modelfits/%s_SelfPhase.mat',subj_id));
    self_weights = Fit.Result.BestFit(1,3:5);
    
    load(sprintf('modelfits/%s_OtherWeights.mat',subj_id));
    other_weights =Fit.Result.BestFit(1,3:5);
    
    clear Fit

    % Initialist parameters
    Fit.Subjects = 1;
    Fit.NIter = 3;
    Fit.Start = ones(1,length(Fit.Subjects));
    Fit.End = ones(1,length(Fit.Subjects))*112;
    Fit.Nparms = 2; % Eata and Softmax
    Fit.LB = [0.0000001 0];
    Fit.UB = [50 1];

    Fit.Priors.Use(1) = 1;   % use (gamma) priors on the Beta (softmax) parameter?
    Fit.Priors.Parms(1,1) = 2;
    Fit.Priors.Parms(1,2) = 3;

    Fit.Priors.Use(2) = 0;   % use (gamma) priors on the Beta (softmax) parameter?

    fprintf('Subject %s... \n',subj_id)

   
    %% Get in lab data
    behav_path = dir(fullfile(dirs.session,sprintf('%s*.test*.mat',subj_id)));
    
    options = [];
    subject_choice = [];
    
    for r = 1:length(behav_path)
        thisData = load(fullfile(dirs.session,behav_path(r).name));
        
        options = [options; thisData.Data.options];
        subject_choice = [subject_choice, thisData.Data.subject_choice];
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

    %%
    Fit.NTrials(s) = length(choice);
    
    for iter = 1:Fit.NIter
        
        Fit.init(s,iter,:) = rand(1,length(Fit.LB)).*(Fit.UB-Fit.LB)+Fit.LB;
        if any(Fit.init(s,iter,:)==inf)
            Fit.init(s,iter,find(Fit.init(s,iter,:)==inf)) = rand*5;
        end
        
        [res,lik,flag,out,lambda,grad,hess] = ...
            fmincon(@(x) CMLL_train(choice,stim1,stim2,Fit.Priors,x,self_weights,other_weights),...
            Fit.init(s,iter,:),[],[],[],[],Fit.LB,Fit.UB,[],optimset('maxfunevals',5000,'maxiter',2000,...
            'GradObj','off','DerivativeCheck','off','LargeScale','off','Algorithm','active-set','Hessian','off','Display','off'));
        
        Fit.Result.Beta(:,iter) = res(1);
        Fit.Result.Eta(:,iter) = res(2);
        Fit.Result.Hessian(:,:,iter) = full(hess); % save the Hessian to do Laplace approx later; it was sparse initially, so use "full" to expand
        Fit.Result.Lik(iter) = lik;
        
        %Calculate BIC here...
        Fit.Result.BIC(iter) = lik + (Fit.Nparms/2*log(Fit.NTrials));
        Fit.Result.AverageBIC(iter) = -Fit.Result.BIC(iter)/Fit.NTrials;
        Fit.Result.CorrectedLikPerTrial(iter) = exp(Fit.Result.AverageBIC(iter));
        Fit.Result.CorrectedLikPerTrial;  % to view progress so far
        
    end
end

% Saving Data
[a,b] = min(Fit.Result.Lik,[],2);
d = length(hess); % how many parameters are we fitting

for s = 1
    Fit.Result.BestFit(1,:) = [Fit.Subjects,...
        Fit.Result.Beta(b),...
        Fit.Result.Eta(b),...
        Fit.Result.Lik(b),...
        Fit.Result.BIC(b),...
        Fit.Result.AverageBIC(b),...
        Fit.Result.CorrectedLikPerTrial(b)];
    % compute Laplace approximation at the ML point, using the Hessian
    %
    Fit.Result.Laplace = -a + 0.5*d*log(2*pi) - 0.5*log(det(squeeze(Fit.Result.Hessian(:,:,b))));
end

Fit.Result.BestFit

Eta = Fit.Result.BestFit(3);

fprintf('Subject %s: Eta = %0.3f\n',subj_id, Eta);
end