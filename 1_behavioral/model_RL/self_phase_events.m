%% self-phase
% YC Leong 10/19
% This script trains the ideal observer model on both pre-screen and self-phase task, and outputs a 3-column .mat and txt file
% with onset, duration, and magnitude for a "self-value" regressor.
%
% IMPORTANT: NOTE THAT THE ONSETS ASSUME DELETION OF FIRST 3TRs
% 
% Example Usage: self_phase_events('sll_opusfmri_01');  



function self_phase_events(subj_id)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                   Set Parameters                                     %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
addpath(genpath('utils/'))

dirs.prescreen = '../../../inlab/data_prescreen';
dirs.session = '../../../inlab/data_session';


% subj_id = 'sll_opusfmri_01';
prescreen_file = fullfile(dirs.prescreen, [subj_id '.json']);

TR = 2;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                     Read in JSON Files                               %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% load movie data
load('movie_info.mat');

fprintf('Reading subject %s \n',subj_id)
fid = fopen(prescreen_file);
raw = fread(fid,inf);
str = char(raw');
fclose(fid);
jsonData = JSON.parse(str);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                              Extract Required Info                                   %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Loop to extract required data (Choice and Condition)
thisData = jsonData.choice_trials;
choice = NaN(length(jsonData.choice_trials),1);
condition = NaN(length(jsonData.choice_trials),6);

for t = 1:length(jsonData.choice_trials)
    Choice(t,1) = thisData{1,t}.key + 1; % Adding one so that I can use it as an index
    
    % Condition/preference conventions:
    %   1 = positive, historical, romance
    %  -1 = negative, sci-fi, action
    
    Condition(t,1:3) = str2num(thisData{1,t}.condition{1});
    Condition(t,4:6) = str2num(thisData{1,t}.condition{2});
    
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                             Fit MML Model to all data                               %
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
    choice = Choice;
    condition = Condition;
    stim1 = condition(:,1:3);
    stim2 = condition(:,4:6);
    Fit.NTrials(s) = length(choice);

    %% Get in lab data
    behav_path = dir(fullfile(dirs.session,sprintf('%s*.self*.mat',subj_id)));
    
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
    
    stim1 = [stim1; option1];
    stim2 = [stim2; option2];
    choice = [choice; subject_choice'];

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

%% Get value and timing information
for s = 1

    behav_path = dir(fullfile(dirs.session,sprintf('%s*.self*.mat',subj_id)));
    for r = 1:length(behav_path)
        thisData = load(fullfile(dirs.session,behav_path(r).name));
        
        onsets = thisData.Data.true_ons' - 6;
        duration = zeros(length(onsets), 1);
        trial_type = 'self_phase';
        rt = thisData.Data.rt';
        choice = thisData.Data.subject_choice';
        options = thisData.Data.options;
        
        clear stim1
        clear stim2
        
        for t = 1:length(options)
            stim1(t,:) = movies(options(t,1)).features;
            stim2(t,:) = movies(options(t,2)).features;
        end
        
        % find value
        opt_parms(1) = Fit.Result.BestFit(s,2);
        opt_parms(2) = Fit.Result.BestFit(s,3);
        opt_parms(3) = Fit.Result.BestFit(s,4);
        opt_parms(4) = Fit.Result.BestFit(s,5);
        
        [lik,latents] = MLL_train(choice,stim1,stim2,Fit.Priors,opt_parms);
        
        self_value1 = latents.V(:,1);
        self_value2 = latents.V(:,2);
        self_value_c = latents.Vchosen;
        
        %onset duration trial_type rt choice self_value1 self_value2 self_value_c
        % create TSV file
        sub_no = subj_id(end-1:end);
        outfile = sprintf('/scratch/groups/hyo/OPUS/BIDS_data/sub-%s/func/sub-%s_task-%s_run-0%i_events.tsv', sub_no, sub_no, 'self', r);
                
        o = fopen(outfile,'w+');
        
        fprintf(o, 'onset\tduration\ttrial_type\tresponse_time\tchoice\tself_value_c\tself_value1\tself_value2\tstim_1\tvalence_1\tsetting_1\tgenre_1\tstim_2\tvalence_2\tsetting_2\tgenre_2\n');
        
        for t = 1:length(options)
            % print first round of stuff
            fprintf(o,'%.3f\t%.3f\t%s\t%.3f\t%i\t%0.3f\t%0.3f\t%0.3f\t', ...
                onsets(t,:), 0, 'self',rt(t,:),choice(t,:),self_value_c(t,:),self_value1(t,:),self_value2(t,:));
            
            stim_1 = options(t,1);
            valence_1 = movies(stim_1).valence;
            setting_1 = movies(stim_1).setting;
            genre_1 = movies(stim_1).genre;

            stim_2 = options(t,2);
            valence_2 = movies(stim_2).valence;
            setting_2 = movies(stim_2).setting;
            genre_2 = movies(stim_2).genre;
            
            % print second round of stuff
            fprintf(o,'%i\t%s\t%s\t%s\t%i\t%s\t%s\t%s\n', ...
                stim_1, valence_1, setting_1, genre_1, stim_2, valence_2, setting_2, genre_2);
   
        end
        
        fclose(o);
    end

end