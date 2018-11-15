%% self-phase
% YC Leong 10/19
% This script loads in the fits from an RL model trained on the training
% phase, and then outputs the value of stimuli in the testing phase 
% IMPORTANT: NOTE THAT THE ONSETS ASSUME DELETION OF FIRST 3TRs
% 



function testing_phase_RL(subj_id)
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
%                             Calculate value of each stimulus                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
load(sprintf('modelfits/%s_train_RL3feature.mat',subj_id));

%% Get value and timing information
for s = 1

    behav_path = dir(fullfile(dirs.session,sprintf('%s*.test*.mat',subj_id)));
    
    % Break it down into runs again    
     for r = 1:length(behav_path)
         thisData = load(fullfile(dirs.session,behav_path(r).name));
         
         clear onsets duration trial_type rt choice outcome V value1 value2 value_c W options
         
         onsets = thisData.Data.true_ons' - 6;
         duration = zeros(length(onsets), 1);
         trial_type = 'testing_phase';
         rt = thisData.Data.rt';
         choice = thisData.Data.subject_choice';
         outcome = thisData.Data.correct';
         
         last_valid_trial = max(find(~isnan(Fit.latents.W(:,1))));
         
         W = Fit.latents.W(last_valid_trial,:);

         % get options
         options = thisData.Data.options;
         option1 = NaN(length(options),3);
         option2 = NaN(length(options),3);
         
         for t = 1:length(options)
             option1(t,:) = movies(options(t,1)).features;
             option2(t,:) = movies(options(t,2)).features;
         end
         
         value1 = option1 * W';
         value2 = option2 * W';
         
         value_c = NaN(length(value1),1);
         
         % find chosen value
         for t = 1:length(value1)
             switch choice(t)
                 case 1
                     value_c(t) = value1(t);
                 case 2
                     value_c(t) = value2(t);
             end
         end
   
         % create TSV file
         outfile = sprintf('testing_phase_RL/%s.tsv',behav_path(r).name(1:end-4));
         o = fopen(outfile,'w+');
         
         fprintf(o, 'onset\tduration\ttrial_type\tresponse_time\tchoice\toutcome\tValue_c\tValue1\tValue2\tW1\tW2\tW3\tstim_1\tvalence_1\tsetting_1\tgenre_1\tstim_2\tvalence_2\tsetting_2\tgenre_2\n');
         
         for t = 1:length(onsets)
             fprintf(o,'%.3f\t%.3f\t%s\t%.3f\t%i\t%0.3f\t%0.3f\t%0.3f\t%0.3f\t%0.3f\t%0.3f\t%0.3f\t', ...
                 onsets(t,:), 0, 'self',rt(t,:),choice(t,:),outcome(t,:),value_c(t,:),value1(t,:),value2(t,:),W(1,:));
             
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