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
         W = Fit.latents.W(end,:);

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
         
         fprintf(o, 'onset\tduration\ttrial_type\trt\tchoice\toutcome\tValue1\tValue2\tValue_c\tW1\tW2\tW3\n');
         
         for t = 1:length(onsets)
             fprintf(o,'%.3f\t%.3f\t%s\t%.3f\t%i\t%0.3f\t%0.3f\t%0.3f\t%0.3f\t%0.3f\t%0.3f\t%0.3f\n', ...
                 onsets(t,:), 0, 'self',rt(t,:),choice(t,:),outcome(t,:),value1(t,:),value2(t,:),value_c(t,:),W(1,:));
         end
         
         fclose(o);
        
     end

end