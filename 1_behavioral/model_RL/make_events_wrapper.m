%% make_events_wrapper

subjs = {'sll_opusfmri_01', 'sll_opusfmri_02', 'sll_opusfmri_03', 'sll_opusfmri_04', 'sll_opusfmri_05' ...
    'sll_opusfmri_06', 'sll_opusfmri_07', 'sll_opusfmri_08'};

run_vanilla = 0;     %  Make original regressors
run_CMLL = 1;        %  Run combined model to estimate eta

if run_CMLL 
   Etas = NaN(length(subjs),1); 
end


for s = 1:length(subjs)
    sub = subjs{s};
    
    %  Make original regressors
    if run_vanilla
        self_phase_events(sub);
        training_phase_RL(sub);
        testing_phase_RL(sub);
    end
    
    % Run combined model
    if run_CMLL
        otherWeights(sub)
        Etas(s,1) = CMLL_model(sub);
   
    end
    
end

save('CMLL_model/Etas.mat','Etas')