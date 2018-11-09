%% make_events_wrapper

subjs = {'sll_opusfmri_01', 'sll_opusfmri_02', 'sll_opusfmri_03', 'sll_opusfmri_04', 'sll_opusfmri_05' ...
    'sll_opusfmri_06', 'sll_opusfmri_07', 'sll_opusfmri_08'};

for s = 1:length(subjs)
    sub = subjs{s};
    self_phase_events(sub);
    training_phase_RL(sub);
    testing_phase_RL(sub);
end