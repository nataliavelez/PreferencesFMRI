function [Eta, Predictions, Model_Acc, Fit, latents] = CMLL_wrapper(sub,test_data,self_weights,other_weights)

% Set up global fitting parameters
Fit.Subjects = sub;
Fit.Model = 'CMLL';
Fit.NIter = 5;

Fit.Nparms = 2; % Number of Fre Parms; Beta + 3 weights
Fit.LB = [0.0000001 0];
Fit.UB = [50 1];

Fit.Priors.Use(1) = 1;   % use (gamma) priors on the Beta (softmax) parameter?
Fit.Priors.Parms(1,1) = 2;
Fit.Priors.Parms(1,2) = 3;

choices = test_data.choice;
stim1 = test_data.option1;
stim2 = test_data.option2;

Fit.NTrials = length(choices) - sum(isnan(choices));
        
for iter = 1:Fit.NIter
    
    Fit.init(iter,:) = rand(1,length(Fit.LB)).*(Fit.UB-Fit.LB)+Fit.LB;
    if any(Fit.init(iter,:)==inf)
        Fit.init(iter,find(Fit.init(iter,:)==inf)) = rand*5;
    end
    
    [res,lik,flag,out,lambda,grad,hess] = ...
        fmincon(@(x) CMLL_train(choices,stim1,stim2,Fit.Priors,x,self_weights,other_weights),...
        Fit.init(iter,:),[],[],[],[],Fit.LB,Fit.UB,[],optimset('maxfunevals',5000,'maxiter',2000,...
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


% Saving Data
[a,b] = min(Fit.Result.Lik,[],2);
d = length(hess); % how many parameters are we fitting
%

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
Fit.Result.BestFit;

%%  Testing
Fit.Priors.Use(1) = 0;
choices = test_data.choice;
stim1 = test_data.option1;
stim2 = test_data.option2;

opt_parms(1) = Fit.Result.BestFit(2);
opt_parms(2) = Fit.Result.BestFit(3);

[lik,latents] = CMLL_train(choices,stim1,stim2,Fit.Priors,opt_parms,self_weights,other_weights);

this_choiceprob = latents.choice_prob;
pctCorrect = sum(this_choiceprob > 0.5)/length(choices);

fprintf('Subject %i \n',sub);
fprintf('Model Accuracy = %0.3f \n',mean(pctCorrect));
fprintf('Eta = %0.2f \n',opt_parms(2));

Eta = opt_parms(2);
Predictions = latents.choice_max;
Model_Acc = pctCorrect;

latents.lik = lik;
latents.BIC = lik + (Fit.Nparms/2*log(Fit.NTrials));
latents.AverageBIC = -latents.BIC/Fit.NTrials;
latents.CorrectedLikPerTrial = exp(latents.AverageBIC);


end
    
    
   
