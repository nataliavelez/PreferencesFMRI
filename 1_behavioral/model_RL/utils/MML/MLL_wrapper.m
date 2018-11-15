clear all
col_code(1,:) = [0.2980392156862745, 0.4470588235294118, 0.6901960784313725];
col_code(2,:) = [0.3333333333333333, 0.6588235294117647, 0.40784313725490196];
col_code(3,:) = [0.7686274509803922, 0.3058823529411765, 0.3215686274509804];

% Load in CSV
d0 = csvread('../../prescreen_data.csv');
Sub = unique(d0(:,1));
nSub = length(Sub);

%Convert keys to 1 and 2 so that it can used as an index 
d0(:,4) = d0(:,4) + 1;

run_train = 1;
run_test = 1;

if run_train
    for training_set = 1:2
        % Set up global fitting parameters
        Fit.Subjects = Sub;
        Fit.Model = 'MLL';
        Fit.NIter = 5;
        
        Fit.Start = ones(1,length(Fit.Subjects));
        Fit.End = ones(1,length(Fit.Subjects))*56;
        
        Fit.Nparms = 4; % Number of Fre Parms; Beta + 3 weights
        Fit.LB = [0.000001 -50 -50 -50];
        Fit.UB = [50 50 50 50];
        
        Fit.Priors.Use(1) = 1;   % use (gamma) priors on the Beta (softmax) parameter?
        Fit.Priors.Parms(1,1) = 2;
        Fit.Priors.Parms(1,2) = 3;
        
        Fit.Priors.Use(2) = 0;   % use (gamma) priors on the Beta (softmax) parameter?
        Fit.Priors.Parms(2,1) = 0;
        Fit.Priors.Parms(2,2) = 10;
        
        Fit.Priors.Use(3) = 0;   % use (gamma) priors on the Beta (softmax) parameter?
        Fit.Priors.Parms(3,1) = 0;
        Fit.Priors.Parms(3,2) = 10;
        
        Fit.Priors.Use(4) = 0;   % use (gamma) priors on the Beta (softmax) parameter?
        Fit.Priors.Parms(4,1) = 0;
        Fit.Priors.Parms(4,2) = 10;
        
        %% Training
        for s = 1:nSub
            
            fprintf('Subject %d... (index %d) \n',Fit.Subjects(s),s)
            
            train_Data = d0((d0(:,1) == s) & (d0(:,2) == training_set),:);
            choices = train_Data(:,4);
            stim1 = train_Data(:,6:8);
            stim2 = train_Data(:,9:11);
            
            Fit.NTrials(s) = length(train_Data);
            
            for iter = 1:Fit.NIter
                
                Fit.init(s,iter,:) = rand(1,length(Fit.LB)).*(Fit.UB-Fit.LB)+Fit.LB;
                if any(Fit.init(s,iter,:)==inf)
                    Fit.init(s,iter,find(Fit.init(s,iter,:)==inf)) = rand*5;
                end
                
                [res,lik,flag,out,lambda,grad,hess] = ...
                    fmincon(@(x) MLL_train(choices,stim1,stim2,Fit.Priors,x),...
                    Fit.init(s,iter,:),[],[],[],[],Fit.LB,Fit.UB,[],optimset('maxfunevals',5000,'maxiter',2000,...
                    'GradObj','off','DerivativeCheck','off','LargeScale','off','Algorithm','active-set','Hessian','off'));

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
                [[1:s]' Fit.Result.CorrectedLikPerTrial]  % to view progress so far
                
            end
        end
        
        % Saving Data
        [a,b] = min(Fit.Result.Lik,[],2);
        d = length(hess); % how many parameters are we fitting
        
        for s = 1:length(Fit.Subjects)
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
            Fit.Result.BestFit
        end
        
        Trained_Weights{training_set} = Fit;
        clear Fit

    end
    save ('MLL_training_results','Trained_Weights')
    
end

%%  Testing
if run_test
    load('MLL_training_results');
    
    for training_set = 1:2
        Fit = Trained_Weights{training_set};
        if training_set == 1
            test_set = 2;
        else
            test_set = 1;
        end
        
        for s = 1:nSub
            Fit.Priors.Use(1) = 0;
            
            test_Data = d0((d0(:,1) == s) & (d0(:,2) == test_set),:);
            
            choices = test_Data(:,4);
            stim1 = test_Data(:,6:8);
            stim2 = test_Data(:,9:11);
            
            opt_parms(1) = Fit.Result.BestFit(s,2);
            opt_parms(2) = Fit.Result.BestFit(s,3);
            opt_parms(3) = Fit.Result.BestFit(s,4);
            opt_parms(4) = Fit.Result.BestFit(s,5);
            
            [lik,latents{s,1}] = MLL_train(choices,stim1,stim2,Fit.Priors,opt_parms);
        end
        
        for s = 1:nSub
            this_choiceprob = latents{s,1}.choice_prob;
            pctCorrect(s,training_set) = sum(this_choiceprob > 0.5)/56;
        end
    end
    
    for s = 1:nSub
        fprintf('Subject %i \n',s);
        fprintf('2-Fold CV Accuracy = %0.3f \n',mean(pctCorrect(s,1)));
        %Positive; Historical; Romance;
        fprintf('Phase 1 Weights: [%0.2f, %0.2f, %0.2f]\n',Trained_Weights{1}.Result.BestFit(s,3:5));
        %Positive; Historical; Romance;
        fprintf('Phase 2 Weights: [%0.2f, %0.2f, %0.2f]\n',Trained_Weights{2}.Result.BestFit(s,3:5));       
    end
 
end


    
    
   
