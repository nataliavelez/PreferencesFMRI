function [lik,latents] = RL_3feature(choice,stim1,stim2,outcome,FitParms,X)

Beta = X(1);
Eta = X(2);

lik = 0;

latents.Vchosen = NaN(length(choice),1);
latents.V = NaN(length(choice),2);
latents.choice_prob = NaN(length(choice),1);
latents.W  = NaN(length(choice),3);
latents.PE = NaN(length(choice),1);

W_init = 0;
W = ones(1,3) * W_init;


for t = 1:length(choice)
    if ~isnan(choice(t))
        
       % choice 
       V(1) = W(:,1) * stim1(t,1) + W(:,2)  * stim1(t,2) + W(:,3) * stim1(t,3);
       V(2) = W(:,1) * stim2(t,1) + W(:,2) * stim2(t,2) + W(:,3) * stim2(t,3);
        
       lik_trial = Beta*V(choice(t)) - logsumexp(Beta*V);
       lik = lik + lik_trial;
       
       latents.Vchosen(t) = V(choice(t));
       latents.V(t,:) = V;
       latents.choice_prob(t) = exp(lik_trial);
       latents.W(t,:) = W;
       
       
       % Learning
       PE  = outcome(t)-V(choice(t)); % prediction error on this trial
       
       switch choice(t)
           case 1
               stim = stim1(t,:);
           case 2
               stim = stim2(t,:);
       end

       W = W + stim * Eta *PE;
       
       latents.PE(t) = PE;
       
       clear stim
       
    else
        latents.W(t,:) = [NaN NaN NaN];     
    end
    
    
end

lik = -lik;

for i = 1:4
    
    if (FitParms.Use(i))    % putting a Gamma prior on Beta
        lik = lik - log(gampdf(X(i),FitParms.Parms(i,1),FitParms.Parms(i,2)));
    end

end

end