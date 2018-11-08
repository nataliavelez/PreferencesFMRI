function [lik,latents] = MLL_train(choice,stim1,stim2,FitParms,X)

Beta = X(1);
Valence = X(2);
Setting = X(3);
Genre = X(4);

lik = 0;
latents.Vchosen = NaN(length(choice),1);
latents.V = NaN(length(choice),2);
latents.choice_prob = NaN(length(choice),1);

for t = 1:length(choice)
    if ~isnan(choice(t))
        
        V(1) = Valence * stim1(t,1) + Setting * stim1(t,2) + Genre * stim1(t,3);
        V(2) = Valence * stim2(t,1) + Setting * stim2(t,2) + Genre * stim2(t,3);
        
       lik_trial = Beta*V(choice(t)) - logsumexp(Beta*V);
       lik = lik + lik_trial;
       
       latents.Vchosen(t) = V(choice(t));
       latents.V(t,:) = V;
       latents.choice_prob(t) = exp(lik_trial);
       
    end
end

lik = -lik;

for i = 1:4
    if (FitParms.Use(i))    % putting a Gamma prior on Beta
        lik = lik - log(gampdf(X(i),FitParms.Parms(i,1),FitParms.Parms(i,2)));
    end

end

end