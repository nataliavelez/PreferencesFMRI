function [lik,latents] = CMLL_train(choice,stim1,stim2,FitParms,X,self_weights,other_weights)

Beta = X(1);
Eta = X(2);

Self_Valence = self_weights(1);
Self_Setting = self_weights(2);
Self_Genre = self_weights(3);

Other_Valence = other_weights(1);
Other_Setting = other_weights(2);
Other_Genre = other_weights(3);

lik = 0;

latents.SelfV = NaN(length(choice),2);
latents.OtherV= NaN(length(choice),2);

latents.Vchosen = NaN(length(choice),1);
latents.V = NaN(length(choice),2);
latents.choice_prob = NaN(length(choice),1);
latents.choice_max = NaN(length(choice),1);

for t = 1:length(choice)
    if ~isnan(choice(t))
        
        SelfV(1) = Self_Valence * stim1(t,1) + Self_Setting * stim1(t,2) + Self_Genre * stim1(t,3);
        SelfV(2) = Self_Valence * stim2(t,1) + Self_Setting * stim2(t,2) + Self_Genre * stim2(t,3);
        
        OtherV(1) = Other_Valence * stim1(t,1) + Other_Setting * stim1(t,2) + Other_Genre * stim1(t,3);
        OtherV(2) = Other_Valence * stim2(t,1) + Other_Setting * stim2(t,2) + Other_Genre * stim2(t,3);
        
        V(1) = Eta * SelfV(1) + (1-Eta)*OtherV(1);
        V(2) = Eta * SelfV(2) + (1-Eta)*OtherV(2);

        
       lik_trial = Beta*V(choice(t)) - logsumexp(Beta*V);
       lik = lik + lik_trial;
       
       latents.SelfV = SelfV;
       latents.OtherV = OtherV;
       latents.Vchosen(t) = V(choice(t));
       latents.V(t,:) = V;
       latents.choice_prob(t) = exp(lik_trial);
       [a, latents.choice_max(t)] = max(V); 
       
    end
end

lik = -lik;

for i = 1
    if (FitParms.Use(i))    % putting a Gamma prior on Beta
        lik = lik - log(gampdf(X(i),FitParms.Parms(i,1),FitParms.Parms(i,2)));
    end
end

end