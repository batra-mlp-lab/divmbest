function [Feats] = squash_features(Feats, type)
    if(strcmp(type, 'power'))
        % power scaling proposed by perronin et al, eccv 2010 
        % (0.75 seemed like a good value)
        power = 0.75;
        for i = 1:size(Feats,2)
            Feats(:,i) = sign(Feats(:,i)).*abs(Feats(:,i)).^power;
        end        
    elseif(strcmp(type, 'inv_power'))
        power = 1/0.75;
        for i = 1:size(Feats,2)
            Feats(:,i) = sign(Feats(:,i)).*abs(Feats(:,i)).^power;
        end                
    elseif(strcmp(type, 'asinh'))
        for i = 1:size(Feats,2)
            Feats(:,i) = asinh(Feats(:,i));
        end
    elseif(strcmp(type, 'sigmoid'))
        for i = 1:size(Feats,2)
            ppp = 0.75; % 0.75 seems fine
            %fea(:,i) = (1.0./(1.0+exp(-ppp*fea(:,i))));          
            fea(:,i) = (((1.0./(1.0+exp(-ppp*Feats(:,i))))* 2) - 1);          
        end
    else
        disp('no such type implemented');
    end
end

