function D = pool_features_fast(pooling_type, D, in_triu, conditioning_sigma, pooling_weights, SPEEDUP, mask_spd)
    if(strcmp(pooling_type, 'avg'))
        %if(isempty(D))
        %    D = zeros(size(in_triu,1),1);
        %else
            D = sum(D,2)/size(D,2);
        %end
    elseif(strcmp(pooling_type, 'max'))
        %if(isempty(D))
        %    D = zeros(size(in_triu,1),1);
        %else
            D = max(D,[],2);
        %end
    else
        % second order pooling
        if(strcmp(pooling_type, 'log_avg'))
            %D = bsxfun(@minus,D,sum(D,2)/size(D,2));

            if(isempty(pooling_weights)) % no local features inside the region
                X = zeros(size(in_triu), 'single');
            else
                if SPEEDUP
                    %t = tic();
                    if(size(in_triu,1) ~=size(mask_spd,1))
                        % there are shape varying features
                        thisD = D;                                        
                        X_sp = -inf(size(in_triu), 'single');
                        range = 1:size(mask_spd,1);

                        X_sp(range, range) = mask_spd;
                        range_shape_feats = max(range)+1:size(X_sp,1);
                        if(numel(range_shape_feats)~=0)
                            f = thisD(range_shape_feats, :);
                            X_sp(range_shape_feats, :) = (f*thisD');
                            X_sp = max(X_sp, X_sp');
                        end
                    else
                        X_sp = mask_spd;
                    end                    
                    X_sp = X_sp / (sum(pooling_weights)+eps);
                    %X_sp = X_sp ./ (size(thisD,2));
                    %toc(t)

                    X = X_sp;

                    if 0
                        % debug
                        t = tic();
                        X_gt = thisD*thisD'./(size(thisD,2));
                        toc(t)
                        disp('done');
                        sc(abs(X_gt - X))
                        max(max(abs(X_gt - X)))
                    end
                else
                    pooling_weights = pooling_weights/(sum(pooling_weights)+eps);
                    D = bsxfun(@times, D, sqrt(pooling_weights));
                    X = D*D';

                    %t = tic();
                    %X = D*D'/(size(D,2));
                    %toc(t)
                end

                %mean(mean(abs(X-X_sp)))
                %sc(X - X_sp)
            end
                       
            try
                X=logm(X + conditioning_sigma*eye(size(X)));
                newX = real(X);
            catch
                newX = zeros(size(in_triu), 'single');
            end
        elseif(strcmp(pooling_type, 'avg2p'))
            newX = (D*D')./(size(D,2));
        elseif(strcmp(pooling_type, 'max2p'))
            if(~isempty(D))                
                newX = max_o2p(D);
            end
        else
            error('no such pooling type');
        end

        if(numel(newX)==1)
            newX = zeros(size(in_triu), 'single');
        end

        D =  newX(in_triu);
    end
end