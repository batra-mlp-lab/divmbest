function new_masks = sp_approx_to_mask(sp, sp_approx)  
    n_masks = size(sp_approx,2);
    
    new_masks = false(size(sp,1), size(sp,2), n_masks);   
    parfor i=1:n_masks
        on = find(sp_approx(:,i));
        off = find(~sp_approx(:,i));
        
        if(numel(on)<numel(off))
            new_mask = false(size(sp));
            for j=1:numel(on)
                new_mask(sp==on(j)) = true;
            end
        else
            new_mask = true(size(sp));
            for j=1:numel(off)
                new_mask(sp==off(j)) = false;
            end            
        end
        new_masks(:,:,i) = new_mask;
    end
end