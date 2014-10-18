function [masks, sp_app] = process_masks(original_masks, domain, sp_app)    
    if(strcmp(domain, 'ground'))
        masks = ~original_masks;
        sp_app = ~sp_app;
    elseif(strcmp(domain, 'figure'))
        masks = original_masks;
    else
        error('no such mode');
    end    
end