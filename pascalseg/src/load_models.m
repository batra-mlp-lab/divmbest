function [models, w] = load_models(folder_models, categories, iter, chunk)    
    for i=1:numel(categories)
        if(isempty(iter))
            var = load([folder_models categories{i} '_chunk_' int2str(chunk) '.mat']);
        else            
            var = load([folder_models categories{i} '_iter_' int2str(iter) '_chunk_' int2str(chunk) '.mat']);
        end
        models{i} = var.model;
        w{i} = models{i}.w';
    end
    w = cell2mat(w);
end