function [boxes] = testmodel_mmodes(name,model,test,suffix,nummodes,one_scale,lambda,type)
% boxes = [boxes] = testmodel_mmodes(name,model,test,suffix,nummodes,one_scale,lambda)
% Returns candidate bounding boxes after DivMBest

divmbest_globals;

if ~exist('one_scale') || isempty(one_scale)
    one_scale = 0;
end

if ~exist('lambda') || isempty(lambda)
    lambda = -1e10;
end

DivMBest = 0;
Perturb = 0;
Nbest = 0;

if(strcmp(type, 'divmbest'))
    DivMBest = 1;
end

if(strcmp(type, 'perturb'))
    Perturb = 1;
end

if(exist([ cachedir type '_' name '_boxes_' num2str(nummodes) '_' num2str(lambda) '_' suffix '.mat'], 'file'))
    load([ cachedir type '_' name '_boxes_' num2str(nummodes) '_' num2str(lambda) '_' suffix '.mat']);
else
    boxes = cell(1,length(test));
    parfor i = 1:length(test)
        fprintf([name ': testing: %d/%d\n'],i,length(test));
        im = imread(test(i).im);
        
        if( DivMBest || Perturb )
            boxes_modes = detect_fast_divmbest(im,model,model.thresh,nummodes,one_scale,lambda, type);
            boxes_modes = modes_highest(boxes_modes);
            boxes{i} = boxes_modes;
        end
    end
    save([ cachedir type '_' name '_boxes_' num2str(nummodes) '_' num2str(lambda) '_' suffix '.mat'], 'boxes','model');
end
end
