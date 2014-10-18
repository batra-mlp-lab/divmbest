function SvmSegm_approximate_segms_ucm_smart(exp_dir, img_names, mask_type, name)
    DefaultVal('*name', '''SP_approx''');
    
    sp_approx_dir = [exp_dir 'MySegmentsMat/' mask_type '_' name '/'];
    if(~exist(sp_approx_dir, 'dir'))
        mkdir(sp_approx_dir);
    end

    % VOC annotations have a small "dontcare" strip around them. We'll try
    % to fix that.
    trainval12_imgs = textread([exp_dir 'ImageSets/Segmentation/trainval.txt'], '%s');
    trainval12_mirror_imgs = textread([exp_dir 'ImageSets/Segmentation/trainval_mirror.txt'], '%s');
    
    parfor (i=1:numel(img_names))
        filename = [sp_approx_dir img_names{i} '.mat'];
                
        if(exist(filename, 'file'))
            continue;
        end
        
        %t = tic();
        
        % load segments
        masks = myload([exp_dir 'MySegmentsMat/' mask_type '/' img_names{i} '.mat'], 'masks');
                
        ucm2 = myload([exp_dir 'PB/' img_names{i} '_PB.mat'], 'ucm2');
        assert(size(ucm2,1) > size(masks,1));
        
        labels2 = bwlabel(ucm2 <= 5);
        label_img = labels2(2:2:end, 2:2:end);

        sp = uint16(label_img);
        
        I = imread([exp_dir 'JPEGImages/' img_names{i} '.jpg']);
        if (strcmp(mask_type, 'ground_truth') && ((~isempty(intersect(img_names{i}, trainval12_imgs)) || ~isempty(intersect(img_names{i}, trainval12_mirror_imgs)))))
            % dilate ground truth masks in trainval, because of "dontcare" borders
            se = strel('disk',2);
            masks = imdilate(masks,se);
        end
        
        [sp_app, masks] = approximate_segm_sp(sp, masks, 0.4); % 0.4 seems fine
        
        % filter out empty masks
        areas = zeros(size(masks,3),1);        
        for j=1:size(masks,3)
            areas(j) = sum(sum(masks(:,:,j)));
        end
        empty_masks = areas==0;
        
        sp_app(:, empty_masks) = [];
        masks(:,:,empty_masks) = [];        
        
        mysave_n_compress(filename, 'masks', masks, 'sp_app', sp_app, 'sp', sp);
    end
end
