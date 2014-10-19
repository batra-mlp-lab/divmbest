function generate_canonical_dataset(exp_dir)    
    % obtain the most recent pascal voc annotations
    voc_file = 'VOCtrainval_11-May-2012.tar';
    voc_link = 'http://pascallin.ecs.soton.ac.uk/challenges/VOC/voc2012/VOCtrainval_11-May-2012.tar';

    mkdir(exp_dir);
    mkdir([exp_dir 'SegmentEval/']);

    % Desired path to VOC images
    img_path = [exp_dir 'JPEGImages/'];
    % We will train using ground truth annotations for all images, with
    % the additional Berkeley annotations

    if(~exist('./benchmark.tgz', 'file'))
        % Get the Berkeley annotations
        disp('Downloading external voc ground truth annotations.');
        disp('See http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/');
        pause(5);
        
        !wget http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz

        % Unpack Berkeley annotations, store in VOC format
        !tar xfz benchmark.tgz

        files = dir('./benchmark_RELEASE/dataset/inst/*.mat');
        mkdir([exp_dir 'SegmentationObject/']);
        mkdir([exp_dir 'SegmentationClass/']);
        cmap = VOClabelcolormap(256);
        parfor i=1:numel(files)
            var = load(['./benchmark_RELEASE/dataset/inst/' files(i).name]);
            I_obj = var.GTinst.Segmentation;
            I_cls = I_obj;
            for j=1:numel(var.GTinst.Categories)
                I_cls(I_obj==j) = var.GTinst.Categories(j);
            end

            I_obj = I_obj+1;
            I_cls = I_cls+1;
            imwrite(I_obj, cmap, [exp_dir 'SegmentationObject/' files(i).name(1:end-4) '.png']);
            imwrite(I_cls, cmap, [exp_dir 'SegmentationClass/' files(i).name(1:end-4) '.png']);
        end
    end

    if(~exist(voc_file, 'file'))
        % Get the latest higher-quality Pascal VOC annotations and replace
        % any overlapping Berkeley annotations by these
        disp(['Downloading PASCAL VOC dataset. See  http://pascallin.ecs.soton.ac.uk/challenges/VOC/']);
        pause(5);
        
        system(['wget ' voc_link]);
        system(['tar xf ' voc_file]);
        mkdir(img_path);
        voc_dir = dir('VOCdevkit');
        voc_dir(1:2) = [];
        system(['cp VOCdevkit/' voc_dir.name '/JPEGImages/* ' img_path]);
        system(['cp VOCdevkit/' voc_dir.name '/SegmentationObject/* ' exp_dir 'SegmentationObject/']);
        system(['cp VOCdevkit/' voc_dir.name '/SegmentationClass/* ' exp_dir 'SegmentationClass/']);
        system(['cp -r VOCdevkit/' voc_dir.name '/Annotations/ ' exp_dir]);
    end

    % Download precomputed short lists of CPMC segments (up to 150 per image)
    !wget http://www.isr.uc.pt/~joaoluis/o2p/cpmc_segms_o2p_release_v1.tgz
    !tar xfz cpmc_segms_o2p_release_v1.tgz

    % These files also contain ucm2 maps, we'll put them in the right folder
    mkdir([exp_dir 'PB/']);
    files = dir('cpmc_segms_150/*.mat');
    disp('Unpacking ucm2 files.');    
    
    for i=1:numel(files)
        var = load(['cpmc_segms_150/' files(i).name], 'ucm2');
        ucm2 = var.ucm2;
        if(exist([exp_dir 'PB/' files(i).name(1:end-4) '_PB.mat'], 'file'))
            save([exp_dir 'PB/' files(i).name(1:end-4) '_PB.mat'],  'ucm2', '-append');
        else
            save([exp_dir 'PB/' files(i).name(1:end-4) '_PB.mat'],  'ucm2');
        end

        % Save also mirrored versions
        ucm2 = ucm2(:, -(-size(var.ucm2,2):-1));
        if(exist([exp_dir 'PB/' files(i).name(1:end-4) '_mirror_PB.mat'], 'file'))
            save([exp_dir 'PB/' files(i).name(1:end-4) '_mirror_PB.mat'],  'ucm2', '-append');
        else
            save([exp_dir 'PB/' files(i).name(1:end-4) '_mirror_PB.mat'],  'ucm2');
        end
    end

    % The tarball contained superpixel approximations of cpmc
    % segments. We'll now generate binary masks from them.
    files = dir('cpmc_segms_150/*.mat');
    masks_dir = [exp_dir 'MySegmentsMat/CPMC_segms_150_sp_approx/'];
    mkdir(masks_dir);
    disp('Decompressing CPMC segments.');
    
    for i=1:numel(files)
        filename = [masks_dir files(i).name(1:end-4) '.mat'];
        if(~exist(filename,'file'))
            vgg_progressbar('Decompressing cpmc segments.', i/numel(files), 5);
            var = load(['cpmc_segms_150/' files(i).name], 'sp_app', 'ucm2');

            labels2 = bwlabel(var.ucm2 <= 5); % 5 was used to create sp_app
            sp = uint16(labels2(2:2:end, 2:2:end));
            sp_app = var.sp_app;

            masks = sp_approx_to_mask(sp, sp_app);

            save([masks_dir files(i).name(1:end-4) '.mat'], 'masks', 'sp_app', 'sp');
        end
    end

    % Copy imgset files
    system(['cp -r VOCdevkit/' voc_dir.name '/ImageSets/ ' exp_dir]);

    % Create all_gt_segm imgset and variations
    imgset_dir = [exp_dir 'ImageSets/Segmentation/'];
    list_all_gt = dir([exp_dir 'SegmentationObject/*.png']);
    list_all_gt = {list_all_gt(:).name};
    list_all_gt_segm = cellfun(@(a) a(1:end-4), list_all_gt, 'UniformOutput', false);
    f = fopen([imgset_dir 'all_gt_segm.txt'], 'w');
    for i=1:numel(list_all_gt_segm)
        fprintf(f, '%s\n', list_all_gt_segm{i});
    end
    fclose(f);
    generate_setdiff_imgset(exp_dir, 'all_gt_segm', 'val');
    generate_imgset_mirror(exp_dir, 'all_gt_segm');
    generate_imgset_mirror(exp_dir, 'all_gt_segm_minus_val');
    generate_imgset_mirror(exp_dir, 'train');
    generate_imgset_mirror(exp_dir, 'val');
    generate_imgset_mirror(exp_dir, 'trainval');
    % Same for 2011 image sets
    system(['cp ./imgsets_voc11/* ' exp_dir 'ImageSets/Segmentation/' ]);
    generate_setdiff_imgset(exp_dir, 'all_gt_segm', 'val11');
    generate_imgset_mirror(exp_dir, 'all_gt_segm_minus_val11');
    generate_imgset_mirror(exp_dir, 'train11');
    generate_imgset_mirror(exp_dir, 'val11');

    % Generate ground truth masks for all images
    disp('Generating ground truth masks.');    
    img_names = textread([exp_dir 'ImageSets/Segmentation/all_gt_segm.txt'], '%s');
    SvmSegm_generate_gt_masks(exp_dir, img_names);

    disp('Creating mirrored images.');
    create_mirrored_images(exp_dir, 'all_gt_segm')
    mirror_imgset = 'all_gt_segm_mirror';

    disp('Generating ground truth masks for mirrored images.');
    img_names = textread([exp_dir 'ImageSets/Segmentation/' mirror_imgset '.txt'], '%s');
    SvmSegm_generate_gt_masks(exp_dir, img_names);

    % Now reconstruct the ground truth masks by sets of ucm superpixels as
    % well, so that feature extraction is more uniform later on
    disp('Approximating ground truth segments by superpixels.');
    img_names = textread([exp_dir 'ImageSets/Segmentation/all_gt_segm.txt'], '%s');
    SvmSegm_approximate_segms_ucm_smart(exp_dir, img_names, 'ground_truth', 'sp_approx');
    img_names = textread([exp_dir 'ImageSets/Segmentation/all_gt_segm_mirror.txt'], '%s');
    SvmSegm_approximate_segms_ucm_smart(exp_dir, img_names, 'ground_truth', 'sp_approx');

    % Compute all necessary segment metadata
    a = SegmBrowser(exp_dir, 'CPMC_segms_150_sp_approx', 'all_gt_segm');
    b = SegmBrowser(exp_dir, 'ground_truth_sp_approx', 'all_gt_segm');
    c = SegmBrowser(exp_dir, 'ground_truth_sp_approx', 'all_gt_segm_mirror');
    
    disp('Finished setting up all required inputs.');
end

