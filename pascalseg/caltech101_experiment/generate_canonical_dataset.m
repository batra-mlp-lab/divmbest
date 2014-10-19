addpath('../src/');

% structure the data 
!mkdir Caltech101/
!mkdir Caltech101/SegmentEval/

exp_dir = './Caltech101/';

% these are merely for adhering to canonical structure
gt_segm_obj_path = [exp_dir 'SegmentationObject/'];
gt_segm_cls_path = [exp_dir 'SegmentationClass/'];
mkdir(gt_segm_obj_path)
mkdir(gt_segm_cls_path)

% path to caltech images
img_path = '101_ObjectCategories/';
if(~exist(img_path, 'dir'))
    !wget http://www.vision.caltech.edu/Image_Datasets/Caltech101/101_ObjectCategories.tar.gz
    !tar xfz 101_ObjectCategories.tar.gz
end

classes = dir(img_path);
classes(1:2) = []; % './' and '../'

counter = 1;
N_VOC_DIGITS = 6;
cmap = VOClabelcolormap(256);

target_img_path = [exp_dir 'JPEGImages/'];
masks_dir = [exp_dir '/MySegmentsMat/'];
if(~exist(target_img_path, 'dir'))
    display('Generating ground truth meta-data...');
    mkdir(target_img_path);
    mkdir(masks_dir);

    for i=1:numel(classes)
        cls_path = [img_path classes(i).name '/'];
        imgs = dir([cls_path '*.jpg']);

        for j=1:numel(imgs)
            img_names{counter} = ['2012_' sprintf(['%0' int2str(N_VOC_DIGITS) 'd'], counter)];

            % copy image file
            copyfile([cls_path imgs(j).name], [target_img_path img_names{counter} '.jpg']);

            % generate ground truth files
            I = imread([target_img_path img_names{counter} '.jpg']);
            obj_I = 2*ones(size(I,1), size(I,2)); % set always a single object
            imwrite(obj_I, cmap, [gt_segm_obj_path img_names{counter} '.png']);
            cls_I = (i+1)*ones(size(I,1), size(I,2));
            imwrite(cls_I, cmap, [gt_segm_cls_path img_names{counter} '.png']);

            counter = counter + 1;
        end
    end
end
    
% generate ground truth "segmentations"
SvmSegm_generate_gt_masks(exp_dir, img_names);

% create all.txt img list
imgset_dir = [exp_dir 'ImageSets/Segmentation/'];
mkdir(imgset_dir);
f = fopen([imgset_dir 'all.txt'], 'w');
for i=1:numel(img_names)
    fprintf(f, '%s\n', img_names{i});
end
fclose(f);





