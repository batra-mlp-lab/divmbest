function [D1, bbox] = compute_shape_varying_feats(mask, F1, xy, scale, shape, internal_shape, extra_xy, N_SHAPE_DIMS, variable_grids, base_scales, I) %, D_main_feat)
    D1 = [];
    if(sum(mask(:))==0)
        mask(1,1) = 1;
    end
    
    [f_i, f_j] = find(mask);
    fmax_i = max(f_i);
    fmin_i = min(f_i);
    fmax_j = max(f_j);
    fmin_j = min(f_j);
    width = (fmax_j - fmin_j);
    height = (fmax_i - fmin_i);        
    bbox = [fmin_i; fmin_j; height; width];
    if(xy)
        x = (F1(1,:)-fmin_j) / max(1,width);
        y = (F1(2,:)-fmin_i) / max(1,height);
        x2 = (F1(1,:)-fmin_j) / max(1,height);
        y2 = (F1(2,:)-fmin_i) / max(1,width);
        D1 = [single(D1); x; y; x2; y2];
    end

    if(extra_xy)        
        % 4 x 4 grid
        N = 4;
        w = width/N;
        h = height/N;
        x_grid = max(1,ceil((F1(1,:)-fmin_j)/w));
        y_grid = max(1,ceil((F1(2,:)-fmin_i)/h));
        newD = zeros(N*N, size(F1,2));
        newD(sub2ind(size(newD), (x_grid-1)*N + y_grid, 1:size(newD,2))) = 1;
        D1 = [D1; newD];
        %x = (F1(1,:)-fmin_j) / max(1,width);
        %y = (F1(2,:)-fmin_i) / max(1,height);
        %D1 = [D1; (x+y); (x-y); (x.*y); sqrt(x).*y; sqrt(y).*x];
    end
    
    if(scale)
        scale_1 = (F1(end,:)*8)/height;
        scale_2 = (F1(end,:)*8)/width;
        D1 = [D1; scale_1; scale_2];
    end

    if(shape)
        if(~isempty(F1))            
            %centroid = regionprops(masks(:,:,i), 'Centroid');
            [B,L] = bwboundaries(mask,'noholes');
            orig_B = single(B{1}');
            shape_feats = get_polar_hist(F1, orig_B, variable_grids, base_scales, N_SHAPE_DIMS);
            D1 = [D1; shape_feats];
        else
            D1 = [D1; zeros(N_SHAPE_DIMS*2,size(D1,2))];
        end
    end
    
    if(internal_shape)
        if 0
            if(~isempty(F1))
                assert(shape); % requires previous computation

                Ied = edge(rgb2gray(I), 'sobel', 0.05);
                Ied = Ied&mask;       
                %sc(Ied)

                [x,y] = find(Ied);
                if(~isempty(x))
                    internal_shape_feats = get_polar_hist(F1, [x';y'], variable_grids, base_scales, N_SHAPE_DIMS);
                else
                    internal_shape_feats = zeros(N_SHAPE_DIMS*2,size(D1,2));
                end

                D1 = [D1; internal_shape_feats];
                %t = tic(); [gradx, grady] = gradient(rgb2gray(im2single(I))); toc(t)
                %Ied = ((grady>0.04 | gradx>0.04) & mask);
                %sc(grad);

                if 0 % debug 
                    sc(sc(I)+sc(mask)); hold on;            
                    plot(orig_B(2,:), orig_B(1,:), 'ob');
                    c = improfile(I,xi,yi,n, 'nearest');
                end
            else
                D1 = [D1; zeros(N_SHAPE_DIMS*2,size(D1,2))];
            end
        else
           %t = tic();
           duh = D_main_feat'*D_main_feat;
           new_desc = hist(duh,2*N_SHAPE_DIMS);
           D1 = [D1; scale_data(new_desc, 'norm_1')];
           %toc(t)
        end
    end
end

function D = get_polar_hist(F1, orig_B, variable_grids, base_scales, N_SHAPE_DIMS)
    % first unique fraction
    if(~variable_grids)
        range = 1:(size(F1,2)/numel(base_scales));
        assert(numel(range)*numel(base_scales) == size(F1,2));
        assert(numel(unique(F1(end, range)))==1);
    else
        range = 1:size(F1,2);
    end
    F_un = F1(:, range);

    %MAX_POINTS = 200; % 1000
    MAX_POINTS = 100; % 1000
    rp = randperm(size(orig_B,2));
    orig_B = orig_B(:, sort(rp(1:min(numel(rp), MAX_POINTS))));

    %plot(orig_B(1,:), orig_B(2,:), 'o');
    p_coords = single(F_un([2 1], :));

    B = reshape(orig_B, [size(orig_B,1) 1 size(orig_B,2)]);
    rp_p_coords = repmat(p_coords, [1 1 size(orig_B,2)]);
    rp_b = repmat(B, [1 size(p_coords,2) 1]);
    displ = rp_p_coords - rp_b;

    at = atan2(displ(2,:,:),displ(1,:,:));
    angles = ((at+pi)*180)/pi;
    angles = squeeze(angles/360);

    [count,bins] = histc(angles',0:1.0/(N_SHAPE_DIMS-1):1);
    count = single(count);
    if(size(count,1)==1)
        % this happens when there's a single point
        count = count';
    end

    dists = squeeze(sqrt(((displ(1,:,:).^2) + (displ(2,:,:).^2))))';

    dists = uint16(dists); % for speed
    max_dist = max(dists(:));

    bins = uint8(bins);
    shape_feats1 = zeros(N_SHAPE_DIMS, size(p_coords,2), 'uint16');
    %shape_feats2 = shape_feats1;
    for j=1:N_SHAPE_DIMS
        new_dists = dists;
        new_dists(bins~=j) = 0;

        shape_feats1(j,:) = max(new_dists);
        %new_dists(bins_not_j) = max_dist;
        %shape_feats2(j,:) = min(new_dists);
    end

    %normalize dists
    % BAD
    %shape_feats = shape_feats./repmat(max(shape_feats), N_SHAPE_DIMS, 1);
    % GOOD
    shape_feats1 = single(shape_feats1)/single(max_dist);
    %shape_feats2 = shape_feats2/max_dist;
    shape_feats3 = scale_data(count, 'norm_1');

    if 0
        close all;
        P_ID = 1;
        plot(count(:,P_ID));
        figure;
        plot(orig_B(2,:), orig_B(1,:), 'ob'); axis equal; hold on;
        plot(p_coords(2,P_ID), p_coords(1,P_ID), 'or');
    end

    % normalize distances
    if(~variable_grids)
        D = repmat([shape_feats1; shape_feats3], 1, numel(base_scales));
    else
        D = [shape_feats1; shape_feats3];
    end
end