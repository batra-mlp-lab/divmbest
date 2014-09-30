function boxes = detect_fast_divmbest(im, model, thresh,nummodes,one_scale,lambda, type)
% boxes = detect(im, model, thresh)
% Detect objects in input using a model and a score threshold.
% Higher threshold leads to fewer detections.
%
% The function returns a matrix with one row per detected object.  The
% last column of each row gives the score of the detection.  The
% column before last specifies the component used for the detection.
% Each set of the first 4 columns specify the bounding box for a part

% This function builds on detect_mmodes_v2 written by AGR. v2 suppresses
% the one configuration with the highest score (for each mode).
%
% Even though the code is written for multiple model.components, I (Dhruv)
% am pretty sure the logic implemented only works for 1 component, which is
% what Deva's model has. Not fixing right now. 
%
% Dhruv Batra (dbatra -at- vt.edu)
% Created: 05/22/2013

if nargin < 5
    one_scale = 0; % AGR: force detection to single scale when one_scale ~= 0
end
if nargin < 6
    lambda = -1e10;
end

% Compute the feature pyramid and prepare filter
pyra     = featpyramid(im,model);
interval = model.interval;
if one_scale ~= 0
    levels = one_scale;
else
    levels = 1:length(pyra.feat);
end
numlevels = length(pyra.feat);


% Cache various statistics derived from model
[components,filters,resp] = modelcomponents(model,pyra);

%boxes = zeros(10000,length(components{1})*4+2);
%cnt   = 0;
% AGR: one boxes array per mode each will hold a MAP for each level/scale
boxes_rowsize = length(components{1})*4+2;
boxes = zeros(numlevels,boxes_rowsize,nummodes);
boxes(:,end,1:end) = -inf; % set scores to -inf

% AGR: we'll have 3 copies of the model:
parts_unmod = cell(numlevels,1); % unmodified to recover orignal scores
parts_smode = cell(numlevels,1); % modified potentials for mmodes
parts_wmsgs = cell(numlevels,1); % temporary for inference/traceback


% Iterate over scales and components,
for rlevel = levels,
    for c  = 1:length(model.components),
        parts    = components{c};
        numparts = length(parts);
        
        % Local scores
        for k = 1:numparts,
            f     = parts(k).filterid;
            level = rlevel-parts(k).scale*interval;
            if isempty(resp{level}),
                resp{level} = fconv(pyra.feat{level},filters,1,length(filters));
            end
            for fi = 1:length(f)
                parts(k).score(:,:,fi) = resp{level}{f(fi)};
            end
            parts(k).level = level;
        end

        % AGR: save copy before message passing
        parts_smode{rlevel} = parts;
        parts_unmod{rlevel} = parts;
        
        % Walk from leaves to root of tree, passing message to parent
        for k = numparts:-1:2,
            par = parts(k).parent;
            [msg,parts(k).Ix,parts(k).Iy,parts(k).Ik] = passmsg(parts(k),parts(par));
            parts(par).score = parts(par).score + msg;
        end
        
        % Add bias to root score
        parts(1).score = parts(1).score + parts(1).b;
        parts_unmod{rlevel}(1).score = parts_unmod{rlevel}(1).score ...
            + parts_unmod{rlevel}(1).b;

        % AGR: traceback from max
        [rscore Ik]    = max(parts(1).score,[],3);
        [ss ys] = max(rscore);
        [s x]   = max(ss);
        y       = ys(x);
        t       = Ik(y,x);
        parts(1).XYT = [x y t];
        [box score_unmod] = backtrack(x,y,t,parts,pyra,parts_unmod{rlevel});
        if abs(rscore(y,x) - score_unmod) > 0.0001
            fprintf('ERROR: rscore(y,x)=%f ~= score_unmod=%f\n',rscore(y,x),score_unmod);
        end
        
        % AGR: save copy after message passing (used when suppressing mode)
        parts_wmsgs{rlevel} = parts;
        
        if rscore(y,x) >= thresh
            boxes(rlevel,:,1) = [box c rscore(y,x)];
        else
            parts_smode{rlevel} = [];
            parts_wmsgs{rlevel} = [];
            boxes(rlevel,end,1) = -inf;
        end
    end
end

% AGR: suppress (overall) map
[score_map level_map] = max(boxes(:,end,1));
if score_map > -inf
    parts = parts_wmsgs{level_map};
    x = parts(1).XYT(1);
    y = parts(1).XYT(2);
    t = parts(1).XYT(3);
    parts_smode{level_map} = ...
        suppress_mode(parts_smode{level_map},parts,x,y,t,lambda,type);
else
    boxes(:,end,1:end) = -inf;
    return;
end

% AGR: so far we have the MAP, now find next best MAPs

% Iterate over modes
for mode = 2:nummodes
    % Iterate over scales and components,
    for rlevel = levels,
        for c  = 1:length(model.components),
            parts    = parts_smode{rlevel}; % components{c};
            if isempty(parts)
                boxes(rlevel,end,mode) = -inf;
                continue;
            end
            numparts = length(parts);
            
            % Walk from leaves to root of tree, passing message to parent
            for k = numparts:-1:2,
                par = parts(k).parent;
                [msg,parts(k).Ix,parts(k).Iy,parts(k).Ik] = passmsg(parts(k),parts(par));
                parts(par).score = parts(par).score + msg;
            end
            
            % Add bias to root score
            parts(1).score = parts(1).score + parts(1).b;
            [rscore Ik]    = max(parts(1).score,[],3);
            
            % AGR: traceback from max
            [ss ys] = max(rscore);
            [s x]   = max(ss);
            y       = ys(x);
            t = Ik(y,x);
            parts(1).XYT = [x y t];
            [box score_unmod] = backtrack(x,y,t,parts,pyra,parts_unmod{rlevel});
            
            % AGR: save copy after message passing (used when suppressing mode)
            parts_wmsgs{rlevel} = parts;
            
            if score_unmod >= thresh
                boxes(rlevel,:,mode) = [box c score_unmod];
            else
                parts_smode{rlevel} = [];
                parts_wmsgs{rlevel} = [];
                boxes(rlevel,end,mode) = -inf;
            end
        end
    end
    % AGR: suppress (overall) map
    [score_map level_map] = max(boxes(:,end,mode));
    if score_map > -inf
        parts = parts_wmsgs{level_map};
        x = parts(1).XYT(1);
        y = parts(1).XYT(2);
        t = parts(1).XYT(3);
        parts_smode{level_map} = ...
            suppress_mode(parts_smode{level_map},parts,x,y,t,lambda,type);
    else
        boxes(:,end,mode:end) = -inf;
        return;
    end
end

% Cache various statistics from the model data structure for later use
function [components,filters,resp] = modelcomponents(model,pyra)
components = cell(length(model.components),1);
for c = 1:length(model.components),
    for k = 1:length(model.components{c}),
        p = model.components{c}(k);
        [p.w,p.defI,p.starty,p.startx,p.step,p.level,p.Ix,p.Iy] = deal([]);
        [p.scale,p.level,p.Ix,p.Iy] = deal(0);
        
        % store the scale of each part relative to the component root
        par = p.parent;
        assert(par < k);
        p.b = [model.bias(p.biasid).w];
        p.b = reshape(p.b,[1 size(p.biasid)]);
        p.biasI = [model.bias(p.biasid).i];
        p.biasI = reshape(p.biasI,size(p.biasid));
        p.sizx  = zeros(length(p.filterid),1);
        p.sizy  = zeros(length(p.filterid),1);
        
        for f = 1:length(p.filterid)
            x = model.filters(p.filterid(f));
            [p.sizy(f) p.sizx(f) foo] = size(x.w);
            %         p.filterI(f) = x.i;
        end
        for f = 1:length(p.defid)
            x = model.defs(p.defid(f));
            p.w(:,f)  = x.w';
            p.defI(f) = x.i;
            ax  = x.anchor(1);
            ay  = x.anchor(2);
            ds  = x.anchor(3);
            p.scale = ds + components{c}(par).scale;
            % amount of (virtual) padding to hallucinate
            step     = 2^ds;
            virtpady = (step-1)*pyra.pady;
            virtpadx = (step-1)*pyra.padx;
            % starting points (simulates additional padding at finer scales)
            p.starty(f) = ay-virtpady;
            p.startx(f) = ax-virtpadx;
            p.step   = step;
        end
        components{c}(k) = p;
    end
end

resp    = cell(length(pyra.feat),1);
filters = cell(length(model.filters),1);
for i = 1:length(filters),
    filters{i} = model.filters(i).w;
end

% Given a 2D array of filter scores 'child',
% (1) Apply distance transform
% (2) Shift by anchor position of part wrt parent
% (3) Downsample if necessary
function [score,Ix,Iy,Ik] = passmsg(child,parent)
INF = 1e10;
K   = length(child.filterid);
Ny  = size(parent.score,1);
Nx  = size(parent.score,2);
[Ix0,Iy0,score0] = deal(zeros([Ny Nx K]));

for k = 1:K
    [score0(:,:,k),Ix0(:,:,k),Iy0(:,:,k)] = shiftdt(child.score(:,:,k), child.w(1,k), child.w(2,k), child.w(3,k), child.w(4,k),child.startx(k),child.starty(k),Nx,Ny,child.step);
end

% At each parent location, for each parent mixture 1:L, compute best child mixture 1:K
L  = length(parent.filterid);
N  = Nx*Ny;
i0 = reshape(1:N,Ny,Nx);
[score,Ix,Iy,Ix,Ik] = deal(zeros(Ny,Nx,L));
for l = 1:L
    b = child.b(1,l,:);
    [score(:,:,l),I] = max(bsxfun(@plus,score0,b),[],3);
    i = i0 + N*(I-1);
    Ix(:,:,l)    = Ix0(i);
    Iy(:,:,l)    = Iy0(i);
    Ik(:,:,l)    = I;
end

% Backtrack through DP msgs to collect ptrs to part locations
%function box = backtrack(x,y,mix,parts,pyra)
function [box score_unmod] = backtrack(x,y,mix,parts,pyra,parts_unmod)

numx     = length(x);
numparts = length(parts);

xptr = zeros(numx,numparts);
yptr = zeros(numx,numparts);
mptr = zeros(numx,numparts);
box  = zeros(numx,4,numparts);

%DB: this code assumes backtracking from a single location. Fix later
assert(isscalar(x) == 1);
score_unmod = 0;
if exist('parts_unmod','var')
    score_unmod = parts_unmod(1).score(y,x,mix);
end

for k = 1:numparts,
    p   = parts(k);
    if k == 1,
        xptr(:,k) = x;
        yptr(:,k) = y;
        mptr(:,k) = mix;
    else
        % I = sub2ind(size(p.Ix),yptr(:,par),xptr(:,par),mptr(:,par));
        par = p.parent;
        [h,w,foo] = size(p.Ix);
        I   = (mptr(:,par)-1)*h*w + (xptr(:,par)-1)*h + yptr(:,par);
        xptr(:,k) = p.Ix(I);
        yptr(:,k) = p.Iy(I);
        mptr(:,k) = p.Ik(I);
    end
    scale = pyra.scale(p.level);
    x1 = (xptr(:,k) - 1 - pyra.padx)*scale+1;
    y1 = (yptr(:,k) - 1 - pyra.pady)*scale+1;
    x2 = x1 + p.sizx(mptr(:,k))*scale - 1;
    y2 = y1 + p.sizy(mptr(:,k))*scale - 1;
    box(:,:,k) = [x1 y1 x2 y2];
    
    %DB: this code assumes backtracking from a single location. Fix later
    if k>1 && exist('parts_unmod','var')
        %keyboard
        score_unmod = score_unmod + parts_unmod(k).score(yptr(1,k),xptr(1,k),mptr(1,k));
        score_unmod = score_unmod ...
            + parts_unmod(k).w(:,mptr(1,k))'*defvector(xptr(1,par),yptr(1,par),xptr(1,k),yptr(1,k),mptr(1,k),parts_unmod(k));
        score_unmod = score_unmod + parts_unmod(k).b(1,mptr(1,par),mptr(1,k));
    end
end
box = reshape(box,numx,4*numparts);

% Compute the deformation feature given parent locations,
% child locations, and the child part
function res = defvector(px,py,x,y,mix,part)
probex = ( (px-1)*part.step + part.startx(mix) );
probey = ( (py-1)*part.step + part.starty(mix) );
dx  = probex - x;
dy  = probey - y;
res = -[dx^2 dx dy^2 dy]';
