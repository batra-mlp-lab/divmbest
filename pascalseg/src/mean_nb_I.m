function m = mean_nb_I(I, i, j, nb)
    DefaultVal('*nb', '0');
    medI = I;
    if(nb ~= 0)        
        for a=1:size(I,3)
            medI(:,:,a) = medfilt2(I(:,:,a), [nb+1 nb+1]);
        end    
    end    
    
    newI = reshape(medI, size(I,1)*size(I,2), size(I,3))';
    ids = sub2ind(size(I(:,:,1)), i, j);
    
    
    m = newI(:, ids);
end