function Feats = project_pca(Feats, coeff)
    counter = 1;
    new_Feats = [];
    cs = cumsum(cellfun(@(a) size(a,1), coeff));
    for i=1:numel(coeff)
        range = counter:cs(i);
        new_Feats = [new_Feats Feats(range, :)'*coeff{i}];
        counter = counter + size(coeff{i},1);
    end
    Feats = new_Feats';
end
