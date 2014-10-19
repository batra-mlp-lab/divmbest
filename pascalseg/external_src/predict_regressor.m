function pred = predict_regressor(Feats, w, noBias)
  DefaultVal('*noBias', 'false');
  assert(~any(isnan(w(:))));

  if(noBias)
    pred = w' * Feats;
  else
    pred = w' * [ones(1,size(Feats,2)); Feats];
  end
  % inverse logit transformation
  %pred = 1 - 1./(exp(pred) + 1);
end
