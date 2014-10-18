function [Feats, dims, scaling, y, dct_scaling] = feat_loading_wrapper_altered(browser, whole_ids, feats, scal_type, power_scaling, cache_file, name, weights, Feats, dims)
  DefaultVal('*Feats', '[]');
  DefaultVal('*dims', '[]');
  DefaultVal('*cache_file', '[]');
  DefaultVal('*n_dct_dims', '[]');
  DefaultVal('*dct_scaling', '[]');
  DefaultVal('*weights', '[]');
  DefaultVal('*power_scaling', 'false');
  DefaultVal('*name', '[]');
  DefaultVal('*lookup_table_encoding', 'false');
  scaling = [];
   
  Feats_provided = false;
  if(~isempty(Feats))
    Feats_provided = true;
  end
  
  if(~isempty(weights))
      disp('WEIGHTS enabled ! You sure ? ');
  end    
        
  %%%% this file is a big mess, should split it and organize it %%%
  if(iscell(cache_file) || ~isempty(cache_file) && (exist([cache_file '_dims.mat'], 'file')) && ~Feats_provided)   
      if(iscell(cache_file))
          array_size(1:2) = [0 0];
          for i=1:numel(cache_file)
              var = load([cache_file{i} '_dims.mat'], 'dims', 'count', 'array_size', 'names', 'y','all_whole_ids', 'imgset'); % dims
              count(i) = var.count;
              array_size(1) = var.array_size(1);
              array_size(2) = array_size(2) + var.array_size(2);
          end          
          Feats = fast_load_large_mult_files(cache_file, count);
          Feats = reshape(Feats, array_size);
          return;
      else          
          load([cache_file '_dims.mat'], 'dims', 'count', 'array_size', 'names', 'y','all_whole_ids', 'imgset'); % dims
          if(any(strcmp(names, name)) || isempty(name))                    
              % just load it
              if(nargout>0)
                Feats = fast_load_large([cache_file '.bin'], count);
                Feats = reshape(Feats, array_size);
              end

              return;
          end
      end        
  end
  
  [srt_whole_id, srt_ids_a] = sort(whole_ids, 'ascend');  
  if(any(srt_whole_id~=whole_ids) && isempty(Feats)) % if you're gonna load it needs to be sorted
      error('whole_ids not sorted!');
  end
  
  % in any other case
  if(~Feats_provided)
    [Feats, dims] = browser.get_whole_feats(whole_ids, feats, scal_type, weights, lookup_table_encoding, n_dct_dims);
    if(power_scaling)
        Feats = squash_features(Feats, 'power');        
    end
  end      
  
  if(~isempty(cache_file))
      t = tic();      
      if(exist([cache_file '_dims.mat'], 'file')  && ~Feats_provided)          
          load([cache_file '_dims.mat'], 'dims', 'count', 'array_size', 'names', 'y', 'all_whole_ids', 'imgset'); % dims
          cont = true;
          count = fast_save_large([cache_file '.bin'], Feats, cont);
          assert(count==numel(Feats));
          array_size = [array_size(1) array_size(2)+size(Feats,2)];
          count = prod(array_size);
      else
          count = fast_save_large([cache_file '.bin'], Feats);          
          array_size = size(Feats);
      end
              
      if(exist('names', 'var'))
          names = cat(1,names, name);
      else
          names = {name};
      end
      
      new_imgset = browser.imgset;
      if(~strcmp(browser.imgset(1:4), 'test'))
          new_y = single(browser.get_overlaps_wholes(whole_ids));
          if(~exist('y', 'var'))
              y = {new_y};
          else
              y = cat(1, y, new_y);
          end
      else
          y = [];
      end
      
      if(~exist('all_whole_ids', 'var'))
        all_whole_ids = {whole_ids};
      else
          all_whole_ids = cat(1, all_whole_ids, whole_ids);
      end          

      if(~exist('imgset', 'var'))
          imgset = {new_imgset};
      else
          imgset = cat(1, imgset, new_imgset);
      end
          
      save([cache_file '_dims.mat'], 'dims', 'count', 'array_size', 'names', 'y', 'all_whole_ids', 'imgset', '-V6');
      t_save = toc(t)
      Feats = [];
  end  
end
