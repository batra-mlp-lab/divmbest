function count = fast_save_large(filename, variab, cont)    
    DefaultVal('*cont', 'false')
    
    if(cont)
        fid = fopen(filename, 'a');
    else        
        fid = fopen(filename, 'w');
    end
    count = fwrite(fid, variab, 'single');
    fclose(fid);
end