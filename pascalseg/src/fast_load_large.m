function var = fast_load_large(filename, count)    
    fid = fopen(filename, 'r');
    var = fread(fid, count, 'single=>single');
    fclose(fid);
end
