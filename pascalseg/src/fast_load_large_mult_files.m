function var = fast_load_large_mult_files(filenames, count)  
    global_counter = 1;
    %var = cell(numel(filenames),1);
    var = zeros(sum(count),1, 'single');
    for i=1:numel(filenames)
        % break into chunks over 1gb           
        n_chunks = ceil((count(i)*4)/10^9); % *4 for single       
        
        fid = fopen([filenames{i} '.bin'], 'r');                
        left = count(i);
        for k=1:n_chunks
            this_count = min(left, (0.25)*10^9);
            var(global_counter:(global_counter+this_count-1)) = fread(fid, this_count, 'single=>single');
            global_counter = global_counter + this_count;
            left = left - this_count;
        end
        
        fclose(fid);
    end
end
