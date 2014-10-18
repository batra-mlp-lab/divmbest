function mysave_n_compress(file, varargin) %#ok<INUSL>
	% save multiple variables trick for parfor
	% call mysave_n(file, 'a', a, 'b', b, ..., 'x', x);

	nparams = size(varargin,2);
	assert(mod(nparams,2) == 0);
	
	for i=1:2:nparams
		eval([varargin{i} ' =  varargin{i+1};']);
	end
	
    eval(strcat('save(file', sprintf(',''%s''', varargin{1:2:end}),');'));
end
