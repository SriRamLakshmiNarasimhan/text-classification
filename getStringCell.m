function dicList  = getStringCell(filename)
	
  n = readFileLength(filename);
  fid = fopen(filename);
  vocablist = cell(n, 1); 
  for i=1:n
       dicList{i, 1} = fgetl(fid);
  endfor
  fclose(fid); 
	
  endfunction