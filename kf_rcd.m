function varargout = kf_rcd (varargin)
% kf_rcd. A memory- or disk-based stack for kfgs data.
%
% r = kf_rcd('init', medium, base_filename)
%   Initialize the stack.
%   medium is either 'mem' for storage in main memory or 'disk' for hard disk
% storage.
%   base_filename is required for medium = 'disk'. It is the base filename of
% the files kf_rcd must save. Example: '/scratch/kf/test'.
  [varargout{1:nargout}] = feval(varargin{:});
end
  
function r = init (medium, fn)
  switch (lower(medium(1)))
   case 'm' % mem
    r.medium = 'm';
    r.stack = {};
    
   case 'd' % disk
    if (nargin ~= 2)
      error('rcd = kf_rcd(''init'', ''disk'', base_filename);');
    end
    r.medium = 'd';
    r.fn = fn;
    r.nbrs = [];
    
   otherwise
    error('medium = ''mem'' or ''disk''.');
  end
end
  
function r = push (r, varargin)
  switch (r.medium)
   case 'm'
    r.stack{end+1} = varargin;
    
   case 'd'
    r.nbrs(end+1) = length(varargin);
    for (i = 1:r.nbrs(end))
      fn = GetFn(r.fn, length(r.nbrs), i);
      fid = fopen(fn, 'w');
      [m n] = size(varargin{i});
      fwrite(fid, [m n], 'int32');
      fwrite(fid, varargin{i}(:), 'double');
      fclose(fid);
    end
    
  end
end
  
function [r varargout] = pop (r)
% Return top-most and delete.
  switch (r.medium)
   case 'm'
    varargout = r.stack{end};
    r.stack(end) = [];
    
   case 'd'
    for (i = 1:r.nbrs(end))
      [varargout{i} fn] = Read(r.fn, length(r.nbrs), i);
      delete(fn);
    end
    r.nbrs(end) = [];
  end
end
  
function [r varargout] = peak (r)
% Return top-most.
  switch (r.medium)
   case 'm'
    varargout = r.stack{end};
    
   case 'd'
    for (i = 1:r.nbrs(end))
      varargout{i} = Read(r.fn, length(r.nbrs), i);
    end
  end
end
  
function r = discard (r)
% Discard top-most without looking.
  switch (r.medium)
   case 'm'
    r.stack(end) = [];
    
   case 'd'
    for (i = 1:r.nbrs(end))
      fn = GetFn(r.fn, length(r.nbrs), i);
      delete(fn);
    end
    r.nbrs(end) = [];
  end  
end
  
function [A fn] = Read (fn, nbr, idx)
  fn = GetFn(fn, nbr, idx);
  fid = fopen(fn, 'r');
  sz = fread(fid, 2, 'int32');
  A = fread(fid, sz', 'double');
  fclose(fid);
end
  
function fn = GetFn (basefn, nbr, idx)
  fn = sprintf('%s_%06d_%06d.dat', basefn, nbr, idx);
end