function normedData = gaussian_normalize( Data )

[ numROI, numScan, numSubject ] = size( Data );

% Subject-wise normalization
normedData = permute( Data, [3,2,1] );
normedData = reshape( normedData, [numSubject, numScan*numROI] );
center = mean( squeeze( mean( normedData, 2 ) ), 2);
denom = sqrt(var(normedData, [], 2));
normedData = bsxfun( @minus, normedData, center );
normedData = bsxfun( @rdivide, normedData, denom );
normedData = reshape( normedData, [numSubject, numScan, numROI] );

end