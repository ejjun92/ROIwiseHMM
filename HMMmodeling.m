function [ TrainfeatureVector , TestfeatureVector ] = HMMmodeling( Data, trainIdx, testIdx, numState, numMixture, numDim )

% [ numROI, numScan, numSubject ] = size( Data );
% numState, numMixture, numDim for HMM modeling 

Data = gaussian_normalize( Data ); % Subject-wise normalization
numSubject = size( Data, 1 );
numScan = size( Data, 2 );
numROI = size( Data, 3 );


%% ROI-wise HMM modeling
M = numMixture;  % number of mixtures
O = numDim;  % number of coefficients in a vector
T = numScan;   % number of vectors in a sequence

for i = 1 : 1
    for Q = 1 : numState         
        nex = numSubject;  % number of sequences
        
        data = zeros( O,T,nex );
        permutedData = permute( Data, [3,2,1]);
        data(O,:,:) = squeeze(permutedData(i,:,:));
        
        % state topology : 'ergodic'
        prior0 = normalise(rand(Q,1));
        transmat0 = mk_stochastic(rand(Q, Q));
        
        [mu0, Sigma0] = mixgauss_init( Q*M, reshape(data, [O T*nex]), 'full' );
        mu0 = reshape( mu0, [O Q M] );
        Sigma0 = reshape( Sigma0, [O O Q M] );
        mixmat0 = ones(Q,1);
        
        % training HMM
        [hmmModel(i).state(Q).LL, hmmModel(i).state(Q).prior,...
            hmmModel(i).state(Q).transmat, hmmModel(i).state(Q).mu,...
            hmmModel(i).state(Q).Sigma, hmmModel(i).state(Q).mixmat] = mhmm_em( data, prior0,...
            transmat0, mu0, Sigma0, mixmat0, 'cov_type', 'full', 'max_iter', 30 );
    end
end

%% Calculate BIC Coefficients
op_state = zeros( 1, numROI );
BIC = zeros( numROI, numState );
for i = 1 : 1
    for j = 1 : numState
        LLF(i,j) = hmmModel(i).state(j).LL(end);
        numParam = j + j^2 + (2*numDim)*numMixture*j + numMixture*j;
        [~, BIC(i,j)] = aicbic(LLF(i,j), numParam, numSubject);
    end
    
    op_state(i) = find(BIC(i,:)==min(BIC(i,:)));  % optimal state for each ROI
end


%% feature vector
loglik = zeros(numROI, numSubject);
for i=1:1
    opState = op_state(i);
    
    for j=1:numSubject
        [loglik(i,j), errors] = mhmm_logprob(Data(j,:,i), hmmModel(i).state(opState).prior, ...
            hmmModel(i).state(opState).transmat, hmmModel(i).state(opState).mu, ...
            hmmModel(i).state(opState).Sigma, hmmModel(i).state(opState).mixmat);
    end
end

TrainfeatureVector = loglik( :, trainIdx );
TestfeatureVector = loglik( :, testIdx );