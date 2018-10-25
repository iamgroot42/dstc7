# dstc7
DSTC7 challenge

Basic codebase from : https://github.com/IBM/dstc7-noesis
Please refer https://github.com/IBM/dstc7-noesis/tree/master/noesis-tf for requirements and training data format.

## Additional features

* BiDirectional LSTM (arg: --bidirectional)
* Mean and Max-pool based feature encoding (arg: --feature_type=mean/max)
* CNN based max-pooling of features (arg: --feature_type cnn)
* Attention (arg: --attention)
* Fast GRNN (arg: --fastgrnn)
* Using C-DSSM based sentence (pass on in input data) embeddings (arg: --dssm)
* Custom Word2Vec embedding learning on dataset (arg: --factorization)
* Extend training set by taking 7th turn onwards as positive utterances


For custom word2vec training, [read here](https://github.com/iamgroot42/dstc7/tree/master/CSharp/Word2Vec)
