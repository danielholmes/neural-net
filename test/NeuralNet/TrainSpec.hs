module NeuralNet.TrainSpec (trainSpec) where

import Test.Hspec
import Numeric.LinearAlgebra
import NeuralNet.Net
import NeuralNet.Activation
import NeuralNet.Example
import NeuralNet.Layer
import NeuralNet.Train

trainSpec :: SpecWith ()
trainSpec =
  describe "NeuralNet.Train" $ do
    describe "nnBackward" $ do
      it "calculates correctly for simple example" $
        let
          layer1W = [-1.31386475, 0.88462238, 0.88131804, 1.70957306
                    , 0.05003364, -0.40467741, -0.54535995, -1.54647732
                    , 0.98236743, -1.10106763, -1.18504653, -0.2056499]
          layer1B = [1.48614836, 0.23671627, -1.02378514]
          layer2W = [-1.02387576, 1.12397796, -0.13191423]
          layer2B = [-1.62328545]
          nn = buildNNFromList (4, [LayerDefinition ReLU 3, LayerDefinition Sigmoid 1]) (layer1W ++ layer1B ++ layer2W ++ layer2B)
          examples = createExampleSet [ ([0.09649747, -0.2773882, -0.08274148, -0.04381817], 1)
                                      , ([-1.8634927, -0.35475898, -0.62700068, -0.47721803], 0)]

          inputs = exampleSetX examples
          layer1As = fromLists [[1.97611078, -1.24412333], [-0.62641691, -0.80376609], [-2.41908317, -0.92379202]]
          layer1Zs = fromLists [[-0.7129932, 0.62524497], [-0.16051336, -0.76883635], [-0.23003072, 0.74505627]]
          layer2As = fromLists [[1.78862847, 0.43650985]]
          layer2Zs = fromLists [[0.64667545, -0.35627076]]
          steps = [createInputForwardPropStep inputs, createForwardPropStep layer1As layer1Zs, createForwardPropStep layer2As layer2Zs]

          dW1 = fromLists [ [0.4101000190122487, 7.807203346853248e-2, 0.13798443685274048, 0.10502167417988162]
                          , [0, 0, 0, 0]
                          , [5.2836516249770524e-2, 1.0058654166727896e-2, 1.77776556985907e-2, 1.3530795262454464e-2]]
          db1 = fromLists [[-0.22007063350033446], [0], [-2.835348711039787e-2]]
          dW2 = fromLists [[-0.39202432174003965, -0.13325854898009654, -4.601088848081533e-2]]
          db2 = fromLists [[0.15187860742650333]]
        in
          nnBackward nn steps examples `shouldBe` [(dW1, db1), (dW2, db2)]

    describe "updateNNParams" $ do
      it "calculates correctly for logreg case" $
        let
          nn = buildNNFromList (2, [LayerDefinition ReLU 1]) [1, 2, 3]
          grads = [(fromLists [[10, 100]], fromLists [[1000]])]
          learningRate = 0.1
          expected = buildNNFromList (2, [LayerDefinition ReLU 1]) [0, -8, -97]
        in
          updateNNParams nn grads learningRate `shouldBe` expected
