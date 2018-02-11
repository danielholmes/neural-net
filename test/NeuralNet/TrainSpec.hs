module NeuralNet.TrainSpec (trainSpec) where

import Test.Hspec
import Data.Matrix
import NeuralNet.Net
import NeuralNet.Activation
import NeuralNet.Example
import NeuralNet.Train

trainSpec :: SpecWith ()
trainSpec =
  describe "NeuralNet.Train" $ do
    describe "nnBackward" $ do
      it "calculates correctly for logreg" $
        let
          nn = buildNNFromList (3, [LayerDefinition ReLU 1]) [1, 2, 3, 4]
          examples = createExampleSet [([1, 2, 3], 4), ([0, 1, 2], 1)]
          as = nnForwardSet nn examples
          dw = fromList 1 3 [7.0, 19.5, 32.0]
          db = fromList 1 2 [7.0, 5.5]
        in nnBackward nn as examples `shouldBe` [(dw, db)]

      it "calculates correctly for multi layer" $
        let
          nn = buildNNFromList (3, [LayerDefinition ReLU 2, LayerDefinition ReLU 1]) [1..11]
          examples = createExampleSet [([1, 2, 3], 4), ([0, 1, 2], 1)]
          as = nnForwardSet nn examples
          dw1 = fromList 1 3 [7.0, 19.5, 32.0]
          db1 = fromList 1 2 [7.0, 5.5]
          dw2 = fromList 1 3 [7.0, 19.5, 32.0]
          db2 = fromList 1 2 [7.0, 5.5]
        in nnBackward nn as examples `shouldBe` [(dw1, db1), (dw2, db2)]
