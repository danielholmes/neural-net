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
          as = nnForward nn [1, 1, 1]
          examples = createExampleSet [([1, 2, 3], 0), ([4, 5, 6], 1)]
          dw = zero 4 4
          db = zero 4 1
        in nnBackward nn as examples `shouldBe` [(dw, db)]
