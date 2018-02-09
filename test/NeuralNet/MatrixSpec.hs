module NeuralNet.MatrixSpec (matrixSpec) where

import Test.Hspec
import Data.Matrix
import NeuralNet.Matrix


matrixSpec :: SpecWith ()
matrixSpec =
  describe "NeuralNet.Matrix" $
    describe "mapMatrix" $ do
      it "should apply to all elements" $
        let
          source :: Matrix Int
          source = fromList 3 3 [0, 1, 2, 3, 4, 5, 6, 7, 8]
          expected :: Matrix Int
          expected = fromList 3 3 [0, 10, 20, 30, 40, 50, 60, 70, 80]
          result :: Matrix Int
          result = mapMatrix (* 10) source
        in
          result `shouldBe` expected
