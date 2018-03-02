module NeuralNet.MatrixSpec (matrixSpec) where

import Test.Hspec
import Numeric.LinearAlgebra
import NeuralNet.Matrix


matrixSpec :: SpecWith ()
matrixSpec =
  describe "NeuralNet.Matrix" $ do
    describe "sumRows" $
      it "should work correctly" $
        let
          source = fromLists [[0, 1], [2, 3], [4, 5]]
          expected = fromLists [[1], [5], [9]]
        in
          sumRows source `shouldBe` expected
