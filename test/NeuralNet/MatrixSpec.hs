module NeuralNet.MatrixSpec (matrixSpec) where

import Test.Hspec
import Data.Matrix
import NeuralNet.Matrix


matrixSpec :: SpecWith ()
matrixSpec =
  describe "NeuralNet.Matrix" $ do
    describe "mapMatrix" $ do
      it "should apply to all elements" $
        let
          source = fromLists [[0, 1], [2, 3], [4, 5]] :: Matrix Int
          expected = fromLists [[0, 10], [20, 30], [40, 50]]
        in
          mapMatrix (* 10) source `shouldBe` expected

      it "should apply to all elements - single" $
        let
          source = fromLists [[0, 1], [2, 3], [4, 5]] :: Matrix Int
          expected = fromLists [[0, 10], [20, 30], [40, 50]]
        in
          mapMatrix (* 10) source `shouldBe` expected

    describe "sumRows" $
      it "should work correctly" $
        let
          source = fromLists [[0, 1], [2, 3], [4, 5]] :: Matrix Int
          expected = fromLists [[1], [5], [9]]
        in
          sumRows source `shouldBe` expected
