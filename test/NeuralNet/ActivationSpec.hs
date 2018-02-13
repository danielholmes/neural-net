module NeuralNet.ActivationSpec (activationSpec) where

import Test.Hspec
import Data.Matrix
import NeuralNet.Activation


activationSpec :: SpecWith ()
activationSpec =
  describe "NeuralNet.Activation" $ do
    describe "forward" $ do
      it "should correctly work with ReLU" $
        let
          source = setElem (-1) (1, 2) (identity 4)
          expected = setElem 0 (1, 2) (identity 4)
        in forward ReLU source `shouldBe` expected

      it "should correctly work with Sigmoid" $
        let
          source = zero 4 4
          expected = matrix 4 4 (const 0.5)
        in forward Sigmoid source `shouldBe` expected

      it "should correctly work with Tanh" $
        let
          source = zero 4 4
          expected = matrix 4 4 (const 0)
        in forward Tanh source `shouldBe` expected

    describe "backward" $ do
      it "should correctly work with ReLU" $
        let
          source = setElem (-1) (1, 2) (identity 4)
          expected = setElem 0 (1, 2) (matrix 4 4 (const 1))
        in backward ReLU source `shouldBe` expected

      it "should correctly work with Tanh" $
        let
          source = diagonalList 3 0 [1..]
          expected = fromLists [[1, 0, 0], [0, 4, 0], [0, 0, 9]]
        in backward Tanh source `shouldBe` expected

      it "should correctly work with Sigmoid" $
        let
          source = fromLists [[1, 2], [3, 4]]
          expected = fromLists [[0, -2], [-6, -12]]
        in backward Sigmoid source `shouldBe` expected
