module NeuralNet.ActivationSpec (activationSpec) where

import Test.Hspec
import Data.Matrix
import NeuralNet.Activation


activationSpec :: SpecWith ()
activationSpec =
  describe "NeuralNet.Activation" $
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
