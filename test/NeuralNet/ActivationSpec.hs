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
        in (forward ReLU source) `shouldBe` expected
