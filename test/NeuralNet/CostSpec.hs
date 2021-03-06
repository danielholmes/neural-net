module NeuralNet.CostSpec (costSpec) where

import Test.Hspec
import Numeric.LinearAlgebra
import NeuralNet.Cost

costSpec :: SpecWith ()
costSpec =
  describe "NeuralNet.Cost" $
    describe "computeCost" $ do
      it "calculates correctly for example" $
        let
          al = fromLists [[0.8, 0.9, 0.4]]
          y = fromLists [[1, 1, 1]]
        in computeCost al y `shouldBe` 0.41493159961539694
