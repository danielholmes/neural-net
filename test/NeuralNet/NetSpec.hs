module NeuralNet.NetSpec (netSpec) where

import Test.Hspec
import Data.Matrix
import NeuralNet.Net
import NeuralNet.Activation
import NeuralNet.Layer
import Control.Exception (evaluate)
import Data.List
import System.Random

matrixSize :: Matrix Double -> (Int, Int)
matrixSize m = (nrows m, ncols m)

allUnique :: (Eq a) => Matrix a -> Bool
allUnique m = length (nub list) == length list
  where list = toList m

createdLayer :: Activation -> (Int, Int) -> NeuronLayer -> Bool
createdLayer a s l = activationEq && wSizeEq && bCorrect && allUnique m
  where
    m = layerW l
    bCorrect = layerB l == replicate (snd s) 0
    activationEq = layerActivation l == a
    wSizeEq = matrixSize m == s

netSpec :: StdGen -> SpecWith ()
netSpec g = do
  describe "initNN" $ do
    it "returns correctly for small definition" $
      let
        nn = initNN g (8, [LayerDefinition ReLU 5, LayerDefinition Tanh 1, LayerDefinition Sigmoid 2])
        layers = nnLayers nn
      in do
        layers!!0 `shouldSatisfy` createdLayer ReLU (8, 5)
        layers!!1 `shouldSatisfy` createdLayer Tanh (5, 1)
        layers!!2 `shouldSatisfy` createdLayer Sigmoid (1, 2)

    it "returns correctly for single layer definition" $
      let
        nn = initNN g (6, [LayerDefinition ReLU 5])
        layers = nnLayers nn
      in
        layers!!0 `shouldSatisfy` createdLayer ReLU (6, 5)

    it "throws exception for no layers" $
      evaluate (initNN g (6, [])) `shouldThrow` anyException

    it "throws exception for no inputs" $
      evaluate (initNN g (0, [LayerDefinition ReLU 5])) `shouldThrow` anyException

  describe "buildNNFromList" $ do
    it "creates log reg correctly" $
      let
        nn = buildNNFromList (5, [LayerDefinition ReLU 1]) [1, 2, 3, 4, 5, 6]
        layers = nnLayers nn
      in do
        length layers `shouldBe` 1
        layerW (layers!!0) `shouldBe` fromList 5 1 [1, 2, 3, 4, 5]
        layerB (layers!!0) `shouldBe` [6]

    it "creates multi output correctly" $
      let
        nn = buildNNFromList (3, [LayerDefinition ReLU 2]) [1, 2, 3, 4, 5, 6, 7, 8]
        layers = nnLayers nn
      in do
        length layers `shouldBe` 1
        layerW (layers!!0) `shouldBe` fromList 3 2 [1, 2, 3, 4, 5, 6]
        layerB (layers!!0) `shouldBe` [7, 8]

    it "throw error for incorrect list size" $
      evaluate (buildNNFromList (5, [LayerDefinition ReLU 1]) [1, 2]) `shouldThrow` anyException

  describe "nnForward" $ do
    it "throws errors when incorrect num inputs provided" $
      let nn = buildNNFromList (3, [LayerDefinition ReLU 1]) [1, 2, 3, 4]
      in evaluate (nnForward nn [1, 1]) `shouldThrow` anyException

    it "calculates correctly for logreg" $
      let nn = buildNNFromList (3, [LayerDefinition ReLU 1]) [1, 2, 3, 4]
      in (nnForward nn [1, 1, 2]) `shouldBe` [1.0]
