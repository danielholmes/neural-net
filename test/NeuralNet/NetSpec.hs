module NeuralNet.NetSpec (netSpec) where

import Test.Hspec
import Data.Matrix
import NeuralNet.Net
import NeuralNet.Activation
import NeuralNet.Layer
import NeuralNet.Example
import Control.Exception (evaluate)
import Data.List

matrixSize :: Matrix Double -> (Int, Int)
matrixSize m = (nrows m, ncols m)

allUnique :: (Eq a) => Matrix a -> Bool
allUnique m = length (nub list) == length list
  where list = toList m

createdLayer :: Activation -> (Int, Int) -> NeuronLayer -> Bool
createdLayer a s l = activationEq && wSizeEq && bCorrect && allUnique m
  where
    m = layerW l
    bCorrect = layerB l == fromList (fst s) 1 (replicate (fst s) 0)
    activationEq = layerActivation l == a
    wSizeEq = matrixSize m == s

netSpec :: SpecWith ()
netSpec =
  describe "NeuralNet.Net" $ do
    describe "initNN" $ do
      it "returns correctly for small definition" $
        let
          nn = initNN [1..] (8, [LayerDefinition ReLU 5, LayerDefinition Tanh 1, LayerDefinition Sigmoid 2])
          layers = nnLayers nn
        in do
          layers!!0 `shouldSatisfy` createdLayer ReLU (5, 8)
          layers!!1 `shouldSatisfy` createdLayer Tanh (1, 5)
          layers!!2 `shouldSatisfy` createdLayer Sigmoid (2, 1)

      it "returns correctly for single layer definition" $
        let
          nn = initNN [1..] (6, [LayerDefinition ReLU 5])
          layers = nnLayers nn
        in
          layers!!0 `shouldSatisfy` createdLayer ReLU (5, 6)

      it "throws exception for no layers" $
        evaluate (initNN [1..] (6, [])) `shouldThrow` anyException

      it "throws exception for no inputs" $
        evaluate (initNN [1..] (0, [LayerDefinition ReLU 5])) `shouldThrow` anyException

    describe "buildNNFromList" $ do
      it "creates log reg correctly" $
        let
          nn = buildNNFromList (5, [LayerDefinition ReLU 1]) [1, 2, 3, 4, 5, 6]
          layers = nnLayers nn
        in do
          length layers `shouldBe` 1
          layerW (layers!!0) `shouldBe` fromLists [[1, 2, 3, 4, 5]]
          layerB (layers!!0) `shouldBe` fromLists [[6]]

      it "creates multi output correctly" $
        let
          nn = buildNNFromList (3, [LayerDefinition ReLU 2]) [1, 2, 3, 4, 5, 6, 7, 8]
          layers = nnLayers nn
        in do
          length layers `shouldBe` 1
          layerW (layers!!0) `shouldBe` fromLists [[1, 2, 3], [4, 5, 6]]
          layerB (layers!!0) `shouldBe` fromLists [[7], [8]]

      it "creates multi layer correctly" $
        let
          nn = buildNNFromList (2, [LayerDefinition ReLU 2, LayerDefinition Tanh 1]) [1, 2, 3, 4, 5, 6, 7, 8, 9]
          layers = nnLayers nn
        in do
          length layers `shouldBe` 2
          layerW (layers!!0) `shouldBe` fromLists [[1, 2], [3, 4]]
          layerB (layers!!0) `shouldBe` fromLists [[5], [6]]
          layerW (layers!!1) `shouldBe` fromLists [[7, 8]]
          layerB (layers!!1) `shouldBe` fromLists [[9]]

      it "throw error for incorrect list size" $
        evaluate (buildNNFromList (5, [LayerDefinition ReLU 1]) [1, 2]) `shouldThrow` anyException


    describe "isExampleSetCompatibleWithNN" $ do
      it "calculates correctly for compatible" $
        let
          nn = buildNNFromList (3, [LayerDefinition ReLU 1]) [1, 2, 3, 4]
          examples = createExampleSet [([1, 2, 3], 0), ([4, 5, 6], 1)]
        in isExampleSetCompatibleWithNN examples nn `shouldBe` True

      it "calculates correctly for incompatible" $
        let
          nn = buildNNFromList (3, [LayerDefinition ReLU 1]) [1, 2, 3, 4]
          examples = createExampleSet [([1, 2], 0), ([4, 5], 1)]
        in isExampleSetCompatibleWithNN examples nn `shouldBe` False

    describe "nnForward" $ do
      -- Not working, can't figure out why. When change to shouldBe [] it does throw an error
--      it "throws errors when incorrect num inputs provided" $
--        let nn = buildNNFromList (3, [LayerDefinition ReLU 1]) [1, 2, 3, 4]
--        in evaluate (nnForward nn [1, 1]) `shouldThrow` anyErrorCall

      it "calculates correctly for logreg" $
        let nn = buildNNFromList (3, [LayerDefinition ReLU 1]) [1, 2, 3, 4]
        in nnForward nn [1, 1, 2] `shouldBe` [13.0]

      it "calculates correctly for multi layer" $
        let
          def = (3, [LayerDefinition Sigmoid 2, LayerDefinition ReLU 1])
          nums = [1, -1, -1
            , 1, 1, -1
            , 1, -1
            , 2, 3,
            0]
          nn = buildNNFromList def nums
        in
          nnForward nn [1, 1, 1] `shouldBe` [2.5]

    describe "nnForwardSet" $ do
      it "calculates correctly for logreg" $
        let
          nn = buildNNFromList (3, [LayerDefinition ReLU 1]) [1, 2, 3, 4]
          examples = createExampleSet [([1, 1, 2], 0)]
          resultAs = map forwardPropA (nnForwardSet nn examples)
        in resultAs `shouldBe` [fromLists [[1], [1], [2]], matrix 1 1 (const 13.0)]

      it "calculates correctly for multiple examples, multiple output" $
        let
          nn = buildNNFromList (3, [LayerDefinition ReLU 2, LayerDefinition ReLU 1]) [1, 4, 2, 5, 3, 6, 7, 8, 1, 1, 0]
          examples = createExampleSet [([1, 1, 2], 0), ([1, 1, 0], 0)]
          resultAs = map forwardPropA (nnForwardSet nn examples)
        in resultAs `shouldBe` [fromLists [[1, 1], [1, 1], [2, 0]], fromLists [[16, 12], [28, 16]], fromLists [[44, 28]]]
