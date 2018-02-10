module NeuralNet.LayerSpec (layerSpec) where

import Test.Hspec
import NeuralNet.Net
import NeuralNet.Activation
import NeuralNet.Example
import NeuralNet.Layer
import Control.Exception (evaluate)
import Data.Matrix


layerSpec :: SpecWith ()
layerSpec =
  describe "NeuralNet.Layer" $ do
    describe "layerForward" $ do
      it "errors on wrong number of inputs" $
        let
          nn = buildNNFromList (3, [LayerDefinition ReLU 1]) [1, 2, 3, 4]
          firstLayer = head (nnLayers nn)
        in evaluate (layerForward firstLayer [1]) `shouldThrow` anyErrorCall

      it "calculates correctly for single output" $
        let
          nn = buildNNFromList (3, [LayerDefinition ReLU 1]) [1, 2, 3, 4]
          firstLayer = head (nnLayers nn)
        in (layerForward firstLayer [1, 1, 2]) `shouldBe` [13.0]

      it "calculates correctly for multiple output" $
        let
          nn = buildNNFromList (3, [LayerDefinition ReLU 2]) [1, 4, 2, 5, 3, 6, 7, 8]
          firstLayer = head (nnLayers nn)
        in (layerForward firstLayer [1, 1, 2]) `shouldBe` [16.0, 29.0]

      it "applies activation" $
        let
          nn = buildNNFromList (3, [LayerDefinition Sigmoid 2]) [1, -1, -1, 1, 1, -1, -1, 1]
          firstLayer = head (nnLayers nn)
        in (layerForward firstLayer [1, 1, 1]) `shouldBe` [0.5, 0.5]

    describe "layerForwardSet" $ do
      it "calculates correctly for multiple output" $
        let
          nn = buildNNFromList (3, [LayerDefinition ReLU 2]) [1, 4, 2, 5, 3, 6, 7, 8]
          firstLayer = head (nnLayers nn)
          examples = exampleSetX (createExampleSet [([1, 1, 2], 0)])
        in (layerForwardSet firstLayer examples) `shouldBe` fromList 2 1 [16.0, 29.0]

      it "calculates correctly for multiple examples, multiple output" $
        let
          nn = buildNNFromList (3, [LayerDefinition ReLU 2]) [1, 4, 2, 5, 3, 6, 7, 8]
          firstLayer = head (nnLayers nn)
          examples = exampleSetX (createExampleSet [([1, 1, 2], 0), ([1, 1, 0], 0)])
        in (layerForwardSet firstLayer examples) `shouldBe` fromList 2 2 [16, 10, 29, 17]

    describe "layerBackward" $ do
      it "calculates correctly for single output" $
        let
          nn = buildNNFromList (3, [LayerDefinition ReLU 1]) [1, 2, 3, 4]
          firstLayer = head (nnLayers nn)
        in (layerForward firstLayer [1, 1, 2]) `shouldBe` [13.0]

      it "calculates correctly for multiple output" $
        let
          nn = buildNNFromList (3, [LayerDefinition ReLU 2]) [1, 4, 2, 5, 3, 6, 7, 8]
          firstLayer = head (nnLayers nn)
        in (layerForward firstLayer [1, 1, 2]) `shouldBe` [16.0, 29.0]

      it "applies activation" $
        let
          nn = buildNNFromList (3, [LayerDefinition Sigmoid 2]) [1, -1, -1, 1, 1, -1, -1, 1]
          firstLayer = head (nnLayers nn)
        in (layerForward firstLayer [1, 1, 1]) `shouldBe` [0.5, 0.5]
