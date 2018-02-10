module NeuralNet.ExampleSpec (exampleSpec) where

import Test.Hspec
import Data.Matrix
import NeuralNet.Example
import Control.Exception (evaluate)

exampleSpec :: SpecWith ()
exampleSpec =
  describe "NeuralNet.Example" $ do
    describe "createExampleSet" $ do
      it "creates x correctly" $
        let
          examples = [([1, 2, 3], 4), ([5, 6, 7], 8)]
          set = createExampleSet examples
        in exampleSetX set `shouldBe` (fromList 3 2 [1, 5, 2, 6, 3, 7])

      it "creates y correctly" $
        let
          examples = [([1, 2, 3], 4), ([5, 6, 7], 8)]
          set = createExampleSet examples
        in exampleSetY set `shouldBe` (fromList 2 1 [4, 8])

      it "throws error for misaligned example sizes" $
        let examples = [([1..10], 4), ([1..2], 8)]
        in evaluate (createExampleSet examples) `shouldThrow` anyErrorCall

      it "throws error for empty example" $
        let examples = [([], 4)]
        in evaluate (createExampleSet examples) `shouldThrow` anyErrorCall

      it "throws error for empty examples" $
        evaluate (createExampleSet []) `shouldThrow` anyErrorCall
