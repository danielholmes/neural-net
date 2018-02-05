import Test.Hspec
import NeuralNet.NeuralNet
import NeuralNet.Activation
import Control.Exception (evaluate)
import Data.Matrix
import Data.List
import System.Random

matrixSize :: Matrix Double -> (Int, Int)
matrixSize m = (nrows m, ncols m)

allUnique :: (Eq a) => Matrix a -> Bool
allUnique m = length (nub list) == length list
  where list = toList m

main :: IO ()
main = do
  g <- getStdGen
  hspec $ do
      describe "buildNet" $ do
          it "returns correctly for small definition" $
              let
                def = [LayerDefinition ReLU 5, LayerDefinition Tanh 1, LayerDefinition Sigmoid 2]
                nn = buildNet g 8 def
                layers = nnLayers nn
              in do
                layerActivation (layers!!0) `shouldBe` ReLU
                layerActivation (layers!!1) `shouldBe` Tanh
                layerActivation (layers!!2) `shouldBe` Sigmoid
                matrixSize (layerW (layers!!0)) `shouldBe` (8, 5)
                matrixSize (layerW (layers!!1)) `shouldBe` (5, 1)
                matrixSize (layerW (layers!!2)) `shouldBe` (1, 2)
                layerW (layers!!0) `shouldSatisfy` allUnique
                layerW (layers!!1) `shouldSatisfy` allUnique
                layerW (layers!!2) `shouldSatisfy` allUnique

          it "returns correctly for single layer definition" $
              let
                def = [LayerDefinition ReLU 5]
                nn = buildNet g 6 def
                layers = nnLayers nn
              in do
                layerActivation (layers!!0) `shouldBe` ReLU
                matrixSize (layerW (layers!!0)) `shouldBe` (6, 5)

          it "throws exception for no layers" $
              evaluate (buildNet g 6 []) `shouldThrow` anyException

          it "throws exception for no inputs" $
              evaluate (buildNet g 0 [LayerDefinition ReLU 5]) `shouldThrow` anyException

      describe "parseMessageWithType" $ do
          it "returns Unknown for random" $
              True `shouldBe` True

          it "returns Unknown for no timestamp" $
              False `shouldBe` False
