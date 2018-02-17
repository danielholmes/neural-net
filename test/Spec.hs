import Test.Hspec
import NeuralNet.NetSpec
import NeuralNet.LayerSpec
import NeuralNet.MatrixSpec
import NeuralNet.ActivationSpec
import NeuralNet.TrainSpec
import NeuralNet.ExampleSpec
import NeuralNet.CostSpec
import System.Random

main :: IO ()
main = do
  generator <- getStdGen
  hspec $ do
    matrixSpec
    activationSpec
    exampleSpec
    costSpec
    layerSpec
    netSpec generator
    trainSpec
