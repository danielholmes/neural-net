import Test.Hspec
import NeuralNet.NetSpec
import NeuralNet.LayerSpec
import NeuralNet.MatrixSpec
import NeuralNet.ActivationSpec
import NeuralNet.TrainSpec
import NeuralNet.ExampleSpec
import System.Random

main :: IO ()
main = do
  generator <- getStdGen
  hspec $ do
    matrixSpec
    activationSpec
    exampleSpec
    layerSpec
    netSpec generator
    trainSpec
