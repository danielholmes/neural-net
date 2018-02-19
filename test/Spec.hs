import Test.Hspec
import NeuralNet.NetSpec
import NeuralNet.LayerSpec
import NeuralNet.MatrixSpec
import NeuralNet.ActivationSpec
import NeuralNet.TrainSpec
import NeuralNet.ExampleSpec
import NeuralNet.CostSpec

main :: IO ()
main =
  hspec $ do
    matrixSpec
    activationSpec
    exampleSpec
    costSpec
    layerSpec
    netSpec
    trainSpec
