module NeuralNet.NeuralNet (
  LayerDefinition (LayerDefinition),
  NeuralNet,
  NeuronLayer (NeuronLayer),
  initNN,
  buildNNFromList,
  nnLayers,
  nnForward,
  layerW,
  layerB,
  layerActivation
) where

import System.Random
import Data.Matrix
import NeuralNet.Activation

type NumNeurons = Int

type NeuralNetDefinition = (NumInputs, [LayerDefinition])

data LayerDefinition = LayerDefinition Activation NumNeurons
  deriving (Show, Eq)

data NeuronLayer = NeuronLayer Activation (Matrix Double) Double
  deriving (Show, Eq)

data NeuralNet = NeuralNet [NeuronLayer]
  deriving (Show, Eq)

type NumInputs = Int

initNN :: StdGen -> NeuralNetDefinition -> NeuralNet
initNN _ (_, []) = error "No layer definitions provided"
initNN g (numInputs, layerDefs)
  | numInputs > 0 = NeuralNet (fst (initNeuronLayers g numInputs layerDefs))
  | otherwise     = error "Need positive num inputs"

initNeuronLayers :: StdGen -> Int -> [LayerDefinition] -> ([NeuronLayer], StdGen)
initNeuronLayers g _ [] = ([], g)
initNeuronLayers g numInputs (LayerDefinition a numNeurons : ds) = (NeuronLayer a w 0 : ls, newG2)
  where
    foldStep :: Int -> ([Double], StdGen) -> ([Double], StdGen)
    foldStep _ (accu, foldG) = (d:accu, newFoldG)
      where (d, newFoldG) = randomR (0.0, 1.0) foldG

    seeds = [1..(numInputs * numNeurons)]
    (nums, newG) = foldr foldStep ([], g) seeds
    w = fromList numInputs numNeurons nums
    (ls, newG2) = initNeuronLayers newG numNeurons ds

buildNNFromList :: NeuralNetDefinition -> [Double] -> NeuralNet
buildNNFromList (numInputs, layerDefs) list = NeuralNet (buildLayersFromList numInputs layerDefs list)

buildLayersFromList :: Int -> [LayerDefinition] -> [Double] -> [NeuronLayer]
buildLayersFromList _ [] _ = []
buildLayersFromList numInputs ((LayerDefinition a numNeurons):ds) xs = layer : (buildLayersFromList numNeurons ds restXs)
  where
    size = numInputs * numNeurons
    (layerXs, b:restXs) = splitAt size xs
    layer = NeuronLayer a (fromList numInputs numNeurons layerXs) b

nnLayers :: NeuralNet -> [NeuronLayer]
nnLayers (NeuralNet layers) = layers

nnForward :: NeuralNet -> [Double] -> [Double]
nnForward _ inputs = inputs

layerW :: NeuronLayer -> Matrix Double
layerW (NeuronLayer _ w _) = w

layerB :: NeuronLayer -> Double
layerB (NeuronLayer _ _ b) = b

layerActivation :: NeuronLayer -> Activation
layerActivation (NeuronLayer a _ _) = a
