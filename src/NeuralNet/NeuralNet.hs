module NeuralNet.NeuralNet (
  LayerDefinition (LayerDefinition),
  NeuralNet,
  NeuronLayer (NeuronLayer),
  buildNet,
  nnLayers,
  layerW,
  layerActivation
) where

import System.Random
import Data.Matrix
import NeuralNet.Activation

type NumNeurons = Int

data LayerDefinition = LayerDefinition Activation NumNeurons
  deriving (Show, Eq)

data NeuronLayer = NeuronLayer Activation (Matrix Double) Double
  deriving (Show, Eq)

data NeuralNet = NeuralNet [NeuronLayer]
  deriving (Show, Eq)

type NumInputs = Int

buildNet :: StdGen -> NumInputs -> [LayerDefinition] -> NeuralNet
buildNet _ _ [] = error "No layer definitions provided"
buildNet g numInputs layerDefs
  | numInputs > 0 = NeuralNet (fst (buildNeuronLayers g numInputs layerDefs))
  | otherwise     = error "Need positive num inputs"

buildNeuronLayers :: StdGen -> Int -> [LayerDefinition] -> ([NeuronLayer], StdGen)
buildNeuronLayers g _ [] = ([], g)
buildNeuronLayers g numInputs (LayerDefinition a numNeurons:ds) = (NeuronLayer a w 0 : ls, newG2)
  where
    foldStep :: Int -> ([Double], StdGen) -> ([Double], StdGen)
    foldStep _ (accu, foldG) = (d:accu, newFoldG)
      where (d, newFoldG) = randomR (0.0, 1.0) foldG

    seeds = [1..(numInputs * numNeurons)]
    (nums, newG) = foldr foldStep ([], g) seeds
    w = fromList numInputs numNeurons nums
    (ls, newG2) = buildNeuronLayers newG numNeurons ds

nnLayers :: NeuralNet -> [NeuronLayer]
nnLayers (NeuralNet layers) = layers

layerW :: NeuronLayer -> Matrix Double
layerW (NeuronLayer _ w _) = w

layerActivation :: NeuronLayer -> Activation
layerActivation (NeuronLayer a _ _) = a
