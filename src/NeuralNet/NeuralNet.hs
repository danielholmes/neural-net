module NeuralNet.NeuralNet (
  LayerDefinition (LayerDefinition),
  NeuralNet,
  NeuronLayer (NeuronLayer),
  buildNet,
  nnLayers,
  layerW,
  layerActivation
) where

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
-- W = layer x
buildNet :: NumInputs -> [LayerDefinition] -> NeuralNet
buildNet _ [] = error "No layer definitions provided"
buildNet numInputs layerDefs
  | numInputs > 0 = NeuralNet (buildNeuronLayers numInputs layerDefs)
  | otherwise     = error "Need positive num inputs"

buildNeuronLayers :: Int -> [LayerDefinition] -> [NeuronLayer]
buildNeuronLayers _ [] = []
buildNeuronLayers numInputs (LayerDefinition a numNeurons:ds) = (NeuronLayer a w 0) : buildNeuronLayers numNeurons ds
  where w = zero numInputs numNeurons

nnLayers :: NeuralNet -> [NeuronLayer]
nnLayers (NeuralNet layers) = layers

layerW :: NeuronLayer -> Matrix Double
layerW (NeuronLayer _ w _) = w

layerActivation :: NeuronLayer -> Activation
layerActivation (NeuronLayer a _ _) = a
