module NeuralNet.NeuralNet (
  NeuralNetDefinition,
  NeuralNet,
  buildNet
) where

import NeuralNet.Activation

type NumNeurons = Int

data LayerDefinition = LayerDefinition Activation NumNeurons
  deriving (Show, Eq)

data NeuralNetDefinition = NeuralNetDefinition [LayerDefinition]
  deriving (Show, Eq)

data NeuronLayer = NeuronLayer Activation Double

data NeuralNet = NeuralNet [NeuronLayer]


buildNet :: NeuralNetDefinition -> NeuralNet
buildNet (NeuralNetDefinition layerDefs) = NeuralNet (buildNeuronLayers layerDefs)

buildNeuronLayers :: [LayerDefinition] -> [NeuronLayer]
buildNeuronLayers [] = []
buildNeuronLayers (LayerDefinition a _:ds) = (NeuronLayer a 0) : buildNeuronLayers ds
