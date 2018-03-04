module NeuralNet.Net (
  NeuralNetDefinition,
  LayerDefinition (LayerDefinition),
  WeightsStream,
  NeuralNet,
  initNN,
  buildNNFromList,
  nnLayers,
  nnNumInputs,
  nnForward,
  nnForwardSet,
  isExampleSetCompatibleWithNN,
  isExampleSetCompatibleWithNNDef,
  updateNNLayers,
  createLogRegDefinition
) where

import NeuralNet.Matrix
import NeuralNet.Activation
import NeuralNet.Layer
import NeuralNet.Example
import Numeric.LinearAlgebra

type NumNeurons = Int
type NumInputs = Int
type NeuralNetDefinition = (NumInputs, [LayerDefinition])

type WeightsStream = [Double]

data LayerDefinition = LayerDefinition Activation NumNeurons
  deriving (Show, Eq)

data NeuralNet = NeuralNet [NeuronLayer]
  deriving (Show, Eq)

createLogRegDefinition :: NumInputs -> Activation -> NeuralNetDefinition
createLogRegDefinition n a = (n, [LayerDefinition a 1])

initNN :: WeightsStream -> NeuralNetDefinition -> NeuralNet
initNN _ (_, []) = error "No layer definitions provided"
initNN g (numInputs, layerDefs)
  | numInputs > 0 = NeuralNet (fst (initNeuronLayers g numInputs layerDefs))
  | otherwise     = error "Need positive num inputs"

initNeuronLayers :: WeightsStream -> Int -> [LayerDefinition] -> ([NeuronLayer], WeightsStream)
initNeuronLayers weightsStream _ [] = ([], weightsStream)
initNeuronLayers weightsStream numInputs (LayerDefinition a numNeurons : ds) = (NeuronLayer a w b : ls, ws2)
  where
    numSeeds = numInputs * numNeurons
    (weights, ws) = splitAt numSeeds weightsStream
    w = scale (sqrt (2.0 / fromIntegral numInputs)) ((numNeurons >< numInputs) weights)
    -- * sqrt (2 / numInputs) should be parameterisable
    b = col (replicate numNeurons 0)
    (ls, ws2) = initNeuronLayers ws numNeurons ds

buildNNFromList :: NeuralNetDefinition -> [Double] -> NeuralNet
buildNNFromList def@(numInputs, layerDefs) list
  | listSize /= defSize = error ("Wrong list size. Expected " ++ show defSize ++ " got " ++ show listSize)
  | otherwise           = NeuralNet (buildLayersFromList numInputs layerDefs list)
    where
      defSize = definitionToNNSize def
      listSize = length list

-- TODO: nnToList and include back and forward of buildNNFromList to ensure works

definitionToNNSize :: NeuralNetDefinition -> Int
definitionToNNSize (numInputs, layerDefs) = defToSizeStep numInputs layerDefs
  where
    defToSizeStep :: Int -> [LayerDefinition] -> Int
    defToSizeStep _ [] = 0
    defToSizeStep stepInputs (LayerDefinition _ numNeurons:ds) = (stepInputs * numNeurons) + numNeurons + defToSizeStep numNeurons ds

buildLayersFromList :: Int -> [LayerDefinition] -> [Double] -> [NeuronLayer]
buildLayersFromList _ [] _ = []
buildLayersFromList numInputs (LayerDefinition a numNeurons:ds) xs = layer : (buildLayersFromList numNeurons ds restXs)
  where
    numWeights = numInputs * numNeurons
    (layerXs, afterLayerXs) = splitAt numWeights xs
    (bs, restXs) = splitAt numNeurons afterLayerXs
    layer = NeuronLayer a ((numNeurons><numInputs) layerXs) (col bs)

nnLayers :: NeuralNet -> [NeuronLayer]
nnLayers (NeuralNet layers) = layers

nnForward :: NeuralNet -> [Double] -> [Double]
nnForward nn inputs = concat (toLists (forwardPropA (last setResult)))
  where
    set = createExampleSet [(inputs, error "Undefined")]
    setResult = nnForwardSet nn set

nnForwardSet :: NeuralNet -> ExampleSet -> [ForwardPropStep]
nnForwardSet nn examples = reverse foldResult
  where
    x = exampleSetX examples :: Matrix Double
    input = createInputForwardPropStep x
    foldResult = foldl (\i l -> layerForwardSet l (forwardPropA (head i)) : i) [input] (nnLayers nn)

nnNumInputs :: NeuralNet -> Int
nnNumInputs = layerNumInputs . head . nnLayers

isExampleSetCompatibleWithNNDef :: ExampleSet -> NeuralNetDefinition -> Bool
isExampleSetCompatibleWithNNDef examples def = isExampleSetCompatibleWithNN examples emptyNN
  where
    nums = replicate (definitionToNNSize def) 0
    emptyNN = buildNNFromList def nums

isExampleSetCompatibleWithNN :: ExampleSet -> NeuralNet -> Bool
isExampleSetCompatibleWithNN examples nn = nnNumInputs nn == exampleSetN examples

updateNNLayers :: NeuralNet -> [NeuronLayer] -> NeuralNet
updateNNLayers nn newLayers
  | layersShapes layers /= layersShapes newLayers = error "Must provide same shaped layers"
  | otherwise                                     = NeuralNet newLayers
    where
      layersShapes :: [NeuronLayer] -> [((Int, Int), (Int, Int))]
      layersShapes sLayers = map (\l -> (dims (layerW l), dims (layerB l))) sLayers
      layers = nnLayers nn
