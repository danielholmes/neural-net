module NeuralNet.Net (
  NeuralNetDefinition,
  LayerDefinition (LayerDefinition),
  NeuralNet,
  initNN,
  buildNNFromList,
  nnLayers,
  nnNumInputs,
  nnForward,
  nnForwardSet,
  isExampleSetCompatibleWithNN,
  isExampleSetCompatibleWithNNDef,
  updateNNLayers
) where

import System.Random
import Data.Matrix
import NeuralNet.Matrix
import NeuralNet.Activation
import NeuralNet.Layer
import NeuralNet.Example

type NumNeurons = Int
type NumInputs = Int

type NeuralNetDefinition = (NumInputs, [LayerDefinition])

data LayerDefinition = LayerDefinition Activation NumNeurons
  deriving (Show, Eq)

data NeuralNet = NeuralNet [NeuronLayer]
  deriving (Show, Eq)

initNN :: StdGen -> NeuralNetDefinition -> NeuralNet
initNN _ (_, []) = error "No layer definitions provided"
initNN g (numInputs, layerDefs)
  | numInputs > 0 = NeuralNet (fst (initNeuronLayers g numInputs layerDefs))
  | otherwise     = error "Need positive num inputs"

initNeuronLayers :: StdGen -> Int -> [LayerDefinition] -> ([NeuronLayer], StdGen)
initNeuronLayers g _ [] = ([], g)
initNeuronLayers g numInputs (LayerDefinition a numNeurons : ds) = (NeuronLayer a w (fromList numNeurons 1 (replicate numNeurons 0)) : ls, newG2)
  where
    foldStep :: Int -> ([Double], StdGen) -> ([Double], StdGen)
    foldStep _ (accu, foldG) = (d:accu, newFoldG)
      where (d, newFoldG) = randomR (0.0, 1.0) foldG
    -- TODO: Look into replicateM instead of this

    seeds = [1..(numInputs * numNeurons)]
    (nums, newG) = foldr foldStep ([], g) seeds
    w = fromList numNeurons numInputs nums
    (ls, newG2) = initNeuronLayers newG numNeurons ds

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
    defToSizeStep stepInputs ((LayerDefinition _ numNeurons):ds) = (stepInputs * numNeurons) + numNeurons + defToSizeStep numNeurons ds

buildLayersFromList :: Int -> [LayerDefinition] -> [Double] -> [NeuronLayer]
buildLayersFromList _ [] _ = []
buildLayersFromList numInputs ((LayerDefinition a numNeurons):ds) xs = layer : (buildLayersFromList numNeurons ds restXs)
  where
    size = numInputs * numNeurons
    (layerXs, afterLayerXs) = splitAt size xs
    (bs, restXs) = splitAt numNeurons afterLayerXs
    layer = NeuronLayer a (fromList numNeurons numInputs layerXs) (fromList (length bs) 1 bs)

nnLayers :: NeuralNet -> [NeuronLayer]
nnLayers (NeuralNet layers) = layers

nnForward :: NeuralNet -> [Double] -> [Double]
nnForward nn inputs = toList (forwardPropA (last setResult))
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
