module NeuralNet.Net (
  LayerDefinition (LayerDefinition),
  NeuralNet,
  initNN,
  buildNNFromList,
  nnLayers,
  nnNumInputs,
  nnForward,
  nnForwardSet,
  isExampleSetCompatibleWithNN
) where

import System.Random
import Data.Matrix
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
initNeuronLayers g numInputs (LayerDefinition a numNeurons : ds) = (NeuronLayer a w (replicate numNeurons 0) : ls, newG2)
  where
    foldStep :: Int -> ([Double], StdGen) -> ([Double], StdGen)
    foldStep _ (accu, foldG) = (d:accu, newFoldG)
      where (d, newFoldG) = randomR (0.0, 1.0) foldG
    -- TODO: Look into replicateM instead of this

    seeds = [1..(numInputs * numNeurons)]
    (nums, newG) = foldr foldStep ([], g) seeds
    w = fromList numInputs numNeurons nums
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
    layer = NeuronLayer a (fromList numInputs numNeurons layerXs) bs

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

isExampleSetCompatibleWithNN :: ExampleSet -> NeuralNet -> Bool
isExampleSetCompatibleWithNN examples nn = nnNumInputs nn == exampleSetN examples
