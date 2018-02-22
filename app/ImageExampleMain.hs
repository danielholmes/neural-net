module ImageExampleMain (main) where

import NeuralNet.Example
import NeuralNet.Activation
import NeuralNet.Net
import NeuralNet.Problem
--import NeuralNet.External
import NeuralNet.UI
import System.Random.Shuffle
import System.Random
import System.Directory
import Codec.Picture
import qualified Data.Vector.Storable as Vector


main :: IO ()
main = do
  problem <- loadProblem
  --weights <- createStdGenWeightsStream
  let weights = repeat 0
  uiRunProblem problem weights

loadProblem :: IO Problem
loadProblem = do
  fullSet <- loadImageSet
  let (trainSet, testSet) = splitExampleSet 0.8 fullSet
  let nnDef = createLogRegDefinition (exampleSetN trainSet) Sigmoid
  return (createProblem nnDef trainSet testSet 0.002 20)

loadImageSet :: IO ExampleSet
loadImageSet = do
  yes <- loadImageExamples "examples/cat-no-cat/cats" 1
  no <- loadImageExamples "examples/cat-no-cat/non-cats" 0
  gen <- getStdGen
  let examples = yes ++ no
  let orderedExamples = shuffle' (yes ++ no) (length examples) gen
  return (createExampleSet orderedExamples)

loadImageExamples :: FilePath -> Double -> IO [Example]
loadImageExamples p y = do
  names <- listDirectory p
  mapM (\n -> loadImageExample y (p ++ "/" ++ n)) (filter (\n -> head n /= '.') names)

loadImageExample :: Double -> FilePath -> IO Example
loadImageExample y f = do
  imageResult <- readImage f
  case imageResult of
    Left message   -> error ("Error on " ++ f ++ ": " ++ message)
    Right rawImage -> do
      let image = convertRGB8 rawImage
      let pixels = map (\p -> fromIntegral p / 255.0) (Vector.toList (imageData image))
      return (pixels, y)
