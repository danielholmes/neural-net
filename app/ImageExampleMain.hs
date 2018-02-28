{-# LANGUAGE DeriveDataTypeable #-}
{-# OPTIONS_GHC -fno-cse #-}
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
import System.IO ()
import System.Console.CmdArgs


data RunOptions = RunOptions {yesDir :: FilePath
                             ,noDir :: FilePath
                             ,constInitWeights :: Bool
                             ,numIterations :: Int
                             ,learningRate :: Double}
  deriving (Eq, Show, Data, Typeable)

optionsDef :: RunOptions
optionsDef = RunOptions
  {yesDir = def &= typ "YESDIR" &= argPos 0
  ,noDir = def &= typ "NODIR" &= argPos 1
  ,constInitWeights = def &= name "c" &= name "const-init-weights" &= help "Use 0 for initial weights"
  ,numIterations = 250 &= name "i" &= name "num-iterations" &= help "Number of training iterations"
  ,learningRate = 0.005 &= name "l" &= name "learning-rate" &= help "Learning rate (alpha)"} &=
  help "Simple Neural Net trainer" &=
  summary "NeuralNet v0.0.0, (C) Daniel Holmes"

main :: IO ()
main = do
  options <- cmdArgs optionsDef
  problem <- loadProblem options
  --weights <- createStdGenWeightsStream
  let weights = repeat 0
  uiRunProblem problem weights

loadProblem :: RunOptions -> IO Problem
loadProblem options = do
  fullSet <- loadImageSet options
  let (trainSet, testSet) = splitExampleSet 0.8 fullSet
  let nnDef = createLogRegDefinition (exampleSetN trainSet) Sigmoid
  return (createProblem nnDef trainSet testSet (learningRate options) (numIterations options))

loadImageSet :: RunOptions -> IO ExampleSet
loadImageSet options = do
  yes <- loadImageExamples (yesDir options) 1
  no <- loadImageExamples (noDir options) 0
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
